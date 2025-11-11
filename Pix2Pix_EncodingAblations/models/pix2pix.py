from .backbones import GeneratorResUNet, GeneratorUNet, GeneratorUNet3Plus
from .discriminators import Discriminator_286x286, Discriminator_70x70, Discriminator_30x30
from utils import GenerateImages

import tensorflow as tf
import os
JOB_ID = os.environ.get("SLURM_JOB_ID")

import datetime
import time

class Pix2Pix():
    '''
    The Pix2Pix class implements the Pix2Pix image-to-image translation model.
    The Generator is responsible for generating images from input images, 
    while the Discriminator is responsible for distinguishing between real and generated images.
    Args:
        backbone (str): The backbone architecture to use for the generator.
        size (tuple): The input size of the images (height, width).
        hadamard (bool): Whether to use Hadamard product or not.
        num_classes (int): The number of classes for segmentation.
        code_size (int): The size of the code vector.
        mode (str): The mode to run the model in ('train' or 'test').
        path (str): The path to the log folder.
        ignore_index (int): The index to ignore for loss calculation.
        like_paper (bool): Whether to use the paper's training strategy or the code's one.
    '''
    def __init__(self, 
                 backbone: str = 'unet', 
                 size: tuple = (512,1024), 
                 hadamard: bool = False, 
                 num_classes: int = 19, 
                 code_size: int = 32, 
                 mode: str = 'train', 
                 path: str = '', 
                 ignore_index: int = 255,
                 like_paper: bool = False):

        '''
        Build generator model and discriminator model.
        Already handles Hadamard product or baseline one-hot encoding.
        The shape of the generator is (None, None, None, 3), because of the input image shape that differs in training and testing.
        The shape of the discriminator is equal to the cropped image shape.
        '''
        if backbone == 'unet':
            self.generator = GeneratorUNet(num_classes = num_classes, code_size = code_size, shape=(size[0], size[1], 3), hadamard = hadamard)
        elif backbone == 'resunet':
            self.generator = GeneratorResUNet(num_classes = num_classes, code_size = code_size, resunet = True, shape=(None,None,3), hadamard = hadamard)
        else:
            self.generator = GeneratorUNet3Plus(num_classes = num_classes, code_size = code_size, shape=(None,None,3), hadamard = hadamard)

        self.generator.build(input_shape=(None, size[0], size[1], 3))

        self.discriminator = Discriminator_30x30([size[0], size[1], 3], [size[0], size[1], 1])
        #self.discriminator = Discriminator_70x70([size[0], size[1], 3], [size[0], size[1], 1])
        #self.discriminator = Discriminator_286x286([size[0], size[1], 3], [size[0], size[1], 1])
        self.discriminator.build(input_shape=[(None, size[0], size[1], 3),
                                       (None, size[0], size[1], num_classes)])

        '''
        Initializing class variables
        '''
        self.like_paper = like_paper
        self.ignore_index = ignore_index
        self.code_size = code_size
        self.hadamard = hadamard
        self.num_classes = num_classes
        self.path = path

        if hadamard:
            self.gen_model_name = 'gen_hadamard'
            self.dis_model_name = 'dis_hadamard'
        else:
            self.gen_model_name = 'gen_baseline'
            self.dis_model_name = 'dis_baseline'

        '''
        Load pre-trained weights for the generator model if in test mode.
        '''
        if mode == 'test':
            gen_path = os.path.join(path, self.gen_model_name + '.h5')
            self.generator.load_weights(gen_path)

        '''
        Define metrics for evaluation.
        Mean Intersection over Union (mIoU) and Pixel Accuracy (PixAcc).
        '''
        self.miou_metric   = tf.keras.metrics.MeanIoU(num_classes=self.num_classes)
        self.pixacc_metric = tf.keras.metrics.Accuracy()

        '''
        Set up TensorBoard logging by creating log directories and summary writers.
        The summary writers will log training losses and validation metrics.
        For each training step, the training losses will be logged, and at the end of each validation epoch, the validation metrics will be logged.
        Can later be visualized using TensorBoard.
        '''
        self.log_dir = os.path.join(path, 'logs', JOB_ID)
        os.makedirs(self.log_dir, exist_ok=True)
        self.train_summary_writer = tf.summary.create_file_writer(self.log_dir + "/train_summary")
        self.validation_summary_writer = tf.summary.create_file_writer(self.log_dir + "/val_summary")

    @tf.function
    def inference(self, img): 
        '''
        Run inference on a single image.
        Args:
            img (tf.Tensor): The input image tensor.
        Returns:
            mask (tf.Tensor): The generated (Soft)mask.
            probabilities (tf.Tensor): The pixel-wise class probabilities.
            codes (tf.Tensor): The encoded representation of the input image.
        '''
        mask, probabilities, codes = self.generator(img, training=False)
        return mask, probabilities, codes

    @tf.function
    def evaluation(self, dataset, step=0):
        '''
        Evaluate the model on a validation dataset.
        Args:
            dataset (tf.data.Dataset): The validation dataset.
            step (int): The current training step.
        Returns:
            dict: A dictionary containing the evaluation metrics.
        '''
        self.miou_metric.reset_state()
        self.pixacc_metric.reset_state()

        tf.print('-------------------------------------------------------------------------------')
        tf.print('Evaluation...')
        tf.print('-------------------------------------------------------------------------------')

        '''
        Calculating metrics through forward pass
        Metrics: mIoU, Pixel Accuracy
        '''
        for img, target in dataset:
            _, probabilities, _ = self.inference(img)
            pred_mask = tf.argmax(probabilities, axis=-1)
            target_mask  =  tf.squeeze(target, axis=-1)

            if self.ignore_index is not None:
                valid_pixel     = tf.not_equal(target_mask, self.ignore_index)
                pred_mask       = tf.boolean_mask(pred_mask, valid_pixel)
                target_mask     = tf.boolean_mask(target_mask, valid_pixel)
            else:
                pred_mask       = tf.reshape(pred_mask, [-1])
                target_mask     = tf.reshape(target_mask, [-1])

            self.miou_metric.update_state(target_mask, pred_mask)
            self.pixacc_metric.update_state(target_mask, pred_mask)

        '''
        Calculating class-wise IoU with respect to the confusion matrix of the mIoU
        '''
        confusion_matrix  = tf.cast(self.miou_metric.total_cm, tf.float32)           # [C,C], Zeilen: GT, Spalten: Pred
        tp_per_class      = tf.linalg.diag_part(confusion_matrix)                    # TP pro Klasse
        gt_per_class      = tf.reduce_sum(confusion_matrix, axis=1)                  # GT pro Klasse (TP+FN)
        pred_per_class    = tf.reduce_sum(confusion_matrix, axis=0)                  # Pred pro Klasse (TP+FP)
        denom             = gt_per_class + pred_per_class - tp_per_class
        per_class_iou     = tp_per_class / tf.maximum(denom, 1e-7)

        '''
        Logging in summary writer for evaluation metrics
        '''
        with self.validation_summary_writer.as_default():
            tf.summary.scalar('mIoU',      self.miou_metric.result(), step=step)
            tf.summary.scalar('pixAcc',    self.pixacc_metric.result(), step=step)
            tf.summary.histogram('class_IoU', per_class_iou, step=step)

        '''
        Printing Results:
        '''
        tf.print('Evaluation Metric Results:')
        tf.print(tf.strings.format('mean IoU:   {}', (tf.strings.as_string(self.miou_metric.result(), precision=3),)))
        tf.print(tf.strings.format('Pixel Acc.: {}', (tf.strings.as_string(self.pixacc_metric.result(), precision=3),)))
        tf.print('-------------------------------------------------------------------------------')
        tf.print('class-wise IoU:')
        tf.print(tf.strings.format('Road:               {}', (tf.strings.as_string(per_class_iou[0], precision=3),)))
        tf.print(tf.strings.format('Sidewalk:           {}', (tf.strings.as_string(per_class_iou[1], precision=3),)))
        tf.print(tf.strings.format('Building:           {}', (tf.strings.as_string(per_class_iou[2], precision=3),)))
        tf.print(tf.strings.format('Wall:               {}', (tf.strings.as_string(per_class_iou[3], precision=3),)))
        tf.print(tf.strings.format('Fence:              {}', (tf.strings.as_string(per_class_iou[4], precision=3),)))
        tf.print(tf.strings.format('Pole:               {}', (tf.strings.as_string(per_class_iou[5], precision=3),)))
        tf.print(tf.strings.format('Traffic Light:      {}', (tf.strings.as_string(per_class_iou[6], precision=3),)))
        tf.print(tf.strings.format('Traffic Sign:       {}', (tf.strings.as_string(per_class_iou[7], precision=3),)))
        tf.print(tf.strings.format('Vegetation:         {}', (tf.strings.as_string(per_class_iou[8], precision=3),)))
        tf.print(tf.strings.format('Terrain:            {}', (tf.strings.as_string(per_class_iou[9], precision=3),)))
        tf.print(tf.strings.format('Sky:                {}', (tf.strings.as_string(per_class_iou[10], precision=3),)))
        tf.print(tf.strings.format('Person:             {}', (tf.strings.as_string(per_class_iou[11], precision=3),)))
        tf.print(tf.strings.format('Rider:              {}', (tf.strings.as_string(per_class_iou[12], precision=3),)))
        tf.print(tf.strings.format('Car:                {}', (tf.strings.as_string(per_class_iou[13], precision=3),)))
        tf.print(tf.strings.format('Truck:              {}', (tf.strings.as_string(per_class_iou[14], precision=3),)))
        tf.print(tf.strings.format('Bus:                {}', (tf.strings.as_string(per_class_iou[15], precision=3),)))
        tf.print(tf.strings.format('Train:              {}', (tf.strings.as_string(per_class_iou[16], precision=3),)))
        tf.print(tf.strings.format('Motorcycle:         {}', (tf.strings.as_string(per_class_iou[17], precision=3),)))
        tf.print(tf.strings.format('Bicycle:            {}', (tf.strings.as_string(per_class_iou[18], precision=3),)))
        tf.print('-------------------------------------------------------------------------------')


        return {
            "mIoU": self.miou_metric.result(),      # tf.Tensor (scalar)
            "pixAcc": self.pixacc_metric.result(),  # tf.Tensor (scalar)
            "class_IoU": per_class_iou,             # tf.Tensor (C)
        }
    
    @tf.function
    def discriminator_loss(self, disc_real_output, disc_generated_output):
        '''
        Calculate the discriminator loss.
        Args:
            disc_real_output (tf.Tensor): The discriminator output for real images.
            disc_generated_output (tf.Tensor): The discriminator output for generated images.
        Return: 
            tf.Tensor: The total discriminator loss.
        '''
        bce_loss         = tf.keras.losses.BinaryCrossentropy(from_logits=True)

        real_loss        = bce_loss(tf.ones_like(disc_real_output), disc_real_output)
        generated_loss   = bce_loss(tf.zeros_like(disc_generated_output), disc_generated_output)

        return real_loss + generated_loss
    
    @tf.function
    def generator_loss(self, discriminator_output, probabilities, target, codes=None, target_codes=None):
        '''
        Calculate the generator loss.
        Args:
            discriminator_output (tf.Tensor): The discriminator output for generated images.
            probabilities (tf.Tensor): The predicted class probabilities.
            target (tf.Tensor): The ground truth target labels.
            codes (tf.Tensor, optional): The generated codes.
            target_codes (tf.Tensor, optional): The target codes.
        Return:
            total_loss (tf.Tensor): The total generator loss.
            gan_loss (tf.Tensor): The GAN loss.
            l1_loss (tf.Tensor): The L1 loss.
            sparse_ce_loss (tf.Tensor): The sparse categorical crossentropy loss.
            gini_loss (tf.Tensor): The Gini loss.
        '''
        LAMBDA = [1, 1000, 100, 250]        # weighting of different losses

        '''
        Definition of loss functions
        Reduction methods set to 'none', due to the ignore index values
        '''
        bce_loss_func = tf.keras.losses.BinaryCrossentropy(
            from_logits=True, 
            reduction='none'
        )    

        sparse_ce_loss_func = tf.keras.losses.SparseCategoricalCrossentropy(
            from_logits=False, 
            ignore_class=255, 
            reduction='none'
        )

        '''
        Generation of ignore mask in different resolutions/shapes
        '''
        ignore_mask = tf.cast(tf.not_equal(target, 255), tf.float32)
        ignore_mask_squeezed = tf.squeeze(ignore_mask, axis=-1)
        ignore_mask_sum = tf.maximum(tf.reduce_sum(ignore_mask), 1)
        ignore_mask_discriminator = tf.squeeze(tf.image.resize(ignore_mask, (tf.shape(discriminator_output)[1], tf.shape(discriminator_output)[2]), method='area'), axis=-1)
        ignore_mask_discriminator_sum = tf.maximum(tf.reduce_sum(ignore_mask_discriminator), 1)

        '''
        Calculate the losses in respect to the ignore mask.
        '''
        gan_loss = bce_loss_func(
            tf.ones_like(discriminator_output), 
            discriminator_output
        )
        gan_loss = tf.reduce_sum(gan_loss * ignore_mask_discriminator) / ignore_mask_discriminator_sum

        target_one_hot = tf.one_hot(tf.squeeze(target, -1), depth=probabilities.shape[-1], dtype=tf.float32)
        l1_loss = tf.reduce_sum(tf.abs(target_one_hot - probabilities), axis=-1)
        l1_loss = tf.reduce_sum(l1_loss* ignore_mask_squeezed) / ignore_mask_sum

        gini_loss = tf.reduce_sum(tf.square(probabilities), axis=-1)
        gini_loss = - tf.reduce_sum(gini_loss * ignore_mask_squeezed) / ignore_mask_sum

        sparse_ce_loss = sparse_ce_loss_func(target, probabilities)
        sparse_ce_loss = tf.reduce_sum(sparse_ce_loss * ignore_mask_squeezed) / ignore_mask_sum

        '''
        Calculate the total loss.
        If like_paper is True, use the original weighting and losses from paper.
        Else use the weighting and losses based on code implementation.
        '''
        if self.like_paper:
            total_loss = LAMBDA[0]*gan_loss + LAMBDA[1]*sparse_ce_loss + LAMBDA[2]*l1_loss
        else:
            total_loss = gan_loss + LAMBDA[0]*l1_loss + LAMBDA[1]*sparse_ce_loss + LAMBDA[2]*gini_loss

        '''
        Calculate the codes loss, if hadamard is True.
        '''
        codes_loss = 0
        if self.hadamard:
            if self.like_paper:
                codes_loss = tf.reduce_sum(tf.abs(target_codes - codes), axis=-1)
            else:
                codes_loss = tf.math.squared_difference(codes, target_codes)
                codes_loss = tf.reduce_mean(codes_loss, axis=-1)

            codes_loss = tf.reduce_sum(codes_loss * ignore_mask_squeezed) / ignore_mask_sum

            total_loss += LAMBDA[3]*codes_loss

        return total_loss, gan_loss, l1_loss, sparse_ce_loss, codes_loss, gini_loss

    @tf.function
    def train_step(self, input_image, target, step, generator_optimizer, discriminator_optimizer, target_codes=None):
        '''
        Training step for the Pix2Pix model.
        Args:
            input_image: The input image tensor.
            target: The target image tensor.
            step: The current training step.
            generator_optimizer: The optimizer for the generator.
            discriminator_optimizer: The optimizer for the discriminator.
            target_codes: The target codes tensor (optional).
        Returns:
            losses: A dictionary containing the training losses.
        '''
        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
            mask, probablilites, codes = self.generator(input_image, training=True)

            # applying ignore values from target to discriminator input
            mask = tf.where(tf.equal(target, self.ignore_index), tf.cast(target, mask.dtype), mask)
            disc_real_output = self.discriminator([input_image, target], training=True)
            disc_generated_output = self.discriminator([input_image, mask], training=True)

            if target_codes != None:
                gen_total_loss, gen_gan_loss, gen_l1_loss, gen_ce_loss, gen_codes_loss, gen_gn_loss = self.generator_loss(disc_generated_output, probablilites, target, codes, target_codes)
            else:
                gen_total_loss, gen_gan_loss, gen_l1_loss, gen_ce_loss, gen_codes_loss, gen_gn_loss = self.generator_loss(disc_generated_output, probablilites, target)

            disc_loss  = self.discriminator_loss(disc_real_output, disc_generated_output)

        generator_gradients     = gen_tape.gradient(gen_total_loss, self.generator.trainable_variables)
        discriminator_gradients = disc_tape.gradient(disc_loss,     self.discriminator.trainable_variables)

        generator_optimizer.apply_gradients    (zip(generator_gradients,     self.generator.trainable_variables))
        discriminator_optimizer.apply_gradients(zip(discriminator_gradients, self.discriminator.trainable_variables))

        with self.train_summary_writer.as_default():
            tf.summary.scalar('gen_total_loss',    gen_total_loss,    step=step)
            tf.summary.scalar('gen_gan_loss',      gen_gan_loss,      step=step)
            tf.summary.scalar('gen_l1_loss',       gen_l1_loss,       step=step)
            tf.summary.scalar('gen_ce_loss',       gen_ce_loss,       step=step)
            tf.summary.scalar('gen_codes_loss',    gen_codes_loss, step=step)
            tf.summary.scalar('gen_gn_loss',       gen_gn_loss,       step=step)
            tf.summary.scalar('disc_loss',         disc_loss,         step=step)

        losses = {
            "gen_total_loss":     gen_total_loss,
            "gen_ce_loss":        gen_ce_loss,
            "gen_codes_loss":     gen_codes_loss,
            "gen_gn_loss":        gen_gn_loss,
            "gen_gan_loss":       gen_gan_loss,
            "gen_l1_loss":        gen_l1_loss,
            "disc_loss":          disc_loss
        }

        return losses

    def train(self, dataset_train, dataset_val, generator_optimizer, discriminator_optimizer, steps):
        '''
        Training loop for the Pix2Pix model.
        Saves checkpoints and logs.
        Args:
            dataset_train: The training dataset.
            dataset_val: The validation dataset.
            generator_optimizer: The optimizer for the generator.
            discriminator_optimizer: The optimizer for the discriminator.
            steps: The number of training steps.
        Returns:
            None
        '''
        image_generation_path = os.path.join(self.path, 'images')
        log_window = 50
        overall_time = time.time()
        batch_time = time.time()
        batch_avg_loss = {}

        for step, item in dataset_train.repeat().take(steps).enumerate():
            losses = {}
            if self.hadamard:
                x, y, z = item
                losses = self.train_step(x, y, step, generator_optimizer, discriminator_optimizer, z) 
            else:
                x, y = item
                losses = self.train_step(x, y, step, generator_optimizer, discriminator_optimizer) 

            for key, value in losses.items():
                if key not in batch_avg_loss:
                    batch_avg_loss[key] = 0.0
                batch_avg_loss[key] += tf.cast(value, tf.float32)

            if step % log_window == 0 and step != 0:
                for key in batch_avg_loss.keys():
                    batch_avg_loss[key] = batch_avg_loss[key] / log_window
                print(f'[{step}/{steps}]: \t\t Total Time ellapsed: {time.time()-overall_time:.2f} seconds. \t Batch Time: {time.time()-batch_time:.2f} seconds. AVG GEN Loss: {batch_avg_loss["gen_total_loss"].numpy()} \t AVG DISC Loss: {batch_avg_loss["disc_loss"].numpy()} \t Detailed losses(unweighted): CE Loss: {batch_avg_loss["gen_ce_loss"].numpy()} \t Codes Loss: {batch_avg_loss["gen_codes_loss"].numpy()} \t L1 Loss: {batch_avg_loss["gen_l1_loss"].numpy()} \t Gini Loss: {batch_avg_loss["gen_gn_loss"].numpy()}')
                batch_avg_loss.clear()
                batch_time = time.time()

            if step % 29750 == 0 and step != 0:
                evaluation_metrics = self.evaluation(dataset_val, step)
                GenerateImages(self.generator, x, y, image_generation_path, str(int(step)).zfill(5)+'.png')

        self.generator.save_weights(self.log_dir + '/' + self.gen_model_name +'.h5')
        self.discriminator.save_weights(self.log_dir + '/' + self.dis_model_name +'.h5')