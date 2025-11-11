import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
JOB_ID = os.environ.get("SLURM_JOB_ID")

import argparse
import absl.logging
absl.logging.set_verbosity(absl.logging.ERROR)

from models.datasets import CityscapesDataset
from models import Pix2Pix 

from utils import Pix2PixScheduler

import tensorflow as tf

def parse_args():
    p = argparse.ArgumentParser(description="Eval Hadamard pix2pix")

    p.add_argument("--hadamard", dest="hadamard", action="store_true",
                   help="Hadamard-Codes aktivieren")
    
    p.add_argument("--no-hadamard", dest="hadamard", action="store_false",
                   help="Hadamard-Codes deaktivieren")
    
    p.set_defaults(hadamard=True)

    p.add_argument("-b", "--batch-size", type=int, default=1,
                   help="Batchgröße")
    
    p.add_argument("-ep", "--epochs", type=int, default=1,
                   help="Anzahl der Epochen")

    p.add_argument("-c", "--codes-size", type=int, default=32,
                   help="Größe des Hadamard-Codewortes")
    
    p.add_argument("-n", "--num-classes", type=int, default=19,
                   help="Anzahl der Klassen")
    
    p.add_argument("-bb", "--backbone", type=str, default="unet_3plus",
                   help="Backbone-Architektur [unet_3plus, unet, resunet]")

    p.add_argument("-p", "--like-paper", dest="like_paper", action="store_true",
                   help="Wie im Papier beschrieben")

    p.set_defaults(like_paper=False)

    return p.parse_args()

def main():
    args = parse_args()

    MODE = 'train'

    ORIGINAL_SIZE = (286, 286)
    CROP_SIZE     = (256, 256)
    VAL_SIZE      = (256, 256)

    NUM_CLASSES   = args.num_classes
    CODES_SIZE    = args.codes_size
    HADAMARD      = args.hadamard
    BATCH_SIZE    = args.batch_size
    BACKBONE      = args.backbone
    EPOCHS        = args.epochs
    LIKE_PAPER    = args.like_paper

    INIT_LR      = 2e-4
    STEPS_PER_EP = 2975
    DECAY_START  = tf.math.round((EPOCHS * STEPS_PER_EP ) / 2)
    TOTAL_STEPS  = EPOCHS * STEPS_PER_EP
    DECAY_STEPS  = TOTAL_STEPS - DECAY_START

    main_folder         = os.getcwd()

    dataset_folder      = os.path.join(main_folder, "data/cityscapes")
    if HADAMARD:
        save_folder     = os.path.join(main_folder, "work_dirs", BACKBONE, "hadamard_test")
    else:
        CODES_SIZE      = NUM_CLASSES
        save_folder     = os.path.join(main_folder, "work_dirs", BACKBONE, "baseline_test")

    if LIKE_PAPER:
        save_folder     = os.path.join(save_folder, "like_paper")
    else:
        save_folder     = os.path.join(save_folder, "like_github")

    dataset = CityscapesDataset(dataset_folder, 
                                batch_size=BATCH_SIZE, 
                                image_shape=ORIGINAL_SIZE, 
                                crop_size=CROP_SIZE, 
                                validation_shape=VAL_SIZE, 
                                hadamard=HADAMARD, 
                                num_classes=NUM_CLASSES, 
                                codes_size=CODES_SIZE)

    train_ds = dataset['train']
    val_ds = dataset['val']
    test_ds = dataset['test']

    print("Dataset size loaded:", train_ds.cardinality().numpy())
    print("Dataset size loaded:", val_ds.cardinality().numpy())
    print("Dataset size loaded:", test_ds.cardinality().numpy())

    print('CROP-SIZE:', CROP_SIZE)

    model = Pix2Pix(backbone=BACKBONE, 
                    size=CROP_SIZE, 
                    hadamard=HADAMARD, 
                    num_classes=NUM_CLASSES, 
                    code_size=CODES_SIZE, 
                    mode=MODE, 
                    path=save_folder)

    if MODE == 'train':
        lr_schedule = Pix2PixScheduler(
            init_lr=2e-4,
            decay_start=DECAY_START,
            total_steps=TOTAL_STEPS,
            end_lr=0.0,
            power=1.0  
        )

        generator_optimizer     = tf.keras.optimizers.Adam(lr_schedule, beta_1=0.5)
        discriminator_optimizer = tf.keras.optimizers.Adam(lr_schedule, beta_1=0.5)

        for var in generator_optimizer.variables():
            var.assign(tf.zeros_like(var))

        for var in discriminator_optimizer.variables():
            var.assign(tf.zeros_like(var))

        model.train(train_ds, val_ds, generator_optimizer, discriminator_optimizer, TOTAL_STEPS)

    evaluation_metrics = model.evaluation(val_ds)


if __name__ == "__main__":
    main()