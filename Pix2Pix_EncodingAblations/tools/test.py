import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
JOB_ID = os.environ.get("SLURM_JOB_ID")

import argparse
import absl.logging
absl.logging.set_verbosity(absl.logging.ERROR)

from models.datasets import CityscapesDataset
from models import Pix2Pix 
from utils import GenerateImages

def parse_args():
    p = argparse.ArgumentParser(description="Eval Hadamard pix2pix")

    p.add_argument("--hadamard", dest="hadamard", action="store_true",
                   help="Hadamard-Codes activate")
    
    p.add_argument("--no-hadamard", dest="hadamard", action="store_false",
                   help="Hadamard-Codes deactivate")
    
    p.set_defaults(hadamard=True)

    p.add_argument("-c", "--codes-size", type=int, default=32,
                   help="Size of the Hadamard codewords")
    
    p.add_argument("-n", "--num-classes", type=int, default=19,
                   help="Number of segmentation classes")
    
    p.add_argument("-bb", "--backbone", type=str, default="unet_3plus",
                   help="Backbone-Architecture [unet_3plus, unet, resunet]")

    p.add_argument("-p", "--like-paper", dest="like_paper", action="store_true",
                   help="Losses described in the paper")

    p.set_defaults(like_paper=False)

    p.add_argument("-j", "--job-id", type=str, default=JOB_ID,
                   help="SLURM Job ID")

    return p.parse_args()


def main():
    args = parse_args()

    MODE = 'test'
    
    ORIGINAL_SIZE = (286, 286)
    CROP_SIZE     = (256, 256)
    VAL_SIZE      = (256, 256)

    NUM_CLASSES   = args.num_classes
    CODES_SIZE    = args.codes_size
    HADAMARD      = args.hadamard
    BACKBONE      = args.backbone
    LIKE_PAPER    = args.like_paper
    JOB_ID_TEST   = args.job_id

    main_folder     = os.getcwd()

    dataset_folder  = os.path.join(main_folder, "data/cityscapes")
    if HADAMARD:
        save_folder     = os.path.join(main_folder, "work_dirs", BACKBONE, "hadamard_test")
    else:
        CODES_SIZE = NUM_CLASSES
        save_folder     = os.path.join(main_folder, "work_dirs", BACKBONE, "baseline_test")

    if LIKE_PAPER:
        save_folder = os.path.join(save_folder, "like_paper", "logs", JOB_ID_TEST)
    else:
        save_folder = os.path.join(save_folder, "like_github", "logs", JOB_ID_TEST)

    save_imgs_folder = os.path.join(save_folder, "imgs")
    os.makedirs(save_imgs_folder, exist_ok=True)

    dataset = CityscapesDataset(dataset_folder, 
                                batch_size=1, 
                                image_shape=ORIGINAL_SIZE, 
                                crop_size=CROP_SIZE,
                                validation_shape=VAL_SIZE,
                                hadamard=HADAMARD, 
                                num_classes=NUM_CLASSES, 
                                codes_size=CODES_SIZE,
                                shuffle=False)

    val_ds = dataset['val']
    #test_ds = dataset['test']

    model = Pix2Pix(backbone=BACKBONE, 
                    size=CROP_SIZE, 
                    hadamard=HADAMARD, 
                    num_classes=NUM_CLASSES, 
                    code_size=CODES_SIZE, 
                    mode=MODE, 
                    path=save_folder, 
                    like_paper=LIKE_PAPER)

    _ = model.evaluation(val_ds)

    '''
    Generate 10 Images
    '''
    print(f"Generating images in {save_imgs_folder}")
    for step, (img, tar) in enumerate(val_ds.take(10)):
        GenerateImages(model.generator, img, tar, save_imgs_folder, str(int(step)).zfill(2)+'.png')
    print("Image generation complete.")


if __name__ == "__main__":
    main()