from init import Options
import os
from monai.transforms import NormalizeIntensityd, SaveImage
from glob import glob



opt = Options().parse()

train_images = sorted(
    glob.glob(os.path.join(opt.images_folder, "*.nii.gz")))

train_transform = NormalizeIntensityd(keys=['image'])