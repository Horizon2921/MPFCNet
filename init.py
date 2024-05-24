import argparse
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
class Options():

    """This class defines options used during both training and test time."""

    def __init__(self):
        """Reset the class; indicates the class hasn't been initailized"""
        self.initialized = False

    def initialize(self, parser):

        # basic parameters
        parser.add_argument('--images_folder', type=str, default='/home/dluser/dataset/nnUNetFrame/DATASET/nnUNet_raw/nnUNet_raw_data/Task55_k/imagesTr')
        parser.add_argument('--labels_folder', type=str, default='/home/dluser/dataset/nnUNetFrame/DATASET/nnUNet_raw/nnUNet_raw_data/Task55_k/labelsTr')

        # parser.add_argument('--images_folder', type=str, default='/home/dluser/dataset/nnUNetFrame/Pytorch--3D-Medical-Images-Segmentation/dataset/Task88/imagesTr')
        # parser.add_argument('--labels_folder', type=str, default='/home/dluser/dataset/nnUNetFrame/Pytorch--3D-Medical-Images-Segmentation/dataset/Task88/labelsTr')
        parser.add_argument('--output_folder', type=str, default='/home/dluser/dataset/nnUNetFrame/Pytorch--3D-Medical-Images-Segmentation/checkpoint')
        parser.add_argument('--increase_factor_data',  default=1, help='Increase data number per epoch')
        parser.add_argument('--gpu_ids', type=str, default='1', help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')
        parser.add_argument('--workers', default=4, type=int, help='number of data loading workers')
        parser.add_argument('--latestcheckpoint', type=str, default='best_metric_model_test_0821_swinunet.pth',
                            help='Store the latest checkpoint in each epoch')
        # dataset parameters
        parser.add_argument('--network', default='HRNet', help='nnunet, unetr, unet, vnet,UNet3D, UNet3Dplus, swinUNetR')
        parser.add_argument('--patch_size', default=(96,96,96), help='Size of the patches extracted from the image')
        parser.add_argument('--spacing', default=[0.7,0.365 ,0.365], help='Original Resolution')
        parser.add_argument('--resolution', default=None, help='New Resolution, if you want to resample the data in training. I suggest to resample in organize_folder_structure.py, otherwise in train resampling is slower')
        parser.add_argument('--batch_size', type=int, default=2, help='batch size, depends on GPU')
        parser.add_argument('--in_channels', default=1, type=int, help='Channels of the input')
        parser.add_argument('--out_channels', default=6, type=int, help='Channels of the output, Classes of Segmentation, Including background')

        # training parameters
        parser.add_argument('--epochs', default=300, help='Number of epochs')
        parser.add_argument('--lr', default=0.01, help='Learning rate')
        parser.add_argument('--benchmark', default=True, help='Accelerate for training on GPU')
        
        # Inference
        # This is just a trick to make the predict script working, do not touch it now for the training.
        parser.add_argument('--result', default=None, help='Keep this empty and go to predict_single_image script')
        parser.add_argument('--weights', default=None, help='Keep this empty and go to predict_single_image script')

        self.initialized = True
        return parser

    def parse(self):
        if not self.initialized:
            parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
            parser = self.initialize(parser)
        opt = parser.parse_args()
        # set gpu ids
        return opt





