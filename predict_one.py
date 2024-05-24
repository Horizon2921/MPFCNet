import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import argparse

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=str, default='/home/dluser/dataset/nnUNetFrame/DATASET/nnUNet_raw/nnUNet_raw_data/Task66_k/imagesTr/9993833.nii.gz', help='source image' )
    parser.add_argument("--label", type=str, default='/home/dluser/dataset/nnUNetFrame/DATASET/nnUNet_raw/nnUNet_raw_data/Task66_k/labelsTr/9993833.nii.gz', help='source label, if you want to compute dice. None for new case')
    parser.add_argument("--result", type=str, default='/home/dluser/dataset/nnUNetFrame/Pytorch--3D-Medical-Images-Segmentation--SALMON/dataset/test_0.nii.gz', help='path to the .nii result to save')
    parser.add_argument("--weights", type=str, default='/home/dluser/dataset/nnUNetFrame/Pytorch--3D-Medical-Images-Segmentation--SALMON/best_metric_model_test_55.pth', help='network weights to load')
    parser.add_argument("--resolution", default=None, help='Resolution used in training phase')
    parser.add_argument("--patch_size", type=int, nargs=3, default=(96, 96, 96), help="Input dimension for the generator, same of training")
    parser.add_argument('--network', default='unetr', help='nnunet, unetr')
    parser.add_argument('--gpu_ids', type=str, default='1', help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')
    args = parser.parse_args()

    segment(args.image, args.label, args.result, args.resolution, args.patch_size, args.network)