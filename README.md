# MPFCNet
## Requirements
Follow the steps in "installation_commands.txt". Installation via Anaconda and creation of a virtual env to download the python libraries and pytorch/cuda.
*******************************************************************************
## Python scripts and their function

- organize_folder_structure.py: Organize the data in the folder structure (training,validation,testing) for the network. 
Labels are resampled and resized to the corresponding image, to avoid array size conflicts. You can set here a new image resolution for the dataset. 

- init.py: List of options used to train the network. 

- check_loader_patches: Shows example of patches fed to the network during the training.  

- networks.py: The architectures available for segmentation.

- train.py: Runs the training

- predict_single_image.py: It launches the inference on a single input image chosen by the user.
*******************************************************************************
## Usage
### Folders structure:

Use first "organize_folder_structure.py" to create organize the data.
Modify the input parameters to select the two folders: images and labels folders with the dataset. Set the resolution of the images here before training.

    .
	├── Data_folder                   
	|   ├── CT               
	|   |   ├── 1.nii 
    |   |   ├── 2.nii 	
	|   |   └── 3.nii                     
	|   ├── CT_labels                         
	|   |   ├── 1.nii 
    |   |   ├── 2.nii 	
	|   |   └── 3.nii  

Data structure after running it:

	.
	├── Data_folder  
	|   ├── CT  
	|   ├── CT_labels 
	|   ├── images              
	|   |   ├── train             
	|   |   |   ├── image1.nii              
	|   |   |   └── image2.nii                     
	|   |   └── val             
	|   |   |   ├── image3.nii             
	|   |   |   └── image4.nii
	|   |   └── test             
	|   |   |   ├── image5.nii              
	|   |   |   └── image6.nii
	|   ├── labels              
	|   |   ├── train             
	|   |   |   ├── label1.nii              
	|   |   |   └── label2.nii                     
	|   |   └── val             
	|   |   |   ├── label3.nii             
	|   |   |   └── label4.nii
	|   |   └── test             
	|   |   |   ├── label5.nii              
	|   |   |   └── label6.nii
	
*******************************************************************************
### Training:
- Modify the "init.py" to set the parameters and start the training/testing on the data. Read the descriptions for each parameter.
- Afterwards launch the "train.py" for training. Tensorboard is available to monitor the training ("runs" folder created)	
- Check and modify the train_transforms applied to the images  in "train.py" for your specific case. (e.g. In the last update there is a HU windowing for CT images)

Sample images: the following images show the segmentation of knee joint from MRI sequence

![Image](images/image.gif)![result](images/result.gif)


*******************************************************************************
### Inference:
- Launch "predict_single_image.py" to test the network. Modify the parameters in the parse section to select the path of the weights, images to infer and result. 
- You can test the model on a new image, with different size and resolution from the training. The script will resample it before the inference and give you a mask
with same size and resolution of the source image.
*******************************************************************************
### Tips:
- Use and modify "check_loader_patches.py" to check the patches fed during training.
- It is possible to add other networks, but for segmentation the U-net architecture is the state of the art.

### Sample script inference
- The label can be omitted (None) if you segment an unknown image. You have to add the --resolution if you resampled the data during training (look at the argsparse in the code).
```console
python predict_single_image.py --image './Data_folder/image.nii' --label './Data_folder/label.nii' --result './Data_folder/prova.nii' --weights './best_metric_model.pth'
```
*******************************************************************************

### Some note:
- Tensorboard can show you all segmented channels, but for now the metric is the Mean-Dice (of all channels). If you want to evaluate the Dice score for each channel you 
  have to modify a bit the plot_dice function. 
- The loss is the DiceLoss + CrossEntropy. You can modify it if you want to try others
