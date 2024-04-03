import os
import tarfile
import h5py
import numpy as np
from PIL import Image
import os
import glob
tar_path = "./laion2b-ar_tar"
output_path = "./laion4"
splits = 498
# Get a list of all files in the folder
file_list = os.listdir(tar_path)


# Paths to the train and validation datasets
train_path = "./laion4"

# Path to save the HDF5 file
hdf5_file_path = './laion4.h5'

def untar(chunck, folder_path, output_path):
    for file_name in chunck:
        file_path = os.path.join(folder_path, file_name)
        # Check if the file is a tar file
        if file_name.endswith(".tar"):
            # Open the tar file
            with tarfile.open(file_path, "r") as tar:
                # Extract all files from the tar archive
                tar.extractall(path=output_path)
                print(f"Extracted files from {file_name}")

def store_images_in_hdf5(dataset_path, hdf5_file):
    # Get the list of image files in the dataset
    imgs_path = os.path.join(dataset_path, "*.jpg")
    image_files = glob.glob(imgs_path)

    for count,image_file in enumerate(image_files):
        try:
            # image_path = os.path.join(dataset_path, image_file)
            image_name = os.path.splitext(image_file)[0]
           
            
           
            image_name = str(int(image_name.split('/')[2].split('.')[0]))
            # Open the image using PIL
    #         image = Image.open(image_path)
            breakpoint()
            with open(image_file, 'rb') as img_f:
                 binary_data = img_f.read()      
            # Convert the image to a numpy array
            image_array = np.asarray(binary_data)
            # Store the image array in the HDF5 file with the image name as the key
            hdf5_file[image_name] = image_array
            if (count%10000 == 0):
                print(count)
        except:
            print("error reading image")

    print(f"Image storage completed for Chunck!")

def store_dataset_in_hdf5(splits, tar_path, output_path, file_list, hdf5_file_path):
    # Create an HDF5 file
    chunck_size = len(file_list)//splits
    with h5py.File(hdf5_file_path, 'w') as hdf5_file:
        # Store the train images
        for i in range(1,splits):
            try:
             untar(file_list[(i-1)*chunck_size:i*chunck_size],tar_path,output_path)
             print("chunch untared")
            except:
                print("if it is called twice, it is not nice")
                untar(file_list[(i-1)*chunck_size::],tar_path,output_path)
            store_images_in_hdf5(output_path, hdf5_file)
            files = glob.glob('/laion4/*')
            for f in files:
                os.remove(f)
            breakpoint()
            print("#################chunck completed!################")
    print("HDF5 file creation completed!")

# Call the function to store the train and validation images in the same HDF5 file
store_dataset_in_hdf5(splits,tar_path, output_path, file_list, hdf5_file_path)