import os
import random
import shutil

def transfer_files(source_folder, destination_folder, num_files):
    file_list = os.listdir(source_folder)
    random.shuffle(file_list)
    
    files_to_transfer = file_list[:num_files]
    
    for file_name in files_to_transfer:
        source_file = os.path.join(source_folder, file_name)
        destination_file = os.path.join(destination_folder, file_name)
        shutil.move(source_file, destination_file)
        print(f"Transferred {file_name} to {destination_folder}")

# Example usage
source_folder = 'normal eye/'
destination_folder = 'train/negative/'
num_files_to_transfer = 37

transfer_files(source_folder, destination_folder, num_files_to_transfer)