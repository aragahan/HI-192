import os

def count_files_in_folder(folder_path):
    file_count = 0

    # Iterate through all items in the folder
    for _, _, files in os.walk(folder_path):
        file_count += len(files)

    return file_count

# Example usage
#folder_path = 'normal eye'
folder_path = 'Glaucomatous eye images'
#folder_path = 'validation/negative'
#folder_path = 'test/positive'
#folder_path = 'train'

num_files = count_files_in_folder(folder_path)
print(f"The number of files in {folder_path} is: {num_files}")