import os

def compare_folders(folder1, folder2):
    folder1_files = set(os.listdir(folder1))
    folder2_files = set(os.listdir(folder2))

    only_in_folder1 = folder1_files - folder2_files
    only_in_folder2 = folder2_files - folder1_files

    return list(only_in_folder1), list(only_in_folder2)

masks = 'train/mask300'
imgs = 'train/img300'
only_in_folder1, only_in_folder2 = compare_folders(masks, imgs)
print(f"Files only in {masks}: {only_in_folder1}")
print(f"Files only in {imgs}: {only_in_folder2}")