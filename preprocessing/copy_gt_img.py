"""
As the name suggest, this copies the training images
to the desired directory. This is part of the process_stack.sh
workflow. Do not use it alone"""

"""Imports"""
import os
from concurrent.futures import ThreadPoolExecutor
import shutil

"""HYPERPARAMETERS"""
MOD = "t1c"

"""Helper Functions"""
def CreateDir(folder_name):
   if not os.path.exists(folder_name):
       os.makedirs(folder_name)   

def CopyTree(src, dst):
    # Ensure the source exists
    if not os.path.exists(src):
        raise FileNotFoundError(f"Source directory '{src}' does not exist.")
    try:
        # Recursively copy the directory tree
        shutil.copytree(src, dst, dirs_exist_ok=True)  # dirs_exist_ok=True to allow copying into an existing directory
        print(f"Successfully copied {src} to {dst}")
    except Exception as e:
        print(f"Error occurred while copying: {e}")

"""Main Runtime"""
def CopyTrainingImages(): 
    input_folder = f"{MOD}_dataset/labels"
    dest_folder = f"stack_dataset/labels"
    CreateDir(dest_folder)

    # Use ThreadPoolExecutor 
    max_workers = 10

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
            executor.submit(CopyTree, input_folder, dest_folder)

if __name__ == "__main__": 
    CopyTrainingImages()
    print("\nFinish copying, please check your directory\n")