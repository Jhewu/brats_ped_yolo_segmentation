"""
All hyperparameters configured within
this file. The other hyperparameters
are in "default" mode, check 
https://docs.ultralytics.com/usage/cfg/#tasks
for more information
"""

"""General"""
MODE = "training"
SEED = 42
MODEL = "yolo11x-seg"   # Choose between yolo11n-seg or yolo11x-seg
                        # 11n is smallest, while 11x is biggest

"""Training"""
LOAD_AND_TRAIN = False
EPOCH = 1
BEST_MODEL_DIR_TRAIN = "runs/segment/yolo11x_single_mod/weights/best.pt"

"""Validation"""
BEST_MODEL_DIR_VAL = "runs/segment/yolo11x_single_mod/weights/best.pt"

"""Testing"""
IMAGE_TO_TEST = "BraTS-PED-00004-00047"
BEST_MODEL_DIR_TEST = "runs/segment/yolo11x_single_mod/weights/best.pt"
