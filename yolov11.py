from ultralytics import YOLO
import torch 
import os
import csv
import cv2 as cv

# to change hyperparameters refer to
# parameters.py 
from parameters import *

TIME = 0

"""Define Custom Callbacks"""
def PrintMemoryUsed(predictor):
    memory_used = torch.cuda.memory_allocated() / (1024 ** 2)
    print(f"\nThis is memory used: {memory_used}")

def LogMetricMemorySpeed(trainer): 
    global TIME

    # get time
    if trainer.epoch_time is not None:
        TIME += trainer.epoch_time

    # get the current GPU memory usage (in MB)
    memory_used = torch.cuda.memory_allocated() / (1024 ** 2)

    # get epoch
    epoch = trainer.epoch

    # get metric
    mAP = trainer.metrics["metrics/mAP50-95(M)"]

    # write csv_file to directory
    data = [{'epoch': epoch, 'mAP50-95': mAP, 'time': TIME, 'memory': memory_used}]
    callback_dir = f"callbacks/csv_callbacks_{MODEL}_{MODE}.csv"
    with open(callback_dir, 'a', newline='') as csvfile:
        fieldnames = ['epoch', 'mAP50-95', 'time', 'memory']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        csvfile.seek(0, 2) 
        if csvfile.tell() == 0: 
            writer.writeheader()
        writer.writerows(data)

def TrainModel(mode):
    if mode == "train":
        if not os.path.exists("callbacks"):
            os.makedirs("callbacks")  

        # load pretrained model (recommended for training)
        model = YOLO(f"{MODEL}.pt")

        # if load and train
        if LOAD_AND_TRAIN: 
            model = YOLO(BEST_MODEL_DIR_TRAIN)

        # add callback for the model
        model.add_callback("on_train_epoch_end", LogMetricMemorySpeed)

        # train the model
        results = model.train(data="config.yaml", epochs=EPOCH, imgsz=240, seed=SEED)
    elif mode == "val":
        # load pretrained model (recommended for training)
        model = YOLO(f"{MODEL}.pt")
        model = YOLO(BEST_MODEL_DIR_VAL)
        metrics = model.val()
    elif mode == "test":
        model = YOLO(f"{MODEL}.pt")
        model = YOLO(BEST_MODEL_DIR_VAL)
        model.add_callback("on_predict_end", PrintMemoryUsed)
        results = model(f"dataset/images/test/{IMAGE_TO_TEST}.png")
        
        # Save the prediction
        for result in results:
            result.save(filename="result.jpg")  # save to disk

if __name__ == "__main__":
    TrainModel(MODE)
    if MODE == "test": 
        print(f"\nFinished {MODE}, Check working directoy for 'result.jpg'\n")
    elif MODE == "train":
        print(f"\nFinished {MODE}, Check runs and callbacks folder in working directory\n")
    else: 
        print(f"\nFinished {MODE}, Check runs folder in working directory\n")
