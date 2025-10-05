import tensorflow as tf
from sklearn.model_selection import train_test_split
from load_data import load_data
from model import build_unet
import cv2
import numpy as np
import os
from glob import glob

dataset_path = "/home/gokul/Breast_Cancer_Detection/Data/Dataset_BUSI_with_GT"
image_size = 256

images, masks = load_data(dataset_path, image_size)
masks = masks[..., np.newaxis]

X_train, X_val, Y_train, Y_val = train_test_split(images, masks, test_size=0.2, random_state=42)

model = build_unet((image_size, image_size, 1))
model.compile(loss="binary_crossentropy", optimizer="Adam", metrics=["accuracy"])

history = model.fit(X_train, Y_train,
                    validation_data=(X_val, Y_val),
                    batch_size=8,
                    epochs=50)

val_loss, val_acc = model.evaluate(X_val, Y_val)
print(f"Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_acc:.4f}")

model.save("unet_segmentation_model.h5")
