import tkinter as tk
from tkinter import filedialog
import cv2
import numpy as np
import tensorflow as tf
from PIL import Image
import matplotlib.pyplot as plt
import cv2
import numpy as np
import os
from glob import glob

model_path = "unet_segmentation_model.h5"
model = tf.keras.models.load_model(model_path)

def load_image():
    file_path = filedialog.askopenfilename(title="Select an Image", filetypes=[("PNG Images", "*.png")])
    if not file_path:
        return
    img = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (256, 256)) / 255.0
    img = np.expand_dims(img, axis=(0, -1))
    pred_mask = model.predict(img)[0, :, :, 0]
    display_images(file_path, pred_mask)

def display_images(image_path, pred_mask):
    fig, ax = plt.subplots(1, 2, figsize=(6, 3))
    ax[0].imshow(Image.open(image_path).convert("L"), cmap="gray")
    ax[0].axis("off")
    ax[0].set_title("Input Image")
    ax[1].imshow(pred_mask, cmap="gray")
    ax[1].axis("off")
    ax[1].set_title("Predicted Mask")
    plt.show(block=False)

root = tk.Tk()
root.title("Breast Cancer Detection")
root.geometry("400x200")

tk.Label(root, text="Select an Image for Mask Prediction", font=("Arial", 12)).pack(pady=10)
tk.Button(root, text="Choose Image", command=load_image, font=("Arial", 12)).pack(pady=10)

root.mainloop()
