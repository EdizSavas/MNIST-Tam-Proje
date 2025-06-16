import numpy as np
import os
import gzip
import matplotlib.pyplot as plt
from model import NeuralNetwork

base_dir = os.path.dirname(__file__)

def extract_images(filename):
    with gzip.open(filename, 'rb') as f:
        f.read(16)  # magic number and dimensions
        return np.frombuffer(f.read(), dtype=np.uint8).reshape(-1, 28*28)

def extract_labels(filename):
    with gzip.open(filename, 'rb') as f:
        f.read(8)
        return np.frombuffer(f.read(), dtype=np.uint8)

def load_custom_data(custom_dir="custom_data"):
    custom_dir = os.path.join(base_dir, custom_dir)
    images, labels = [], []
    i = 0
    while True:
        img_path = os.path.join(custom_dir, f"image_{i}.npy")
        lbl_path = os.path.join(custom_dir, f"label_{i}.npy")
        if not os.path.exists(img_path) or not os.path.exists(lbl_path):
            break
        images.append(np.load(img_path))
        labels.append(np.load(lbl_path)[0])
        i += 1
    print(f"[+] Custom veri yüklendi: {len(images)} adet")
    return np.array(images), np.array(labels)
