# --- CHECKPOINT 1 (Imports & Paths) ---

import os
import shutil
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report

import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.models import Model

BASE = "/Users/aditya/Desktop/Repositories/MinorProject-1/Dataset"
ALL_DIR = os.path.join(BASE, "ALL")

TRAIN_NEW = os.path.join(BASE, "train_new")
VAL_NEW   = os.path.join(BASE, "val_new")
TEST_NEW  = os.path.join(BASE, "test_new")

print("Paths loaded successfully.")
