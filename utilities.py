from PIL import Image
from numpy import asarray
import numpy as np

def pillow_to_numpy(image):
  return asarray(image)

def numpy_to_pillow(data):
  return Image.fromarray(data)

def start_centroids(k, data):
  random_indices = np.random.choice(data.shape[0], k, replace=False)
  return data[random_indices, :]