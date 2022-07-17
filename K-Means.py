'''
Machine Learning
Lab 09: K-means
Daniel Ricardo Ramírez Umaña, B45675
'''

'''
Los imports
'''
from ctypes import resize
import PIL
print('Pillow Version:', PIL.__version__)
from PIL import Image
from numpy import asarray
import utilities as uti
import numpy as np
from matplotlib import pyplot as plt



'''
1) Carga una imagen, la convierte de pillow a numpy y edita su tamaño
'''
def load_image(filename, resize):
  image = Image.open(filename)
  image.show()
  print(image.size)
  data = uti.pillow_to_numpy(image)
  data.resize(resize)
  image1 = uti.numpy_to_pillow(data)
  image1.show()
  print(image1.size)
  data = data.reshape((data.shape[0] * data.shape[1], data.shape[2]))
  return data

'''
2) Descrpción
'''
def euclidean_distance(p1, p2):
  return np.sqrt(np.sum(np.square(p1-p2)))
  

'''
3) Descrpción
'''
def manhattan_distance(p1, p2):
  return sum(abs(p1_val-p2_val) for p1_val, p2_val in zip(p1,p2))



'''
4) Recibo un punto y devuelvo el centroide al
   que está más cerca y su distancia a él
'''
def nearest_centroid(point, centroids, distance):
  nearest_dist = None
  idx_centroid = None

  if distance == 'euclidean':
    for index, cent in enumerate(centroids):
      dist = euclidean_distance(point, cent)
      if nearest_dist == None or dist < nearest_dist:
        nearest_dist = dist
        idx_centroid = index
  elif distance == 'manhattan':
    for index, cent in enumerate(centroids):
      dist = manhattan_distance(point, cent)
      if nearest_dist == None or dist < nearest_dist:
        nearest_dist = dist
        idx_centroid = index
  else:
    print('ERROR: función de distancia no válida')
  
  return idx_centroid, nearest_dist


def choose_better_known(data, k, iters, distance):
  better_error_known = None
  better_centroids_known = None
  better_clusters_known = None
  ## inicializamos los clusters ##
  clusters = [[] for i in range (k)]
  for epoch in range(iters):
    print(f'epoch: {epoch+1}')
    error = 0
    centroids = uti.start_centroids(k, data)
    for point in data:
      nearest_index, dist = nearest_centroid(point, centroids, distance)
      clusters[nearest_index].append(point)
      error += dist
    print(error)
    print(better_error_known)
    if better_error_known == None or error < better_error_known:
      better_error_known = error
      better_centroids_known = centroids
      better_clusters_known = clusters
      print(f'Better error known: {better_error_known}\n')
      print(f'Better centroids known:\n{better_centroids_known}\n')
  return better_clusters_known, better_centroids_known, better_error_known 

'''
5) Descrpción
'''
def lloyd(data, k, iters, type, distance):

  
  if type == 'means':
    better_clusters_known, better_centroids_known, better_error_known  = choose_better_known(data, k, iters, distance)
    for i, cluster in enumerate(better_clusters_known):
      better_centroids_known[i] = np.array(cluster).mean(axis=0)
    return better_centroids_known, better_error_known 
  elif type == 'mediods':
    better_clusters_known, better_mediods_known, better_error_known  = choose_better_known(data, k, iters, distance)
    for i, cluster in enumerate(better_clusters_known):
      random_mediod = np.random.choice(cluster[0], replace=False)
      better_mediods_known[i] = cluster[random_mediod]
    return better_mediods_known, better_error_known 
  else:
    print('tipo de clustering no válido')

  


def main():
  filename = '.\images\X-Men-04.jpg'
  resize = (600,400,3)
  data = load_image(filename, resize)
  print(data.shape)
  'Cantidad de clusters:'
  k = 5
  'Cantidad de épocas:'
  iters = 30
  centroids_euc, error_centroids_euc = lloyd(data, k, iters, 'means', 'euclidean')
  medioids_euc, error_medioids_euc = lloyd(data, k, iters, 'mediods', 'euclidean')
  centroids_manh, error_centroids_manh = lloyd(data, k, iters, 'means', 'manhattan')
  medioids_manh, error_medioids_manh = lloyd(data, k, iters, 'mediods', 'manhattan')


  print(f'\n\n-----------------------------------')
  print(f'K-Means       Euclidean')
  print(f'Better error known: {error_centroids_euc}')
  print(f'Better centroids known:\n{centroids_euc}')
  print(f'-----------------------------------')

  print(f'\n\n-----------------------------------')
  print(f'K-Mediods     Euclidean')
  print(f'Better error known: {error_medioids_euc}')
  print(f'Better centroids known:\n{medioids_euc}')
  print(f'-----------------------------------')

  print(f'\n\n-----------------------------------')
  print(f'K-Means       Manhattam')
  print(f'Better error known: {error_centroids_manh}')
  print(f'Better centroids known:\n{centroids_manh}')
  print(f'-----------------------------------')

  print(f'\n\n-----------------------------------')
  print(f'K-Mediods     Manhattam')
  print(f'Better error known: {error_medioids_manh}')
  print(f'Better centroids known:\n{medioids_manh}')
  print(f'-----------------------------------')

  results = [centroids_euc, medioids_euc, centroids_manh, medioids_manh]
  size = k*4
  gama = [None] * size
  for i in range(4):
    for c in range(k):
      pos = k*i+c
      gama[pos] = plt.subplot(4, 5, pos+1)
  for i in range(4):
    for c in range(k):
      pos = k*i+c
      gama[pos].imshow([[results[i][c].astype(int)]])
  plt.show()
  plt.savefig('color_palette%s.png' % 0)





if __name__ == "__main__":
    main()