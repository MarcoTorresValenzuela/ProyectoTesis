import numpy as np
import os
import cv2
import six.moves.urllib as urllib
import sys
import tarfile
import tensorflow as tf
import zipfile
from google.colab.patches import cv2_imshow
from collections import defaultdict
from io import StringIO
import matplotlib
import pandas as pd
import matplotlib.pyplot as plt
import time
import tensorflow as tf
from PIL import Image
from object_detection.utils import ops as utils_ops
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util
def deteccion_imagen(image_path,PATH_TO_CKPT):
  # Valores de la red neuronal propia
  num_classes = 1 # numero de clases de la red
  IMAGE_SIZE = (12, 8) # dimension de la imagen de salida

  # Para lectura de todas las imagenes dentro de una carpeta
  #IMAGE_DIR = "/content/gdrive/MyDrive/customTF1/img"
  #IMAGE_PATHS = []
  #for file in os.listdir(IMAGE_DIR):
  #   if file.endswith(".jpg") or file.endswith(".png"):
  #       IMAGE_PATHS.append(os.path.join(IMAGE_DIR, file))

  # se configuracion los direcctorios de label map y inference.pb
  PATH_TO_LABELS = '/content/gdrive/MyDrive/customTF1/data/label_map.pbtxt'

  # seteo de tensorflow graph
  detection_graph = tf.Graph()
  with detection_graph.as_default():
      od_graph_def = tf.GraphDef()
      with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
          serialized_graph = fid.read()
          od_graph_def.ParseFromString(serialized_graph)
          tf.import_graph_def(od_graph_def, name='')

  # Seteo de las  categorrias
  label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
  categories = label_map_util.convert_label_map_to_categories(
      label_map, max_num_classes=num_classes, use_display_name=True)
  category_index = label_map_util.create_category_index(categories)

  #se utiliza para la posterior lectura de imagenes dentro de una carpeta(todas la imagenes)
  #for image_path in IMAGE_PATHS:
    #  image = Image.open(image_path)

  #solo para una imagen
  # correr inferencia en la imagen seteada
  image = Image.open(image_path)
  # conversion en la imagen en  numpy array (pasar la imagen a matriz de numeros)
  image_np = load_image_to_numpy(image)
  # Expancion de las dimensiones ya que el modelo espera que las imágenes tengan forma: [1, None, None, 3]
  image_np_expanded = np.expand_dims(image_np, axis=0)
  # Realiza la inferencia
  inicio = time.perf_counter()
  output_dict = run_inference(image_np, detection_graph)
  final = time.perf_counter()
  # Visualizacion
  inicio1 = time.perf_counter()
  vis_util.visualize_boxes_and_labels_on_image_array(
      image_np,
      output_dict['detection_boxes'],
      output_dict['detection_classes'],
      output_dict['detection_scores'],
      category_index,
      instance_masks=output_dict.get('detection_masks'),
      use_normalized_coordinates=True,
      line_thickness=4,
      min_score_thresh=0.7)
  plt.figure(figsize=IMAGE_SIZE, dpi=200)
  plt.axis("off")
  plt.imshow(image_np)
  final1 = time.perf_counter()
  print("Tiempo de deteccion",(final-inicio),"segundos")
  print("Tiempo de visualizacion",(final1-inicio1)*1000,"milisegundos")
  
  return
