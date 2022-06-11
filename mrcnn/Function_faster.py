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

def load_image_to_numpy(image):
    (im_width, im_height) = image.size
    return np.array(image.getdata()).reshape(
        (im_height, im_width, 3)).astype(np.uint8)

# Inference pipeline
def run_inference(image, graph):
    with graph.as_default():
        with tf.Session() as sess:
            # Get handles to input and output tensors
            ops = tf.get_default_graph().get_operations()
            all_tensor_names = {
                output.name for op in ops for output in op.outputs}
            tensor_dict = {}
            for key in [
                'num_detections', 'detection_boxes', 'detection_scores',
                'detection_classes', 'detection_masks'
            ]:
                tensor_name = key + ':0'
                if tensor_name in all_tensor_names:
                    tensor_dict[key] = tf.get_default_graph().get_tensor_by_name(
                        tensor_name)
            if 'detection_masks' in tensor_dict:
                # The following processing is only for single image
                detection_boxes = tf.squeeze(
                    tensor_dict['detection_boxes'], [0])
                detection_masks = tf.squeeze(
                    tensor_dict['detection_masks'], [0])
                # Reframe is required to translate mask from box coordinates to image coordinates and fit the image size.
                real_num_detection = tf.cast(
                    tensor_dict['num_detections'][0], tf.int32)
                detection_boxes = tf.slice(detection_boxes, [0, 0], [
                                           real_num_detection, -1])
                detection_masks = tf.slice(detection_masks, [0, 0, 0], [
                                           real_num_detection, -1, -1])
                detection_masks_reframed = utils_ops.reframe_box_masks_to_image_masks(
                    detection_masks, detection_boxes, image.shape[0], image.shape[1])
                detection_masks_reframed = tf.cast(
                    tf.greater(detection_masks_reframed, .5), tf.uint8)
                # Follow the convention by adding back the batch dimension
                tensor_dict['detection_masks'] = tf.expand_dims(
                    detection_masks_reframed, 0)
            image_tensor = tf.get_default_graph().get_tensor_by_name('image_tensor:0')

            # Run inference
            output_dict = sess.run(tensor_dict,
                                   feed_dict={image_tensor: np.expand_dims(image, 0)})

            # all outputs are float32 numpy arrays, so convert types as appropriate
            output_dict['num_detections'] = int(
                output_dict['num_detections'][0])
            output_dict['detection_classes'] = output_dict[
                'detection_classes'][0].astype(np.uint8)
            output_dict['detection_boxes'] = output_dict['detection_boxes'][0]
            output_dict['detection_scores'] = output_dict['detection_scores'][0]
            if 'detection_masks' in output_dict:
                output_dict['detection_masks'] = output_dict['detection_masks'][0]
    return output_dict

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
  
  return image_np
