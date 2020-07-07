from utils import preprocess_image, postprocess_boxes
from utils.draw_boxes import draw_boxes
from tensorflow.keras.models import load_model
import tensorflow as tf

import os
import cv2
import time
import json
import numpy as np

def infer(image:str,phi:int=0,saved_model:str='./savedmodel',classes:dict=None,score_threshold:float=0.3,nms_threshold:float=0.5,device:str='gpu'):
	if device!='gpu':
		os.environ['CUDA_VISIBLE_DEVICES'] = '-1' #Using CPU
	else:
		os.environ['CUDA_VISIBLE_DEVICES'] = '0' #Using GPU

	#For COCO dataset
	if classes==None:
		classes = {value['id'] - 1: value['name'] for value in json.load(open('coco_90.json', 'r')).values()}


	#select resolution according to architecture
	image_sizes = (512, 640, 768, 896, 1024, 1280, 1408)
	image_size = image_sizes[phi]

	#To get different color for each class
	num_classes = len(classes.values())
	colors = [np.random.randint(0, 256, 3).tolist() for _ in range(num_classes)]

	#load the model
	model = load_model(saved_model)

	#load and preprocess image
	img = cv2.imread(image)
	src_image = img.copy()
	img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
	h, w = img.shape[:2]
	img, scale = preprocess_image(img, image_size=image_size)

	#detect and post process
	start = time.time()
	boxes, scores, labels = model.predict_on_batch([np.expand_dims(img, axis=0)])
	boxes, scores, labels = np.squeeze(boxes), np.squeeze(scores), np.squeeze(labels)
	end = time.time()
	boxes = postprocess_boxes(boxes=boxes, scale=scale, height=h, width=w)

	# print(f'infer time: {end-start}, fps: {1/(end-start)}')

	# select indices which have a score above the threshold
	indices = np.where(scores[:] > score_threshold)[0]
	# indices = tf.image.non_max_suppression(boxes,scores,max_output_size=[100],iou_threshold = nms_threshold,score_threshold = score_threshold)
	boxes = boxes[indices]
	labels = labels[indices]

	#draw boxes on the original image
	draw_boxes(src_image, boxes, scores, labels, colors, classes)

	return src_image