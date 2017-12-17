from styx_msgs.msg import TrafficLight


import numpy as np
import os
import sys
import tensorflow as tf


# added by Nalini 12/17/2017
# added model, testing model


class TLClassifier(object):

	def __init__(self):
		#TODO load classifier
		# code for the simulator model     
		model_path = '/home/student/CarND-Capstone/ros/src/tl_detector/light_classification/Models/faster_rcnn-traffic-udacity_sim/frozen_inference_graph.pb'
		
	def get_classification(self, image):
		"""Determines the color of the traffic light in the image
			Args:
			image (cv::Mat): image containing the traffic light
			Returns:
		int: ID of traffic light color (specified in styx_msgs/TrafficLight)
		"""
		#TODO implement light color prediction

		print("In classifier 4")

		model_path = '/home/student/CarND-Capstone/ros/src/tl_detector/light_classification/Models/faster_rcnn-traffic-udacity_sim/frozen_inference_graph.pb'

		# for now get color for simulator
		# if the model is able to classify any imagem then we can just use this one method for all images
		# predicted_color = predict_light(image)

		detection_graph = tf.Graph()

		# load model
		with detection_graph.as_default():

			od_graph_def = tf.GraphDef()

			with tf.gfile.GFile(model_path, 'rb') as fid:

				serialized_graph = fid.read()
				od_graph_def.ParseFromString(serialized_graph)
				tf.import_graph_def(od_graph_def, name='')

		with tf.Session(graph=detection_graph) as sess:

			# Definite input and output Tensors for detection_graph
			image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')

			# Each box represents a part of the image where a particular object was detected.
			detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')

			# Each score represent level of confidence for each of the objects.
			# Score is shown on the result image, together with the class label.
			detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
			detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')
			num_detections = detection_graph.get_tensor_by_name('num_detections:0')

			print("Image Size=", image.size)

			(im_width, im_height) = image.size

			image_np = np.array(image.getdata()).reshape((im_height, im_width, 3)).astype(np.uint8)

			# Expand dimensions since the model expects images to have shape: [1, None, None, 3]
			image_np_expanded = np.expand_dims(image_np, axis=0)


			# Actual detection.
			(boxes, scores, classes, num) = sess.run(
				[detection_boxes, detection_scores, detection_classes, num_detections],
				feed_dict={image_tensor: image_np_expanded})

			boxes = np.squeeze(boxes)
			scores = np.squeeze(scores)
			classes = np.squeeze(classes).astype(np.int32)


			min_score_thresh = .80
			for i in range(boxes.shape[0]):
				if scores is None or scores[i] > min_score_thresh:

					# use class and score to detect the red light
					class_number = classes[i]
					# if 2 is red, identify that
					# assuming 1 == green, 4 == off, 2 == red, 3 == yellow
					if class_number == 2:
						prediction = TrafficLight.RED
					elif class_number == 1:
						prediction = TrafficLight.GREEN
					elif class_number == 3:
						prediction = TrafficLight.YELLOW
					else:
						prediction = TrafficLight.UNKNOWN

			print("Prediction=", prediction)

		# return TrafficLight.UNKNOWN
		return prediction


	# creating a separate function is not working for now. 	
	def predict_light(image):

		detection_graph = tf.Graph()

		# load model
		with detection_graph.as_default():

			od_graph_def = tf.GraphDef()

			with tf.gfile.GFile(model_path, 'rb') as fid:

				serialized_graph = fid.read()
				od_graph_def.ParseFromString(serialized_graph)
				tf.import_graph_def(od_graph_def, name='')

		with tf.Session(graph=detection_graph) as sess:

			# Definite input and output Tensors for detection_graph
			image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')

			# Each box represents a part of the image where a particular object was detected.
			detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')

			# Each score represent level of confidence for each of the objects.
			# Score is shown on the result image, together with the class label.
			detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
			detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')
			num_detections = detection_graph.get_tensor_by_name('num_detections:0')

			(im_width, im_height) = image.size
			image_np = np.array(image.getdata()).reshape((im_height, im_width, 3)).astype(np.uint8)
			# Expand dimensions since the model expects images to have shape: [1, None, None, 3]
			image_np_expanded = np.expand_dims(image_np, axis=0)


			# Actual detection.
			(boxes, scores, classes, num) = sess.run(
				[detection_boxes, detection_scores, detection_classes, num_detections],
				feed_dict={image_tensor: image_np_expanded})

			boxes = np.squeeze(boxes)
			scores = np.squeeze(scores)
			classes = np.squeeze(classes).astype(np.int32)


			min_score_thresh = .80
			for i in range(boxes.shape[0]):
				if scores is None or scores[i] > min_score_thresh:

					# use class and score to detect the red light
					class_number = classes[i]
					# if 2 is red, identify that
					# assuming 1 == green, 4 == off, 2 == red, 3 == yellow
					if class_number == 2:
						prediction = TrafficLight.RED
					elif class_number == 1:
						prediction = TrafficLight.GREEN
					elif class_number == 3:
						prediction = TrafficLight.YELLOW
					else:
						prediction = TrafficLight.UNKNOWN

			print("Prediction=", prediction)

		return prediction
	


