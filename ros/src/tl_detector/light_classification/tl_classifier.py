from styx_msgs.msg import TrafficLight


import numpy as np
import os
import sys
import tensorflow as tf



# added by Nalini 12/18/2017
# added model, testing model


class TLClassifier(object):

    def __init__(self):
        #TODO load classifier

        # code for the simulator model - Michael    
        # model_path = os.path.join(os.path.dirname(__file__), 'Models/faster_rcnn-traffic-udacity_sim/frozen_inference_graph.pb')

	# code for the simulator model - Chinmaya  

	print("ChinmayaModel 3")  
        model_path = os.path.join(os.path.dirname(__file__), 'Models/frozen_sim_inception/frozen_inference_graph.pb')

        self.detection_graph = tf.Graph()

        # load model
        with self.detection_graph.as_default():
            od_graph_def = tf.GraphDef()
            with tf.gfile.GFile(model_path, 'rb') as fid:
                serialized_graph = fid.read()
                od_graph_def.ParseFromString(serialized_graph)
                tf.import_graph_def(od_graph_def, name='')

            self.session = tf.Session(graph=self.detection_graph)

            # Definite input and output Tensors for detection_graph
            self.image_tensor = self.detection_graph.get_tensor_by_name('image_tensor:0')

            # Each box represents a part of the image where a particular object was detected.
            self.detection_boxes = self.detection_graph.get_tensor_by_name('detection_boxes:0')

            # Each score represent level of confidence for each of the objects.
            # Score is shown on the result image, together with the class label.
            self.detection_scores = self.detection_graph.get_tensor_by_name('detection_scores:0')
            self.detection_classes = self.detection_graph.get_tensor_by_name('detection_classes:0')
            self.num_detections = self.detection_graph.get_tensor_by_name('num_detections:0')
        
    def get_classification(self, image):
        """Determines the color of the traffic light in the image
            Args:
            image (cv::Mat): image containing the traffic light
            Returns:
        int: ID of traffic light color (specified in styx_msgs/TrafficLight)
        """
        #TODO implement light color prediction
        print("Image Size=", image.shape)

        im_width, im_height, channels = image.shape

        # image_np = np.array(image.getdata()).reshape((im_height, im_width, 3)).astype(np.uint8)

        # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
        image_np_expanded = np.expand_dims(image, axis=0)


        # Actual detection.
        (boxes, scores, classes, num) = self.session.run(
                [self.detection_boxes, self.detection_scores, self.detection_classes, self.num_detections],
                feed_dict={self.image_tensor: image_np_expanded})

        boxes = np.squeeze(boxes)
        scores = np.squeeze(scores)
        classes = np.squeeze(classes).astype(np.int32)

        # prediction = TrafficLight.UNKNOWN
        min_score_thresh = .80
        for i in range(boxes.shape[0]):
            if scores is None or scores[i] > min_score_thresh:
                # use class and score to detect the red light
                class_number = classes[i]

                print("Class Number=", class_number)

                # if 2 is red, identify that
                # assuming 1 == green, 4 == off, 2 == red, 3 == yellow
                if class_number == 2:
                    print("in Red=", class_number)
                    return TrafficLight.RED
                elif class_number == 1:
                    print("in green=", class_number)
                    return TrafficLight.GREEN
                elif class_number == 3:
                    return TrafficLight.YELLOW

        return TrafficLight.UNKNOWN
