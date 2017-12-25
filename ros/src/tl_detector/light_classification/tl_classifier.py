from styx_msgs.msg import TrafficLight
import numpy as np
import os
import tensorflow as tf
import time

# added by Nalini 12/18/2017
# added model, testing model


class TLClassifier(object):

    def __init__(self):
        """Loads the classifier model from source"""
        #TODO remove timer as son as we have solved performance issue
        now = time.time()
        self.loaded_model = False
        # code for the simulator model - Chinmaya
        print("1. - in Classifier load model 1---")
        model_path = os.path.join(os.path.dirname(__file__), 'Models/frozen_sim_mobile/frozen_inference_graph.pb')

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

        print("Done loading detection model, time needed=", time.time() - now)
        self.loaded_model = True
        
    def get_classification(self, image):
        """Determines the color of the traffic light in the image
            Args:
            image (cv::Mat): image containing the traffic light
            Returns:
        int: ID of traffic light color (specified in styx_msgs/TrafficLight)
        """
        #TODO remove timer as son as we have solved performance issue

        print("5. - in get_classification---")

        now = time.time()

        cropped_image = self.crop_region_of_interest(image)

        # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
        image_np_expanded = np.expand_dims(cropped_image, axis=0)

        # Actual detection.
        (boxes, scores, classes, num) = self.session.run(
                [self.detection_boxes, self.detection_scores, self.detection_classes, self.num_detections],
                feed_dict={self.image_tensor: image_np_expanded})

        scores = np.squeeze(scores)
        classes = np.squeeze(classes).astype(np.int32)

        min_score_thresh = .80
        above_thresh = np.argwhere(scores > min_score_thresh)
        if above_thresh is not None and above_thresh.size > 0:
            class_number = classes[above_thresh[0]]
            if class_number == 2:
                print("Done detection, time needed=", time.time() - now)
                print("in Red=", class_number)
                return TrafficLight.RED
            elif class_number == 1:
                print("Done detection, time needed=", time.time() - now)
                print("in green=", class_number)
                return TrafficLight.GREEN
            elif class_number == 3:
                print("Done detection, time needed=", time.time() - now)
                print("in yellow =", class_number)
                return TrafficLight.YELLOW

        print("Done detection, time needed=", time.time() - now)
        return TrafficLight.UNKNOWN

    def crop_region_of_interest(self, img):
        cropy = int(round(img.shape[0] * 0.7))
        image_width = img.shape[1]
        startx = int(round(img.shape[1] * 0.3))
        return img[0:cropy, startx:image_width, :]
