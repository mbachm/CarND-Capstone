from styx_msgs.msg import TrafficLight

# adding code assuming that we have a model that classifies the lights


import cv2
import numpy      as np
import tensorflow as tf
from   keras                    import backend as K
from   keras                    import layers
from   keras.models             import load_model

class TLClassifier(object):
    def __init__(self):

        #TODO load classifier

        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        sess = tf.Session(config=config)
        K.set_session(sess)

        # load model here 
        # self.model = 

        self.get_output = K.function([self.model.layers[0].input, K.learning_phase()],
                                     [self.model.layers[-1].output])

       
       # pass

    def get_classification(self, image):
        """Determines the color of the traffic light in the image

        Args:
            image (cv::Mat): image containing the traffic light

        Returns:
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)

        """
        #TODO implement light color prediction


        image = cv2.resize(image, (224, 224), interpolation=cv2.INTER_NEAREST)
        image = np.expand_dims(image, axis=0)
        image = image.astype(dtype=np.float64, copy=False)
        image = image / 255.0
        
        pred = self.get_output([image, 0])[0]
        pred = np.argmax(pred)

        # determine light color based on what is returned by model

        return TrafficLight.UNKNOWN
