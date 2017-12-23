#!/usr/bin/env python
import rospy
from std_msgs.msg import Int32
from geometry_msgs.msg import PoseStamped, Pose
from styx_msgs.msg import TrafficLightArray, TrafficLight
from styx_msgs.msg import Lane
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from light_classification.tl_classifier import TLClassifier
import tf
import yaml
import time

# latest updates by Nalini 12/11/2017
# not completed 

import numpy as np
import math


STATE_COUNT_THRESHOLD = 3


class TLDetector(object):
    def __init__(self):
        rospy.init_node('tl_detector')

        self.car_pose = None
        self.waypoints = None
        self.camera_image = None
        # Extrem slow startup of TLClassifier(). Perhaps there is a more elegant solution
        self.light_classifier = None
        self.has_image = False
        self.lights = []
        self.state = TrafficLight.UNKNOWN
        self.last_state = TrafficLight.UNKNOWN
        self.last_wp = -2
        self.state_count = 0

        sub1 = rospy.Subscriber('/current_pose', PoseStamped, self.pose_cb)
        sub2 = rospy.Subscriber('/base_waypoints', Lane, self.waypoints_cb)

        '''
        /vehicle/traffic_lights provides you with the location of the traffic light in 3D map space and
        helps you acquire an accurate ground truth data source for the traffic light
        classifier by sending the current color state of all traffic lights in the
        simulator. When testing on the vehicle, the color state will not be available. You'll need to
        rely on the position of the light and the camera image to predict it.
        '''
        sub3 = rospy.Subscriber('/vehicle/traffic_lights', TrafficLightArray, self.traffic_cb)
        sub6 = rospy.Subscriber('/image_color', Image, self.image_cb)

        config_string = rospy.get_param("/traffic_light_config")
        self.config = yaml.load(config_string)

        self.upcoming_red_light_pub = rospy.Publisher('/traffic_waypoint', Int32, queue_size=1)

        self.bridge = CvBridge()
        self.light_classifier = TLClassifier()
        self.listener = tf.TransformListener()

        rospy.spin()

    def pose_cb(self, msg):
        self.car_pose = msg

    def waypoints_cb(self, waypoints):
        self.waypoints = waypoints.waypoints

    def traffic_cb(self, msg):
        self.lights = msg.lights

    # Some utility functions:

    def get_car_coordinates (self, car_pose):
        car_x = car_pose.pose.position.x
        car_y = car_pose.pose.position.y
        car_z = car_pose.pose.position.z
        return car_x, car_y, car_z

    def get_waypoint_coordinates(self, waypoint):
        w_x = waypoint.pose.pose.position.x
        w_y = waypoint.pose.pose.position.y
        w_z = waypoint.pose.pose.position.z
        return w_x, w_y, w_z

    def get_light_coordinates(self, light):
        l_x = light.pose.pose.position.x
        l_y = light.pose.pose.position.y
        l_z = light.pose.pose.position.z
        return l_x, l_y, l_z

    def distance(self, x1, y1, z1, x2, y2, z2):
        dx, dy, dz = x1-x2, y1-y2, z1-z2
        dist = math.sqrt(dx*dx + dy*dy + dz*dz)
        return dist

    def waypoint_ahead(self, waypoint, pose):
        # check if waypoint is ahead of car
        xw, yw, zw = self.get_waypoint_coordinates(waypoint)
        xc, yc, zc = self.get_car_coordinates(self.car_pose)
        direction = atan2((yc -yw), (xc - xw))
        return abs(direction) < math.pi*0.5


    def image_cb(self, msg):
        """Identifies red lights in the incoming camera image and publishes the index
            of the waypoint closest to the red light's stop line to /traffic_waypoint
            Values: -2 = uninitialized, -1 = no red light, > 0 = next red light

        Args:
            msg (Image): image from car-mounted camera

        """
        #TODO remove timer as son as we have solved performance issue
        now = time.time()
        self.has_image = True
        self.camera_image = msg
        light_wp, state = self.process_traffic_lights()

        '''
        Publish upcoming red lights at camera frequency.
        Each predicted state has to occur `STATE_COUNT_THRESHOLD` number
        of times till we start using it. Otherwise the previous stable state is
        used.
        '''
        if self.state != state:
            self.state_count = 0
            self.state = state
        elif self.state_count >= STATE_COUNT_THRESHOLD:
            self.last_state = self.state
            light_wp = light_wp if state == TrafficLight.RED else -1
            self.last_wp = light_wp
            self.upcoming_red_light_pub.publish(Int32(light_wp))
        else:
            self.upcoming_red_light_pub.publish(Int32(self.last_wp))
            self.state_count += 1

        print("image_cb, time needed=", time.time() - now)

    def get_closest_waypoint(self, pose):
        """Identifies the closest path waypoint to the given position
            https://en.wikipedia.org/wiki/Closest_pair_of_points_problem
        Args:
            pose (Pose): position to match a waypoint to 

        Returns:
            int: index of the closest waypoint in self.waypoints

        Modified by Nalini 11/23/2017

        """
        distances = []

        for wp in self.waypoints:
            x1, y1, z1 = self.get_waypoint_coordinates(wp)
            dist = self.distance(x1, y1, z1, pose.position.x, pose.position.y, pose.position.z)
            distances.append(dist)

        closest_wp = np.argmin(distances)

        return closest_wp

    def get_light_state(self, light):
        """Determines the current color of the traffic light

        Args:
            light (TrafficLight): light to classify

        Returns:
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)

        """
        if not self.has_image:
            return TrafficLight.UNKNOWN

        cv_image = self.bridge.imgmsg_to_cv2(self.camera_image, "bgr8")

        # Get classification
        # Extrem slow startup of TLClassifier(). Perhaps there is a more elegant solution
        if self.light_classifier is not None:
            return self.light_classifier.get_classification(cv_image)
        else:
            # Prevent the car to move before our traffic light classifier is ready
            return TrafficLight.UNKNOWN

    def process_traffic_lights(self):
        """Finds closest visible traffic light, if one exists, and determines its
            location and color

        Returns:
            int: index of waypoint closest to the upcoming stop line for a traffic light (-1 if none exists)
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)

            
        """
        closest_tl_wpt = -1
        closest_light_number = -1

        if self.car_pose and self.waypoints and self.has_image:

            start_time = time.time()

            # get closest waypoint to the car
            waypoint_car_idx = self.get_closest_waypoint(self.car_pose.pose)

            print("ClosestWaypointCar=", waypoint_car_idx)

            # check all the lights
            for i, light in enumerate(self.lights):
                # get the closest waypoint to the light
                closest_light_wp_index = self.get_closest_waypoint(light.pose.pose)

                # if the light waypooint is behind the car waypoint, move on to the next one
                if closest_light_wp_index < waypoint_car_idx:
                    continue

                # find the light that is closest to the car and return the color of that light
                if closest_tl_wpt < 0 or closest_light_wp_index < closest_tl_wpt:
                    closest_tl_wpt = closest_light_wp_index
                    closest_light_number = i
  
            print("Closest light waypoint =", closest_tl_wpt)
            print("Closest light number =", closest_light_number)

            # get the closest light and the state of that light only
            closest_light = self.lights[closest_light_number]
            light_state = self.get_light_state(closest_light)

            print("\nTotal Detection time =", time.time() - start_time)

            #send back the index of the waypoint closest to the car and light
            # changing as suggested by Jingjing - nalini 12/9/2017
            if closest_tl_wpt != -1:
                print("Light State In Detector=", light_state)
                return closest_tl_wpt, light_state

            else:
                print("Returning unknown")
                return -1, TrafficLight.UNKNOWN

        else:
            print("Not ready")
            return -2, TrafficLight.UNKNOWN


if __name__ == '__main__':
    try:
        TLDetector()
    except rospy.ROSInterruptException:
        rospy.logerr('Could not start traffic node.')
