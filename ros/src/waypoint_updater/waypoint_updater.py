#!/usr/bin/env python

import rospy
from geometry_msgs.msg import PoseStamped, TwistStamped
from styx_msgs.msg import Lane, Waypoint
from std_msgs.msg import Int32
import tf
import math

'''
This node will publish waypoints from the car's current position to some `x` distance ahead.

As mentioned in the doc, you should ideally first implement a version which does not care
about traffic lights or obstacles.

Once you have created dbw_node, you will update this node to use the status of traffic lights too.

Please note that our simulator also provides the exact location of traffic lights and their
current status in `/vehicle/traffic_lights` message. You can use this message to build this node
as well as to verify your TL classifier.

TODO (for Yousuf and Aaron): Stopline location for each traffic light.
'''

LOOKAHEAD_WPS = 200 # Number of waypoints we will publish. You can change this number
MAX_DECEL = 0.5
STOP_DIST = 5.0
TARGET_SPEED_METER_PER_SECOND = 10 * 1609.34/3600

class WaypointUpdater(object):
    def __init__(self):
        rospy.init_node('waypoint_updater')

        rospy.Subscriber('/current_pose', PoseStamped, self.pose_cb)
        rospy.Subscriber('/base_waypoints', Lane, self.waypoints_cb)

        # TODO: Add a subscriber for /traffic_waypoint and /obstacle_waypoint below
        rospy.Subscriber('/traffic_waypoint', Int32, self.traffic_cb)
        rospy.Subscriber('/current_velocity', TwistStamped, self.current_velocity_cb)

        self.final_waypoints_pub = rospy.Publisher('final_waypoints', Lane, queue_size=1)

        # TODO: Add other member variables you need below
        self.waypoints = None
        self.waypoints_length = None
        self.current_pose = None
        self.current_velocity = None
        self.idx_of_nearest = None
        self.red_light_wp = -1
        self.traffic_time_received = rospy.get_time()

        rospy.loginfo("Init")
        rospy.spin()

    def pose_cb(self, msg):
        # TODO: Implement
        rospy.loginfo("Pose cb")
        self.current_pose = msg.pose
        if self.waypoints is not None:
            self.create_waypoints()
        pass

    def waypoints_cb(self, msg):
        # TODO: Implement
        rospy.loginfo("Waypoints cb")
        if self.waypoints is None:
            self.waypoints = msg.waypoints
            self.waypoints_length = len(msg.waypoints)
            self.create_waypoints()
        pass

    def traffic_cb(self, msg):
        # TODO: Callback for /traffic_waypoint message. Implement
        self.red_light_wp = msg.data
        self.traffic_time_received = rospy.get_time()
        #self.create_waypoints()

    def current_velocity_cb(self, msg):
        # TODO: Callback for /current_velocity message. Implement
        self.current_velocity = msg.twist.linear.x
        rospy.loginfo("Current velocity %i", self.current_velocity)


    def obstacle_cb(self, msg):
        # TODO: Callback for /obstacle_waypoint message. We will implement it later
        pass

    def get_waypoint_velocity(self, waypoint):
        return waypoint.twist.twist.linear.x

    def set_waypoint_velocity(self, waypoints, waypoint, velocity):
        waypoints[waypoint].twist.twist.linear.x = velocity

    def euclidean_distance(self, a, b):
        return math.sqrt((a.x-b.x)**2 + (a.y-b.y)**2 + (a.z-b.z)**2)

    def distance(self, waypoints, wp1, wp2):
        dist = 0
        for i in range(wp1, wp2+1):
            dist += self.euclidean_distance(waypoints[wp1].pose.pose.position, waypoints[i].pose.pose.position)
            wp1 = i
        return dist

    def get_nearest_waypoint(self):
        closest_len = 100000
        closest_waypoint = 0
        for index, waypoint in enumerate(self.waypoints):
            dist = self.euclidean_distance(self.current_pose.position, waypoint.pose.pose.position)
            if dist < closest_len:
                closest_len = dist
                closest_waypoint = index

        self.idx_of_nearest = closest_waypoint

    def check_if_car_already_passed_nearest_waypoint(self):
        map_x = self.waypoints[self.idx_of_nearest].pose.pose.position.x
        map_y = self.waypoints[self.idx_of_nearest].pose.pose.position.y

        heading = math.atan2((map_y - self.current_pose.position.y), (map_x - self.current_pose.position.x))
        quaternion = (self.current_pose.orientation.x, self.current_pose.orientation.y, self.current_pose.orientation.z, self.current_pose.orientation.w)
        _, _, yaw = tf.transformations.euler_from_quaternion(quaternion)
        angle = abs(yaw - heading)

        if angle > (math.pi / 4):
            rospy.loginfo("add 1 to index")
            self.idx_of_nearest += 1


    def create_waypoints(self):
        if self.current_pose is not None:
            if self.idx_of_nearest is None:
                #initial waypoint search
                self.get_nearest_waypoint()

            if self.idx_of_nearest is not None:
                self.check_if_car_already_passed_nearest_waypoint()

                stop = self.idx_of_nearest + LOOKAHEAD_WPS
                next_waypoints = self.waypoints[self.idx_of_nearest:stop]

                # Use this code only as long as our red light detection does not work
                for i in range(len(next_waypoints)-1):
                    self.set_waypoint_velocity(next_waypoints, i, TARGET_SPEED_METER_PER_SECOND)

                """ Use this code as soon as we detect red lights!
                if self.red_light_wp < 0:
                    for i in range(len(next_waypoints)-1):
                        self.set_waypoint_velocity(next_waypoints, i, TARGET_SPEED_METER_PER_SECOND)
                else:
                    redlight_lookahead_index = max(0, self.red_light_wp - self.idx_of_nearest)
                    next_waypoints = self.decelerate(next_waypoints, redlight_lookahead_index)
                """
                lane = Lane()
                lane.waypoints = next_waypoints
                lane.header.frame_id = '/world'

                self.final_waypoints_pub.publish(lane)

    def decelerate(self, waypoints, redlight_lookahead_index):

        if len(waypoints) < 1:
            return []

        last_wp = waypoints[-1]
        last_wp.twist.twist.linear.x = 0

        first_wp = waypoints[0]

        total_dist = self.euclidean_distance(first_wp.pose.pose.position, last_wp.pose.pose.position)

        for index, wp in enumerate(waypoints):
            if index > redlight_lookahead_index:
                vel = 0
            else:
                dist = self.euclidean_distance(wp.pose.pose.position, last_wp.pose.pose.position)
                dist = max(0, dist - STOP_DIST)
                vel = math.sqrt(2*MAX_DECEL*dist)
                if vel < 1.0:
                    vel = 0.0

            wp.twist.twist.linear.x = min(vel, wp.twist.twist.linear.x)

        return waypoints


if __name__ == '__main__':
    try:
        WaypointUpdater()
    except rospy.ROSInterruptException:
        rospy.logerr('Could not start waypoint updater node.')
