#!/usr/bin/env python

import rospy
import math
from geometry_msgs.msg import PoseStamped
from styx_msgs.msg import Lane, Waypoint
from std_msgs.msg import Int32

'''
This node will publish waypoints from the car's current position to some `x` distance ahead.
As mentioned in the doc, you should ideally first implement a version which does not care
about traffic lights or obstacles.
Once you have created dbw_node, you will update this node to use the status of traffic lights too.
Please note that our simulator also provides the exact location of traffic lights and their
current status in `/vehicle/traffic_lights` message. You can use this message to build this node
as well as to verify your TL classifier.

'''

LOOKAHEAD_WPS = 200 # Number of waypoints we will publish. You can change this number
MAX_DECEL = 0.5
STOP_DIST = 30.0
TARGET_SPEED_METER_PER_SECOND = 1609.34/3600


class WaypointUpdater(object):
    def __init__(self):
        rospy.init_node('waypoint_updater')

        rospy.Subscriber('/current_pose', PoseStamped, self.pose_cb, queue_size=1)
        rospy.Subscriber('/base_waypoints', Lane, self.waypoints_cb, queue_size=1)

        # TODO: Add a subscriber for /traffic_waypoint and /obstacle_waypoint below
        rospy.Subscriber('/traffic_waypoint', Int32, self.traffic_cb, queue_size=1)
        rospy.Subscriber('/obstacle_waypoint', Lane, self.obstacle_cb, queue_size=1)

        self.final_waypoints_pub = rospy.Publisher('final_waypoints', Lane, queue_size=1)
        self.current_pose = None
        self.waypoints = None
        self.red_light_wp = None
        self.start_velocity = None
        self.tl_detector_started = False

        rospy.spin()

    def pose_cb(self, msg):
        self.current_pose = msg.pose
        if self.waypoints is not None and self.tl_detector_started is True:
            self.create_final_waypoints()

    def waypoints_cb(self, msg):
        # do this once and not all the time
        if self.waypoints is None:
            self.waypoints = msg.waypoints
            self.start_velocity = self.waypoints[0].twist.twist.linear.x

    def traffic_cb(self, msg):
        self.red_light_wp = msg.data
        self.tl_detector_started = True
        if self.red_light_wp > -2:
            self.create_final_waypoints()

    def obstacle_cb(self, msg):
        # TODO: Callback for /obstacle_waypoint message. We will implement it later
        pass

    def get_waypoint_velocity(self, waypoint):
        return waypoint.twist.twist.linear.x

    def set_waypoint_velocity(self, waypoints, waypoint, velocity):
        waypoints[waypoint].twist.twist.linear.x = velocity

    def distance(self, waypoints, wp1, wp2):
        dist = 0
        dl = lambda a, b: math.sqrt((a.x-b.x)**2 + (a.y-b.y)**2  + (a.z-b.z)**2)
        for i in range(wp1, wp2+1):
            dist += dl(waypoints[wp1].pose.pose.position, waypoints[i].pose.pose.position)
            wp1 = i
        return dist

    def euclidean_distance(self, p1, p2):
        x, y, z = p1.x - p2.x, p1.y - p2.y, p1.z - p2.z
        return math.sqrt(x*x + y*y + z*z)

    def get_closest_waypoint(self, pose):
        # simply take the code from the path planning module and re-implement it here
        closest_len = 100000
        closest_waypoint = 0
        dl = lambda a, b: math.sqrt((a.x-b.x)**2 + (a.y-b.y)**2)
        for index, waypoint in enumerate(self.waypoints):
            dist = dl(pose.position, waypoint.pose.pose.position)
            if (dist < closest_len):
                closest_len = dist
                closest_waypoint = index

        return closest_waypoint

    def decelerate(self, waypoints, redlight_index):
        if len(waypoints) < 1:
            return []

        last = waypoints[redlight_index]
        last.twist.twist.linear.x = 0.

        for index, wp in enumerate(waypoints):

            if index > redlight_index:
                vel = 0.
            else:
                dist = self.euclidean_distance(wp.pose.pose.position, last.pose.pose.position)
                dist = max(0., dist - STOP_DIST)
                vel = math.sqrt(2 * MAX_DECEL * dist)
                if vel < 1.:
                    vel = 0.
            wp.twist.twist.linear.x = min(vel, wp.twist.twist.linear.x)

        return waypoints

    def create_final_waypoints(self):

        if self.current_pose is not None:
            idx_of_nearest_wp = self.get_closest_waypoint(self.current_pose)
            next_waypoints = self.waypoints[idx_of_nearest_wp:idx_of_nearest_wp+LOOKAHEAD_WPS]

            if self.red_light_wp is None or self.red_light_wp < 0:
                for i in range(len(next_waypoints) - 1):
                    self.set_waypoint_velocity(next_waypoints, i, self.start_velocity * TARGET_SPEED_METER_PER_SECOND)

            else:
                redlight_lookahead_index = max(0, self.red_light_wp - idx_of_nearest_wp)
                next_waypoints = self.decelerate(next_waypoints, redlight_lookahead_index)

            lane = Lane()
            lane.header.frame_id = '/world'
            lane.waypoints = next_waypoints

            self.final_waypoints_pub.publish(lane)


if __name__ == '__main__':
    try:
        WaypointUpdater()
    except rospy.ROSInterruptException:
        rospy.logerr('Could not start waypoint updater node.')
