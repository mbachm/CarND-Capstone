#!/usr/bin/env python

import rospy
from geometry_msgs.msg import PoseStamped
from styx_msgs.msg import Lane, Waypoint

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


class WaypointUpdater(object):
    def __init__(self):
        rospy.init_node('waypoint_updater')

        rospy.Subscriber('/current_pose', PoseStamped, self.pose_cb)
        rospy.Subscriber('/base_waypoints', Lane, self.waypoints_cb)

        # TODO: Add a subscriber for /traffic_waypoint and /obstacle_waypoint below


        self.final_waypoints_pub = rospy.Publisher('final_waypoints', Lane, queue_size=1)

        # TODO: Add other member variables you need below
        self.waypoints = None
        self.waypoints_lenght = None
        self.current_pose = None
        self.idx_of_nearest = None

        rospy.loginfo("Init")
        rospy.spin()

    def pose_cb(self, msg):
        # TODO: Implement
        rospy.loginfo("Pose cb")
        self.current_pose = msg.pose
        if self.waypoints is not None:
            self.create_final_waypoints()
        pass

    def waypoints_cb(self, msg):
        # TODO: Implement
        rospy.loginfo("Waypoints cb")
        if self.waypoints is None:
            self.waypoints = msg.waypoints
            self.waypoints_lenght = len(msg.waypoints)
            self.create_final_waypoints()
        pass

    def traffic_cb(self, msg):
        # TODO: Callback for /traffic_waypoint message. Implement
        pass

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

    def get_initial_nearest_waypoint(self):
        min_dist = float("inf")
        for idx in range(self.waypoints_lenght):
            d = self.euclidean_distance(self.waypoints[idx].pose.pose.position, self.current_pose.position)
            if d < min_dist:
                min_dist = d
                self.idx_of_nearest = idx

    def create_final_waypoints(self):
        if self.waypoints is not None and self.current_pose is not None:
            if self.idx_of_nearest is None:
                #initial waypoint search
                self.get_initial_nearest_waypoint()

            if self.idx_of_nearest is not None:
                d1 = self.euclidean_distance(self.waypoints[self.idx_of_nearest].pose.pose.position, self.current_pose.position)
                d2 = self.euclidean_distance(self.waypoints[(self.idx_of_nearest+1) % self.waypoints_lenght].pose.pose.position, self.current_pose.position)
                if d2 < d1:
                    self.idx_of_nearest += 1
                    self.idx_of_nearest %= self.waypoints_lenght

                next_waypoints = [self.waypoints[i] for i in range(len(self.waypoints)) if self.idx_of_nearest <= i < self.idx_of_nearest + LOOKAHEAD_WPS]
                lane = Lane()
                lane.waypoints = next_waypoints
                lane.header.frame_id = '/world'

                self.final_waypoints_pub.publish(lane)





if __name__ == '__main__':
    try:
        WaypointUpdater()
    except rospy.ROSInterruptException:
        rospy.logerr('Could not start waypoint updater node.')
