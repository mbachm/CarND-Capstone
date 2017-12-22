#!/usr/bin/env python

import unittest
import rospy
from geometry_msgs.msg import PoseStamped, Quaternion
from styx_msgs.msg import Lane, Waypoint
import tf
import time
import roslaunch

package = 'waypoint_updater'
executable = 'waypoint_updater.py'
node_name = 'waypoint_updater'

## A sample python unit test
class TestwaypointUpdater(unittest.TestCase):

    def __init__(self, *args):
        super(TestwaypointUpdater, self).__init__(*args)
        self.shouldBeFalseAfterTests = False
        self.shouldBeTrueAfterTests = False

    def run_waypoint_updater_node(self):
        node = roslaunch.core.Node(package=package, node_type=executable, name=node_name)
        launch = roslaunch.scriptapi.ROSLaunch()
        launch.start()
        self.process = launch.launch(node)

    def create_lane_object_to_publish(self):
        waypoints = []
        p1 = Waypoint()
        p1.pose.pose.position.x = float(909.48)
        p1.pose.pose.position.y = float(1128.67)
        p1.pose.pose.position.z = float(0.0)
        q1 = self.quaternion_from_yaw(float(0))
        p1.pose.pose.orientation = Quaternion(*q1)
        p1.twist.twist.linear.x = float(10)
        waypoints.append(p1)
        lane = Lane()
        lane.header.frame_id = '/world'
        lane.header.stamp = rospy.Time(0)
        lane.waypoints = waypoints
        self.lane = lane

    def create_position_to_publish(self):
        pos = PoseStamped()
        pos.pose.position.x = float(900.0)
        pos.pose.position.y = float(1128.0)
        pos.pose.position.z = float(0.0)
        q1 = self.quaternion_from_yaw(float(0))
        pos.pose.orientation = Quaternion(*q1)
        self.position = pos

    def quaternion_from_yaw(self, yaw):
        return tf.transformations.quaternion_from_euler(0., 0., yaw)

    def no_updates_expected(self, msg):
        self.shouldBeFalseAfterTests = True

    def updates_expected(self, msg):
        self.shouldBeTrueAfterTests = True

    def test_initially_only_publish_waypoints_not_result_in_update(self):
        # given
        self.run_waypoint_updater_node()
        rospy.init_node('test_node', anonymous=True)
        base_waypoints_pub = rospy.Publisher('/base_waypoints', Lane, queue_size=1)
        rospy.Subscriber('/final_waypoints', Lane, self.no_updates_expected)
        self.create_lane_object_to_publish()

        # when
        base_waypoints_pub.publish(self.lane)

        # then
        timeout_t = time.time() + 3.0 # 3 seconds
        while not rospy.is_shutdown() and not self.shouldBeFalseAfterTests and time.time() < timeout_t:
            time.sleep(0.1)
        self.assertFalse(self.shouldBeFalseAfterTests)

        # cleanup
        self.process.stop()


    def test_initially_only_publish_position_not_result_in_update(self):
        # given
        self.run_waypoint_updater_node()
        rospy.init_node('test_node', anonymous=True)
        current_pose_pub = rospy.Publisher('/current_pose', PoseStamped, queue_size=1)
        rospy.Subscriber('/final_waypoints', Lane, self.no_updates_expected)
        self.create_position_to_publish()

        # when
        current_pose_pub.publish(self.position)

        # then
        timeout_t = time.time() + 3.0 # 3 seconds
        while not rospy.is_shutdown() and not self.shouldBeFalseAfterTests and time.time() < timeout_t:
            time.sleep(0.1)
        self.assertFalse(self.shouldBeFalseAfterTests)

        # cleanup
        self.process.stop()


    def test_waypoints_are_updated(self):
        # given
        self.run_waypoint_updater_node()
        self.wasCalled = False
        current_pose_pub = rospy.Publisher('/current_pose', PoseStamped, queue_size=1)
        base_waypoints_pub = rospy.Publisher('/base_waypoints', Lane, queue_size=1)
        rospy.Subscriber('/final_waypoints', Lane, self.updates_expected)
        rospy.init_node('test_node', anonymous=True)
        self.create_lane_object_to_publish()
        self.create_position_to_publish()

        timeout_t = time.time() + 3.0 # 3 seconds
        while not rospy.is_shutdown() and not self.shouldBeTrueAfterTests and time.time() < timeout_t:
            # when
            base_waypoints_pub.publish(self.lane)
            current_pose_pub.publish(self.position)
            time.sleep(0.1)

        # then
        self.assertTrue(self.shouldBeTrueAfterTests)

        # cleanup
        self.process.stop()

if __name__ == '__main__':
    import rosunit
    rosunit.unitrun('waypoint_updater_test', 'test_bare_bones', TestwaypointUpdater)