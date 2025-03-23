#!/usr/bin/env python
import numpy as np
import matplotlib.pyplot as plt
import copy
from functools import partial
from scipy.spatial.transform import Rotation as R
from sensor_msgs.msg import PointCloud
from geometry_msgs.msg import Point32
import math
import time
import rospy
import tf
import rospkg

from std_msgs.msg import Header, Float32, Float64MultiArray
from geometry_msgs.msg import WrenchStamped, Pose, Twist, TwistStamped, Point,Quaternion, PoseArray
from nav_msgs.msg import Odometry
# from rokubimini_msgs import Reading
from xela_server_ros.msg import SensStream
from visualization_msgs.msg import Marker

from tf.transformations import euler_from_quaternion, quaternion_from_euler, quaternion_matrix
from std_srvs.srv import Empty, EmptyResponse

from dynamic_reconfigure.server import Server
import config.controller_config as cc


def quaternion_pos_to_pose(q, p):
    pose = Pose()
    pose.orientation = Quaternion(x= q[0], y= q[1], z= q[2], w= q[3])
    pose.position = Point(x = p[0], y = p[1], z = p[2])
    return pose
def make_text(text, attractor):
    marker = Marker()
    marker.header.frame_id = "world"
    marker.header.stamp = rospy.Time.now()
    marker.type = marker.TEXT_VIEW_FACING
    marker.action = marker.ADD
    marker.pose.position.x = attractor[0]
    marker.pose.position.y = attractor[1]
    marker.pose.position.z = attractor[2] + 1.0 # yaw
    marker.scale.z = 0.05
    marker.text = text
    marker.color.a = 1.0  # Alpha
    # white text  
    marker.color.r = 1.0  # Red
    marker.color.g = 1.0  # Green
    marker.color.b = 1.0  # Blue
    return marker
def make_sphere(x,y, z, radius, color = [1.0, 0.0, 0.0]):
    marker = Marker()
    marker.header.frame_id = "world"
    marker.header.stamp = rospy.Time.now()

    marker.type = marker.SPHERE
    marker.action = marker.ADD

    marker.pose.position.x = x
    marker.pose.position.y = y
    marker.pose.position.z = z

    marker.pose.orientation.x = 0.0
    marker.pose.orientation.y = 0.0
    marker.pose.orientation.z = 0.0
    marker.pose.orientation.w = 1.0

    marker.scale.x = radius  # Radius
    marker.scale.y = radius  # Radius
    marker.scale.z = radius  # Radius

    marker.color.a = 1.0  # Alpha
    marker.color.r = color[0]  # Red
    marker.color.g = color[1]  # Green
    marker.color.b = color[2]  # Blue
    return marker

class SensorSyncPublisher(object):

    def __init__(self):
        rospy.init_node('sensor_sync_publisher', anonymous=True)
        
        # Publishers
        # self.ft_wrench_pub  = rospy.Publisher('/bus0/ft_sensor0/ft_sensor_readings/wrench', WrenchStamped, queue_size=10)
        # self.ft_reading_pub = rospy.Publisher("/bus0/ft_sensor0/ft_sensor_readings/reading", Reading, queue_size=10)
        # self.object_vis_pub = rospy.Publisher('/object_vis_marker', Marker, queue_size=10)
        self.xela_data_pub = rospy.Publisher('/xela_data', SensStream, queue_size=10)
        self.ft_wrench_pub = rospy.Publisher('/ft_wrench_data', WrenchStamped, queue_size=10)
        self.robot_pose_pub = rospy.Publisher('/iiwa_ee_pose', Pose, queue_size=10)
        self.tf_world_to_ft_pub = rospy.Publisher('/ft_world_tf', Pose, queue_size=10)

        # Subscribers
        self.listener = tf.TransformListener()
        self.xela_sub = rospy.Subscriber('xServTopic', SensStream, self.xela_callback, tcp_nodelay=True)
        self.ft_wrench_sub = rospy.Subscriber('/bus0/ft_sensor0/ft_sensor_readings/wrench', WrenchStamped, self.ft_callback)
        # self.ft_reading_sub = rospy.Subscriber("/bus0/ft_sensor0/ft_sensor_readings/reading", Reading, self.ft_reading_callback, tcp_nodelay=True)
        self.robot_state_sub = rospy.Subscriber("/iiwa/task_states", Pose, self.robot_callback, tcp_nodelay=True)

        # Buffers to hold latest sensor data
        self.xela_data = None
        self.ft_wrench_data = None
        self.robot_pose_data = None
        self.tf_world_to_ft = Pose()

        # Publishing rate
        self.rate = rospy.Rate(100)  # 100 Hz

    def xela_callback(self, msg):
        self.xela_data = msg
        sensors = msg.sensors
        rospy.loginfo("-------------------------")
        rospy.loginfo("Broadcast: %s sensor(s)", len(sensors))
        for sensor in sensors:
            rospy.loginfo("Model `%s` at message `%s` with %d taxels and %s calibration",
                          sensor.model, sensor.message, len(sensor.taxels) / 3,
                          "with" if sensor.forces else "without")
            
    def ft_callback(self, msg):
        self.ft_wrench_data = msg
        rospy.loginfo("Received force: %s", msg.wrench.force)

    def robot_callback(self, msg):
        self.robot_pose_data = msg

    def get_wrench_rotation(self):
        try:
            (trans, rot) = self.listener.lookupTransform('world', 'ft_sensor_frame_id', rospy.Time(0))
            self.tf_world_to_ft = quaternion_pos_to_pose(rot, trans)
        except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException):
            rospy.logerr_throttle(3.0, "No tf from world to FT_sensor")
            return      
        
    def run(self):
        while not rospy.is_shutdown():
            # Re-publish messages
            if self.xela_data and self.ft_wrench_data and self.robot_pose_data:
                self.xela_data_pub.publish(self.xela_data)
                self.ft_wrench_pub.publish(self.ft_wrench_data)
                self.robot_pose_pub.publish(self.robot_pose_data)
            if self.robot_pose_data:
                self.tf_world_to_ft_pub.publish(self.tf_world_to_ft)
            self.rate.sleep()


def main():
    try:
        publisher = SensorSyncPublisher()
        publisher.run()
    except rospy.ROSInterruptException:
        pass


if __name__ == '__main__':
    print("Starting Sensor Sync Publisher")
    main()
