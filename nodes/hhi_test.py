#!/usr/bin/env python3
# #This is a ros node that republishes the topic from sensors to change some features.

import numpy as np
import matplotlib.pyplot as plt
import copy
from sensor_msgs.msg import PointCloud
from geometry_msgs.msg import Pose, WrenchStamped, Quaternion, Point, Point32
from std_msgs.msg import Float32, String
from xela_server_ros.msg import SensStream
from visualization_msgs.msg import Marker
import math
import time
import rospy
import tf
import tf.transformations as tf_transform
from rosbag.bag import Bag
from subprocess import Popen, PIPE
from pathlib import Path
import inspect
import os
import glob
import subprocess
import signal
import sys


# Functions
def skew_sym(vec):
    return np.array([[0, -vec[2], vec[1]], [vec[2], 0, -vec[0]], [-vec[1], vec[0], 0]])
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

class SpecificMessageException(Exception):
    pass

output_bag = None
start_time = None   
xela_data = None
ft_wrench_data = None
robot_ee_pose = None    
rate = 20
    
#Jh = [eye(3) zeros(3,3);
    # -skew_sym([0,0,-0.15]),eye(3)];
# /bus0/ft_sensor0/ft_sensor_readings/wrench    35530 msgs : geometry_msgs/WrenchStamped 
def init_bag_files():
    test = '2024-08-04-18-54-52_ft_human_6d'
    input_bag_dir = '/home/yifei/kuka_ws/bags/6d_exp/'+test
    input_bag = input_bag_dir+".bag"
    output_bag_dir = input_bag_dir + '/repub/'
    os.makedirs(output_bag_dir, exist_ok=True)
    output_bag_path = os.path.join(output_bag_dir, (test + "_repub.bag"))
    if os.path.exists(output_bag_path):
        os.remove(output_bag_path)
    output_bag = Bag(output_bag_path, 'w')
    return input_bag, output_bag

def bag_write_callback(msg, topic):
    global output_bag
    output_bag.write(topic, msg)

# Publishers
# object_vis_pub = rospy.Publisher('/object_vis_marker', Marker, queue_size=10)
xela_data_pub = rospy.Publisher('/xela_data', SensStream, queue_size=rate)
ft_wrench_pub = rospy.Publisher('/ft_wrench_data', WrenchStamped, queue_size=rate)
robot_ee_pose_pub = rospy.Publisher('/iiwa_ee_pose', Pose, queue_size=rate)
tf_world_to_ee_pub = rospy.Publisher('/tf_world_ee', Pose, queue_size=rate)
tf_world_to_ft_pub = rospy.Publisher('/tf_world_ft', Pose, queue_size=rate)

def robot_callback(msg):
    global robot_ee_pose
    robot_ee_pose = Pose()
    robot_ee_pose = copy.deepcopy(msg)
    robot_ee_pose_pub.publish(robot_ee_pose)
    rospy.loginfo("Received robot pose: %s", msg.position)

def ft_callback(msg):
    global ft_wrench_data
    ft_wrench_data = WrenchStamped()
    ft_wrench_data = copy.deepcopy(msg)
    ft_wrench_pub.publish(ft_wrench_data)
    rospy.loginfo("Received force: %s", msg.wrench.force)

def xela_callback(msg):
    global xela_data
    xela_data = SensStream()
    xela_data = copy.deepcopy(msg)
    xela_data_pub.publish(xela_data)


def listener():
    global output_bag, rate
    # In ROS, nodes are uniquely named. If two nodes with the same name are launched, the 
    # previous one is kicked off. The anonymous=True flag means that rospy will choose a 
    # unique name for our 'listener' node so that multiple listeners can run simultaneously.
    rospy.init_node('listener', anonymous=True)

    input_bag, output_bag = init_bag_files()
    # rospy.loginfo("Playing {}".format(input_bag))
    # play_process = Popen(['rosbag', 'play', input_bag], stdout=PIPE, stderr=PIPE)
    # play_process = os.system("rosbag play {}".format(input_bag))
    
    # Subscribers
    rospy.Subscriber('xServTopic', SensStream, xela_callback)
    rospy.Subscriber('/bus0/ft_sensor0/ft_sensor_readings/wrench', WrenchStamped, ft_callback)
    # rospy.Subscriber("/bus0/ft_sensor0/ft_sensor_readings/reading", Reading, ft_reading_callback)
    rospy.Subscriber("/iiwa/task_states", Pose, robot_callback)
    rate = rospy.Rate(rate) # 10hz

    tflistener = tf.TransformListener()
    trans_ee = None
    trans_ft = None
    while trans_ee is None or trans_ft is None:
        try:
            (trans_ee, rot_ee) = tflistener.lookupTransform('world', 'iiwa_link_ee', rospy.Time(0))        
            (trans_ft, rot_ft) = tflistener.lookupTransform('world', 'ft_sensor_frame_id', rospy.Time(0))
            tf_world_to_ee = quaternion_pos_to_pose(rot_ee, trans_ee)
            tf_world_to_ft = quaternion_pos_to_pose(rot_ft, trans_ft)
            print("tf world-to-ee:  ", tf_world_to_ee)
            print("tf world-to-ft:  ", tf_world_to_ft)
            tf_world_to_ee_pub.publish(tf_world_to_ee)
            tf_world_to_ft_pub.publish(tf_world_to_ft)
        except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException):
            rospy.logerr_throttle(3.0, "No tf data")
        
    
    # print(input_bag)
    # Start the process
    play_process = subprocess.Popen(['rosbag', 'play', input_bag], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    # exit()

    while not rospy.is_shutdown(): 
        # Extra calculations if needed

        messages = {
                '/xela_data': xela_data,
                '/ft_wrench_data': ft_wrench_data,
                '/robot_ee_pose': robot_ee_pose,
                '/tf_world_ee': tf_world_to_ee,
                '/tf_world_ft': tf_world_to_ft
                }

        for topic, message in messages.items():
            output_bag.write(topic, message)

        rate.sleep()

        # Check if the process has ended
        if play_process.poll() is not None:
            # Get the output
            stdout, stderr = play_process.communicate()

            # Check the output for the specific message
            if "process has finished cleanly" in stdout.decode() or "process has finished cleanly" in stderr.decode():
                raise SpecificMessageException()


if __name__ == '__main__':
    # listener()
    try:
        listener()
    except KeyboardInterrupt:
        print("Interrupt received")
    except ProcessLookupError:
        print("ProcessLookupError")
    # except Exception as e:
    #     print("Unexpected error:", e)
    # except SpecificMessageException():
    #     print("Specific message received, exiting...")
    #     sys.exit()
    except rospy.ROSInterruptException:
        pass
    finally:
        output_bag.close()
        print("Bag file closed")
    