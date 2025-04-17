#!/usr/bin/env python3
# #This is a ros node that republishes the topic from sensors to change some features.

import numpy as np
import matplotlib.pyplot as plt
import copy
from sensor_msgs.msg import PointCloud
from geometry_msgs.msg import Point, Pose, WrenchStamped
from geometry_msgs.msg import Point32
from std_msgs.msg import Float32, String
import math
import time
import rospy
import tf
from dynamic_reconfigure.msg import Config
import dynamic_reconfigure.client as Client
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

class SpecificMessageException(Exception):
    pass

output_bag = None
start_time = None   
human_force_data = None
tank_repub = None
tank_derivative_repub = None
ee_vel = None
ee_vel_repub = None
ee_pos = None
ee_pos_repub = None
desired_ds = None
desired_A = None
desired_goal = None
damping_gain = None
confidence = None
confidence_derivative = None

# variables

DESIRED_ROLL = np.pi
DESIRED_PITCH = np.pi/2
USE_MOCAP = False
DESIRED_HEIGHT = 0.324
A_Z = -3.0
rate=20

robot_r = np.zeros((3,1))
human_r = np.zeros((3,1))
ee_vel = np.zeros(3)
human_force = np.zeros((6,1))
desired_A  = -np.eye(3)
desired_goal = np.zeros(2)
tank_state = 0.0
tank_state_rot = 0.0
grasp_matrix = np.zeros((6,6))
damping_gain = np.ones(2)
ee_pos = np.zeros(7)
tank_state_derivative = 0.0
    
#Jh = [eye(3) zeros(3,3);
    # -skew_sym([0,0,-0.15]),eye(3)];
# /bus0/ft_sensor0/ft_sensor_readings/wrench    35530 msgs : geometry_msgs/WrenchStamped 

particle_pub = rospy.Publisher("/particles_repub", PointCloud, queue_size=rate)
force_pub = rospy.Publisher("/force_repub", Float32, queue_size=rate)
torque_pub = rospy.Publisher("/torque_repub", Float32, queue_size=rate)
W_dot_pub = rospy.Publisher("/W_dot_repub", Point, queue_size=rate)
W_dot_single_pub = rospy.Publisher("/W_dot_single_repub", Float32, queue_size=rate)
robot_task_states_pub = rospy.Publisher("/robot_task_state_repub", Pose, queue_size=rate)
robot_ee_pose_pub = rospy.Publisher("/robot_ee_pose_repub", Pose, queue_size=rate)
robot_ee_velocity_pub = rospy.Publisher("/robot_ee_velocity_repub", Point, queue_size=rate)
tank_state_pub = rospy.Publisher("/tank_state_repub", Float32, queue_size=rate)
tank_state_derivative_pub = rospy.Publisher("/tank_state_derivative_repub", Float32, queue_size=rate)
human_force_pub = rospy.Publisher("/human_force_repub", WrenchStamped, queue_size=rate)
desired_ds_pub = rospy.Publisher("/desired_ds_repub", MatrixVec, queue_size=rate)
damping_gain_pub = rospy.Publisher("/damping_gain_repub", Float32, queue_size=rate)

#  according to (101) (eq order may change, the eq \dot W = \dot x * u' - c \dot x  lambda \dot x - \dot c /2(x - x_goal).....
# first term is \dot x u', second term -c...., third term -\dot c /2 ...

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

def skew_sym(vec):
    return np.array([[0, -vec[2], vec[1]], [vec[2], 0, -vec[0]], [-vec[1], vec[0], 0]])

def robot_callback(data):
    global ee_pos, ee_pos_repub
    ee_pos[0] = data.position.x
    ee_pos[1] = data.position.y
    ee_pos[2] = data.position.z
    ee_pos[3] = data.orientation.x
    ee_pos[4] = data.orientation.y
    ee_pos[5] = data.orientation.z
    ee_pos[6] = data.orientation.w
    ee_pos_repub = Pose()
    ee_pos_repub = copy.deepcopy(data)
    robot_ee_pose_pub.publish(ee_pos_repub)
    
def robot_vel_callback(data):
    global ee_vel, ee_vel_repub
    alpha = 0.05
    ee_vel[0] = alpha * data.position.x + (1-alpha) * ee_vel[0]
    ee_vel[1] = alpha * data.position.y + (1-alpha) * ee_vel[1]
    ee_vel[2] = alpha * data.position.z + (1-alpha) * ee_vel[2]
    ee_vel_repub = Point()
    ee_vel_repub.x = ee_vel[0]
    ee_vel_repub.y = ee_vel[1]
    ee_vel_repub.z = ee_vel[2]
    robot_ee_velocity_pub.publish(ee_vel_repub)

def damping_gain_callback(msg):
    # global desired_A
    global damping_gain
    for param in msg.doubles:
        # print(type(param), "Parameter: {} Value: {}".format(param.name, param.value))
        if param.name == "damping_lin": #"damping_gain":
            damping_gain = np.array(2*[param.value])
            # Add handling for other parameter types (ints, strs, etc.) as needed
    damping_gain_pub.publish(Float32(damping_gain[0]))
    print("damping_gain = ", damping_gain)
    # damping_gain = np.array(data.doubles)
    # desired_A = desired_A * damping_gain[0]
    # print("desired_A = ", desired_A)
    # print("desired_goal = ", desired_goal)

def desired_ds_callback(data):
    global desired_A, desired_goal, desired_ds
    desired_A = np.array(data.mat).reshape(6,6)[:2,:2]
    desired_goal = np.array([data.pose.position.x, data.pose.position.y])
    desired_ds = MatrixVec()
    desired_ds = copy.deepcopy(data)
    desired_ds_pub.publish(desired_ds)

def force_callback(data):
    global human_force, human_force_data
    force_vec = np.array([data.wrench.force.x, data.wrench.force.y, data.wrench.force.z])
    force_pub.publish(np.linalg.norm(force_vec))
    torque_vec = np.array([data.wrench.torque.x, data.wrench.torque.y, data.wrench.torque.z])
    torque_pub.publish(np.linalg.norm(torque_vec))
    human_force = np.hstack([force_vec, torque_vec])
    human_force_data = WrenchStamped()
    human_force_data.wrench.force.x = force_vec[0]
    human_force_data.wrench.force.y = force_vec[1]
    human_force_data.wrench.force.z = force_vec[2]
    human_force_data.wrench.torque.x = torque_vec[0]
    human_force_data.wrench.torque.y = torque_vec[1]
    human_force_data.wrench.torque.z = torque_vec[2]
    human_force_pub.publish(human_force_data)

def tank_state_callback(data):
    global tank_state#, tank_state_derivative, tank_repub, tank_derivative_repub
    # tank_state_derivative = (data.data - tank_state) / 0.1 # 10hz
    tank_state = data.data
    # tank_repub = Float32(tank_state)
    # tank_derivative_repub = Float32(tank_state_derivative)
    # tank_state_pub.publish(tank_repub)
    # tank_state_derivative_pub.publish(tank_derivative_repub)

def tank_state_rot_callback(data):
    global tank_state_rot
    # tank_state_derivative = (data.data - tank_state) / 0.1 # 10hz
    tank_state_rot = data.data
    # tank_repub = Float32(tank_state)
    # tank_derivative_repub = Float32(tank_state_derivative)
    # tank_state_pub.publish(tank_repub)
    # tank_state_derivative_pub.publish(tank_derivative_repub)

def particles_pub_callback(data):
    # particles_pub_msg = PointCloud()
    # particles_pub_msg.header = h
    for i in range(len(data.points)):
        #Change z of all particles to DESIRED_HEIGHT
        data.points[i].z = DESIRED_HEIGHT
    particle_pub.publish(data)


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
    
    rospy.Subscriber("/particles", PointCloud, particles_pub_callback)
    rospy.Subscriber("/bus0/ft_sensor0/ft_sensor_readings/wrench", WrenchStamped, force_callback)
    rospy.Subscriber("/iiwa/ee_vel_cartimp", Pose, robot_vel_callback)
    rospy.Subscriber("/tank_state", Float32, tank_state_callback)
    rospy.Subscriber("/tank_state_rot", Float32, tank_state_rot_callback)
    rospy.Subscriber("/iiwa/desired_ds", MatrixVec, desired_ds_callback)
    rospy.Subscriber("/iiwa_cartesian_impedance_bringup/parameter_updates",Config, damping_gain_callback)
    rospy.Subscriber("/iiwa/task_states", Pose, robot_callback)
    rate = rospy.Rate(rate) # 10hz

    tflistener = tf.TransformListener()
    trans = None
    trans2 = None
    while trans is None or trans2 is None:
        try:
            (trans, rot) = tflistener.lookupTransform('iiwa_link_pot', 'iiwa_link_ee', rospy.Time(0))        
            (trans2, rot2) = tflistener.lookupTransform('iiwa_link_pot', 'iiwa_link_human', rospy.Time(0))
        except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException):
            rospy.logerr_throttle(3.0, "No tf data")
    print("TF from pot to ee", trans)       # TF from pot to ee [0.01, 0.01, -0.16]
    print("TF from pot to human", trans2)   # TF from pot to human [0.01, 0.01, 0.3231]
    print("rot from pot to ee", rot)        # rot from pot to ee [0.0, 0.0, 0.0, 1.0]
    print("rot from pot to human", rot2)    # rot from pot to human [0.0, 0.0, 0.707, 0.707]
    
    grasp_matrix[:3,:3] = np.eye(3)
    grasp_matrix[3:6,3:6] = np.eye(3)
    grasp_matrix[3:6,:3] = -skew_sym(-np.array(trans2))
    #where ri is the vector from Σei to Σo described in the payload frame according to paper
    # here trans 2 is from o to ei, so negate it
    # print(input_bag)
    # Start the process
    play_process = subprocess.Popen(['rosbag', 'play', input_bag], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    # exit()

    while not rospy.is_shutdown(): 
        # this should be computed for 2d!! third dim is rotation
        #compute W_dot
        confidence = -tank_state
        confidence_derivative = -tank_state_derivative
        #first term \dot x * u'
        u_prime = grasp_matrix @ human_force
        # now convert u_prime to world frame
        u_prime_xyz = tf_transform.quaternion_matrix(ee_pos[3:])[:3,:3]@ u_prime[:3]
        W_dot_first_term = np.dot(ee_vel[:2], u_prime_xyz[:2])
        # print(W_dot_first_term)
        #second term -c \dot x lambda \dot x
        c_mul_lambda = np.diag(damping_gain)
        # print("c_mul_lambda = ", c_mul_lambda.shape)
        W_dot_second_term = -confidence * np.dot(ee_vel[:2], c_mul_lambda @ ee_vel[:2].reshape(-1,1))
        # third term -\dot c /2 (x - x_goal).T lambda A (x - x_goal)
        W_dot_third_term = -confidence_derivative / 2.0 * np.dot((ee_pos[:2] - desired_goal), np.diag(2*[85.0]) @ desired_A @ (ee_pos[:2] - desired_goal).reshape(-1,1))
        # W_dot_third_term = -confidence_derivative / 2.0 * np.dot((ee_pos[:2] - desired_goal), c_mul_lambda @ desired_A @ (ee_pos[:2] - desired_goal).reshape(-1,1))
        W_dot = Point()
        W_dot.x = W_dot_first_term
        W_dot.y = W_dot_second_term
        W_dot.z = W_dot_third_term
        W_dot_pub.publish(W_dot)

        W_dot_instance = Float32()
        W_dot_instance.data = W_dot_first_term + W_dot_second_term + W_dot_third_term
        W_dot_single_pub.publish(W_dot_instance)

        # print("W dot = ", W_dot_first_term + W_dot_second_term + W_dot_third_term)

        messages = {
                '/W_dot_repub': W_dot,
                '/W_dot_single_repub': W_dot_instance,
                '/human_force_repub': human_force_data,
                '/tank_state_repub': tank_repub, 
                '/tank_state_derivative_repub': tank_derivative_repub,
                '/robot_ee_velocity_repub': ee_vel_repub,
                '/robot_ee_pose_repub': ee_pos_repub,
                '/desired_ds_repub': desired_ds,
                '/damping_gain_repub': Float32(damping_gain[0])
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
    