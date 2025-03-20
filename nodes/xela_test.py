#!/usr/bin/env python
import numpy as np
import matplotlib.pyplot as plt
import copy
from plot_util import generate_surface, plot_sample
from functools import partial
from scipy.spatial.transform import Rotation as R
from sensor_msgs.msg import PointCloud
from geometry_msgs.msg import Point32
import math
import time
import rospy
import tf
from PFDS import PFDS
import rospkg
import PyKDL

from std_msgs.msg import Header, Float32, Float64MultiArray
from geometry_msgs.msg import WrenchStamped, Pose, Twist, TwistStamped, Point,Quaternion, PoseArray
from nav_msgs.msg import Odometry
from tf.transformations import euler_from_quaternion, quaternion_from_euler, quaternion_matrix
from std_srvs.srv import Empty, EmptyResponse

from visualization_msgs.msg import Marker

from dynamic_reconfigure.server import Server
import dynamic_reconfigure.client
# from intent_capability_hri.cfg import IntentCapabilityHRIConfig
import config.controller_config as cc


DESIRED_ROLL = np.pi
DESIRED_PITCH = np.pi/2
USE_MOCAP = False
# DESIRED_HEIGHT = 0.5 # defined in helper fun
A_Z = -3.0 # only during initialization, later determined by speed

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

class DesiredPublisher:
    def __init__(self):
        rospy.init_node('desired_publisher', anonymous=True)
        
        # Publishers
        self.ds_pub     = rospy.Publisher('/iiwa/desired_ds'   , MatrixVec, queue_size=1)
        self.odom_pub   = rospy.Publisher('/iiwa/desired_odom' , Odometry, queue_size=1)
        
        # Publishing rate
        self.rate = rospy.Rate(10)  # 10 Hz
        
        # Initialize messages
        self.ds_msg = MatrixVec()
        self.odom_msg = Odometry()
        
        # Set up constant diagonal matrix (-0.4 on diagonal)
        self.matrix = [
            -5.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, -5.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, -5.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, -15.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0, -15.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0, -15.0
        ]
            
        # Set up desired pose
        self.desired_position = Point(0.57878, -0.0522, 0.3108)
        self.desired_orientation = Quaternion(0.014, 0.7253, 0.0912, 0.682)

    def update_messages(self):
        # Update MatrixVec message
        self.ds_msg.mat = self.matrix
        self.ds_msg.pose.position = self.desired_position
        self.ds_msg.pose.orientation = self.desired_orientation
        
        # Update Odometry message
        self.odom_msg.header.stamp = rospy.Time.now()
        self.odom_msg.header.frame_id = "world"
        self.odom_msg.pose.pose.position = self.desired_position
        self.odom_msg.pose.pose.orientation = self.desired_orientation
        # Zero twist (velocity)
        self.odom_msg.twist.twist = Twist()

    def run(self):
        while not rospy.is_shutdown():
            # Update messages
            self.update_messages()
            
            # Publish messages
            self.ds_pub.publish(self.ds_msg)
            self.odom_pub.publish(self.odom_msg)
            
            # Sleep to maintain rate
            self.rate.sleep()


class FT_Sensor:
    def __init__(self, dt) -> None:
        self.listener = tf.TransformListener()
        self.dt = dt

        self.time_start = rospy.Time.now()
        self.receive_force_time = rospy.Time.now()
        self.counter = 10
        self.counter_save = 10

        self.force_frame_rotation = np.zeros((6,6))
        self.force_arr = np.zeros(6)
        self.object_pos = np.zeros(7)

        DP = DesiredPublisher()
        
        # Object position
        if USE_MOCAP:
            self.object_sub = rospy.Subscriber("/object_pos", Pose, self.object_callback, tcp_nodelay=True)
        else:
            self.desired_position = DP.desired_position
            self.desired_orientation = DP.desired_orientation
            self.object_pose = quaternion_pos_to_pose(self.desired_orientation, self.desired_position)
            self.object_sub = self.object_pose
        
        # Bota FT_SENSOR
        self.ft_sensor_sub = rospy.Subscriber('/bus0/ft_sensor0/ft_sensor_readings/wrench', WrenchStamped, self.force_callback)
        self.ft_sensor_pub = rospy.Publisher("/ft_sensor_wrench", WrenchStamped, queue_size=10)
        self.object_vis_pub = rospy.Publisher('/object_vis_marker', Marker, queue_size=10)


        self.start = False
        self.finish_srv = rospy.Service("finish_calibration", Empty, self.finish_callback)
        self.reset_srv = rospy.Service("reset_move", Empty, self.reset_callback)

        self.curr_t = 0.0

        #the string should be node name, not config type, config auto use dictionary, so no need to know the type
        # self.controller_config_client = dynamic_reconfigure.client.Client("/iiwa_cartesian_impedance_bringup", timeout=0.5)
        # , config_callback=controller_config_callback) # don't print anything
        self.move_config_client = None 


    def finish_callback(self, emp):
        self.start = True
        return EmptyResponse()
    
    def reset_callback(self, emp):
        return EmptyResponse()

    def get_optitrack_pos(self):
        try:
            (trans, rot) = self.listener.lookupTransform('Object', 'Kuka', rospy.Time(0))
        except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException):
            rospy.logerr_throttle(3.0, "No tf from Object to Kuka")
            return
        # print(trans)

    def get_wrench_rotation(self):
        try:
            (trans, rot) = self.listener.lookupTransform('world', 'ft_sensor_frame_id', rospy.Time(0))
        except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException):
            rospy.logerr_throttle(3.0, "No tf from world to FT_sensor")
            return 
        
        rotation_mat = quaternion_matrix(rot)[:3,:3]
        self.force_frame_rotation[:3,:3] = rotation_mat
        self.force_frame_rotation[-3:,-3:] = rotation_mat 
        #no need to transform torque, people are looking at the arm and know its axis


    def force_callback(self, data):
        self.receive_force_time = data.header.stamp
        self.force_arr[0] = data.wrench.force.x
        self.force_arr[1] = data.wrench.force.y
        self.force_arr[2] = data.wrench.force.z
        self.force_arr[3] = data.wrench.torque.x
        self.force_arr[4] = data.wrench.torque.y
        self.force_arr[5] = data.wrench.torque.z

        self.get_wrench_rotation()
        # F_extra  = self.force_frame_rotation[:3,:3].transpose() @ np.array([0.0, 0.0 , (-4.444)*9.81]).reshape(-1,1)
        # rospy.loginfo_throttle(1.0, "F_extra = \n{:.0f}, {:.0f}, {:.0f}".format(F_extra.flatten()[0], F_extra.flatten()[1], F_extra.flatten()[2]))
        # r_load = np.array([0,-0.04, 0.155]).reshape(-1,1) #compensate for clamps
        # torque_extra = np.cross(r_load.flatten(), F_extra.flatten()).reshape(-1,1)
        # F_extra_world = (self.force_frame_rotation[:3,:3] @ F_extra).flatten()
        # torque_extra_world =(self.force_frame_rotation[:3,:3] @ torque_extra).flatten()
        
        # rospy.loginfo_throttle(1.0, "F_extra_world = \n{:.0f}, {:.0f}, {:.0f}, torque_extra = \n{:.1f}, {:.1f}, {:.1f}".format(F_extra_world[0], F_extra_world[1], F_extra_world[2], torque_extra_world[0], torque_extra_world[1], torque_extra_world[2]))
        # self.force_arr[0:3] -= F_extra.reshape(-1)
        # self.force_arr[3:6] -= torque_extra.reshape(-1) # torque needs more thoughts
        # should the full torque when rotating be compensated, or just half??
        
        # #republish the force sensor msg with load compensated
        # data.wrench.force.x = self.force_arr[0]
        # data.wrench.force.y = self.force_arr[1]
        # data.wrench.force.z = self.force_arr[2]
        # data.wrench.torque.x = self.force_arr[3]
        # data.wrench.torque.y = self.force_arr[4]
        # data.wrench.torque.z = self.force_arr[5]
        # # self.force_torque_compensate_pub.publish(data)

        self.force_arr = (self.force_frame_rotation @ self.force_arr.reshape(-1,1)).flatten()
        # rospy.loginfo_throttle(1.0, "F_compensated = \n{:.0f}, {:.0f}, {:.0f}, torque_compensated = \n{:.1f}, {:.1f}, {:.1f}".format(self.force_arr[0], self.force_arr[1], self.force_arr[2], self.force_arr[3], self.force_arr[4], self.force_arr[5]))

    def robot_callback(self, data):
        self.ee_pos[0] = data.position.x
        self.ee_pos[1] = data.position.y
        self.ee_pos[2] = data.position.z
        self.ee_pos[3] = data.orientation.x
        self.ee_pos[4] = data.orientation.y
        self.ee_pos[5] = data.orientation.z
        self.ee_pos[6] = data.orientation.w
        self.object_pos = copy.deepcopy(self.ee_pos)

    def object_callback(self, data):
        self.object_pos[0] = data.position.x
        self.object_pos[1] = data.position.y
        self.object_pos[2] = data.position.z
        self.object_pos[3] = data.orientation.x
        self.object_pos[4] = data.orientation.y
        self.object_pos[5] = data.orientation.z
        self.object_pos[6] = data.orientation.w
        # (roll, pitch, yaw) = euler_from_quaternion (orientation_list)

    def make_ellipsoid_viz(self, pos, covMat, color_red):
        (eigValues,eigVectors) = np.linalg.eig (covMat)

        eigx_n=-PyKDL.Vector(eigVectors[0,0],eigVectors[1,0],eigVectors[2,0])
        eigy_n=-PyKDL.Vector(eigVectors[0,1],eigVectors[1,1],eigVectors[2,1])
        eigz_n=-PyKDL.Vector(eigVectors[0,2],eigVectors[1,2],eigVectors[2,2])

        rot = PyKDL.Rotation (eigx_n,eigy_n,eigz_n)
        quat = rot.GetQuaternion ()
        marker = Marker()
        #painting the Gaussian Ellipsoid Marker
        marker.header.frame_id = "world"
        marker.header.stamp = rospy.Time.now()

        marker.type = marker.SPHERE
        marker.action = marker.ADD
        marker.pose.position.x = pos[0]
        marker.pose.position.y = pos[1]
        marker.pose.position.z = pos[2]
        marker.pose.orientation.x =quat[0]
        marker.pose.orientation.y = quat[1]
        marker.pose.orientation.z = quat[2]
        marker.pose.orientation.w = quat[3]
        marker.scale.x = eigValues[0]
        marker.scale.y = eigValues[1]
        marker.scale.z =eigValues[2]
        marker.color.a = 0.5
        marker.color.r = color_red
        marker.color.g = 1.0
        marker.color.b = 0.0
        return marker

    def moving(self):

        if not self.start:
            A_UPPER_BOUND_TEST = -0.5
            A_ROT_UPPER_BOUND_TEST = -0.4
            A = np.diag([ A_UPPER_BOUND_TEST, A_UPPER_BOUND_TEST, A_UPPER_BOUND_TEST]) # x, y, z not x, y, theta as in PFDS
            desired_x = 0.8 + 0.1*np.sin(self.curr_t/2)
            desired_y = 0.3*np.sin(self.curr_t/4)
            
            b = np.array([desired_x, desired_y, DESIRED_HEIGHT]).reshape(-1,1)
            # Direct position control is a lot eaiser and don't involve two levels of stiffness
            # rospy.loginfo_throttle(1.0, self.ee_pos[:3])
            err = A @ (self.ee_pos[:3].reshape(-1,1) - b)
            
            h = Header()
            h.frame_id = 'world'
            h.stamp = rospy.Time.now()
            
            desired_twist = self.get_desired_twist(h, b)
            desired_odom = self.get_default_odom(h, b, err, desired_twist.twist.angular.z)

            A_full = np.diag([A_ROT_UPPER_BOUND_TEST, A_ROT_UPPER_BOUND_TEST, A_ROT_UPPER_BOUND_TEST, A[0,0], A[1,1], A[2,2]]) #order ang and then x y z
            desired_Ab = self.get_desired_Ab(A_full, desired_odom.pose.pose)
            
            self.curr_t = (rospy.Time.now() - self.time_start).to_sec()


            # this is not gonna be used since we set the default to be the correct values
            # self.controller_config_client.update_configuration({"stiffness_lin_z":cc.STIFFNESS_LIN_HIGH, 
            #                                                     "damping_lin_z":cc.DAMPING_LIN_HIGH, 
            #                                                     "stiffness_rot": cc.STIFFNESS_ROT_LOW, 
            #                                                     "stiffness_lin": cc.STIFFNESS_LIN_LOW,
            #                                                     "integral_lin": cc.INTEGRAL_LIN_HIGH,
            #                                                     "integral_rot": cc.INTEGRAL_ROT_HIGH})
                            
            
        self.ds_pub.publish(desired_Ab)
        self.assist_pub.publish(desired_twist)
        self.odom_pub.publish(desired_odom)
        self.pose_pub.publish(desired_odom.pose.pose)

    def config_callback(self, config, level):
        self.controller_type = config['controller_type']
        self.ascent_rate = config['ascent_rate']
        self.multiply_factor = config['multiply_factor']
        # for i in self.controllers:
        #     i.L_human[0] = config['L_human_0']
        #     i.L_human[1] = config['L_human_1']
        #     i.L_robot[0] = config['L_robot_0']
        #     i.L_robot[1] = config['L_robot_1']
        #     i.update_lambda_robot(config['lambda_robot'])
        return config


def main():
    rospy.init_node('move')
    freq=20
    r = rospy.Rate(freq)
    move = Move(1/freq)
    srv = Server(IntentCapabilityHRIConfig, move.config_callback)
    move.move_config_client =  dynamic_reconfigure.client.Client("/move", timeout=0.5)
    # , config_callback=move_config_callback)
    while not rospy.is_shutdown():
        move.moving()
        r.sleep()



if __name__ == '__main__':
    try:
        main()
    except rospy.ROSInterruptException:
        pass
