#!/usr/bin/env python3
import rospy
import numpy as np
from pfds_msgs.msg import MatrixVec
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Pose, Twist, Point, Quaternion

class DesiredPublisher:
    def __init__(self):
        rospy.init_node('desired_publisher', anonymous=True)
        
        # Publishers
        self.ds_pub = rospy.Publisher('/iiwa/desired_ds', MatrixVec, queue_size=1)
        self.odom_pub = rospy.Publisher('/iiwa/desired_odom', Odometry, queue_size=1)
        
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

if __name__ == '__main__':
    try:
        publisher = DesiredPublisher()
        publisher.run()
    except rospy.ROSInterruptException:
        pass
