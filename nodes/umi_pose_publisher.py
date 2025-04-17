#!/usr/bin/env python

import cv2
import numpy as np
import pyrealsense2 as rs
import apriltag
import rospy
from geometry_msgs.msg import PoseStamped
import tf.transformations as tft
import tf
from geometry_msgs.msg import Vector3
from visualization_msgs.msg import Marker
import os

np.set_printoptions(suppress=True)

def create_text_marker(text, position, frame_id="world", marker_id=0):
    marker = Marker()
    marker.header.frame_id = frame_id
    marker.header.stamp = rospy.Time.now()
    marker.ns = "text_markers"
    marker.id = marker_id
    marker.type = Marker.TEXT_VIEW_FACING
    marker.action = Marker.ADD
    marker.pose.position.x = position[0]
    marker.pose.position.y = position[1]
    marker.pose.position.z = position[2]
    marker.pose.orientation.x = 0.0
    marker.pose.orientation.y = 0.0
    marker.pose.orientation.z = 0.0
    marker.pose.orientation.w = 1.0
    marker.scale.z = 0.1  # Text size
    marker.color.a = 1.0  # Alpha (transparency)
    marker.color.r = 1.0  # Red
    marker.color.g = 1.0  # Green
    marker.color.b = 1.0  # Blue
    marker.text = text
    return marker

def create_pose_msg(matrix, frame_id):
    pose_msg = PoseStamped()
    pose_msg.header.stamp = rospy.Time.now()
    pose_msg.header.frame_id = frame_id
    pose_msg.pose.position.x = matrix[0, 3]
    pose_msg.pose.position.y = matrix[1, 3]
    pose_msg.pose.position.z = matrix[2, 3]
    quat = tft.quaternion_from_matrix(matrix)
    pose_msg.pose.orientation.x = quat[0]
    pose_msg.pose.orientation.y = quat[1]
    pose_msg.pose.orientation.z = quat[2]
    pose_msg.pose.orientation.w = quat[3]
    return pose_msg

# Transform from camera to world frame (can be replaced with OptiTrack)
# rotate to switch z and x axis
# camera_to_world = np.array([[0, 0, 1, 1],
#                            [0, 1, 0, 1],
#                            [-1, 0, 0, 1],
#                            [0, 0, 0, 1]])
# rotate to switch z and x axis
# camera_to_world = np.array([[0, 0, -1, 0],
#                            [0, 1, 0, 0],
#                            [1, 0, 0, 0],
#                            [0, 0, 0, 1]])
camera_to_world = np.array([[1, 0, 0, 1],
                            [0, 1, 0, 1],
                            [0, 0, 1, 1],
                            [0, 0, 0, 1]])


# Initialize ROS node
rospy.init_node('umi_gripper_pose_publisher')
umi_pose_pub = rospy.Publisher('/umi_ee/umi_pose_camera', PoseStamped, queue_size=10)
base_pose_pub = rospy.Publisher('/umi_ee/base_pose_camera', PoseStamped, queue_size=10)
umi_pose_world_pub = rospy.Publisher('/umi_ee/umi_pose_base', PoseStamped, queue_size=10)
camera_pose_pub = rospy.Publisher('/umi_ee/camera_pose', PoseStamped, queue_size=10)

marker_pub = rospy.Publisher('/umi_ee/markers', Marker, queue_size=10)

tf_broadcaster = tf.TransformBroadcaster()

# Load calibration: umi_tag → gripper transform
def find_transform(tag_id):
    # Load the transforms from a file if the file exists
    try:
        # load file from the same folder as this node
        file_path = os.path.dirname(os.path.abspath(__file__))
        # print("file_path", file_path)

        transforms = np.load(os.path.join(file_path, 'calibrated_transforms.npz'), allow_pickle=True)
        return transforms[f'from_{tag_id}']
    except FileNotFoundError:
        # rospy.logerr("Calibration file not found.")
        umi_to_gripper = np.array([[1, 0, 0, 0],
                                   [0, 1, 0, 0],
                                   [0, 0, 1, 0.2],
                                   [0, 0, 0, 1]])
        return umi_to_gripper    



# Start RealSense
pipe = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 60)

options = apriltag.DetectorOptions(families='tag36h11', nthreads=4)
detector = apriltag.Detector(options)

profile = pipe.start(config)
color_profile = profile.get_stream(rs.stream.color)
intr = color_profile.as_video_stream_profile().get_intrinsics()

K = np.array([[intr.fx, 0, intr.ppx],
              [0, intr.fy, intr.ppy],
              [0, 0, 1]])
camera_params = (intr.fx, intr.fy, intr.ppx, intr.ppy)
umi_tag_size = 0.040
base_tag_size = 0.050

# List of valid tag IDs for the base and umi with predefined rotation matrices
base_tags = {
    17: np.array([[0, 0, -1], [-1, 0, 0], [0, 1, 0]]),      # np.linalg.inv(np.array([[0, 1, 0], [0, 0, 1], [1, 0, 0]])),        ##### fully wrong? (z down, x inside)
    18: np.array([[0, 1, 0], [-1, 0, 0], [0, 0, 1]]),       ### CORRECT
    19: np.array([[0, 0, 1], [-1, 0, 0], [0, -1, 0]]),      # np.array([[-1, 0, 0], [0, -1, 0], [0, 0, 1]]), # np.array([[0, 1, 0], [0, 0, -1], [-1, 0, 0]]), #### Z up? x out
    20: np.array([[0, -1, 0], [-1, 0, 0], [0, 0, -1]]),     ## CORRECT #### x axis is down, should be up (180 degree about z axis)
    21: np.array([[0, 0, 1], [0, 1, 0], [-1, 0, 0]]),       ##### CORRECT #### 180 about y then test
    22: np.array([[0, 1, 0], [0, 0, 1], [1, 0, 0]])         # np.eye(3) # np.array([[0, 0, 1], [-1, 0, 0], [0, -1, 0]])    #np.array([[1, 0, 0], [0, 0, 1], [0, -1, 0]])#####?
}

umi_tags = {
    579: np.eye(3), #np.array([[0, 0, -1], [-1, 0, 0], [0, 1, 0]]),
    580: np.eye(3), #np.array([[0, 1, 0], [0, 0, -1], [-1, 0, 0]]),    ## 90 deg around y axis: np.array([[0, 1, 0], [0, 0, -1], [-1, 0, 0]]),
    581: np.eye(3),
    582: np.eye(3),
    583: np.eye(3),
    584: np.eye(3)
}

# umi_tags = {579, 580, 581, 582}
# base_tags = {17, 19, 21}
pixel_coordinates_list = []


while not rospy.is_shutdown():
    frames = pipe.wait_for_frames()
    color_frame = frames.get_color_frame()
    color_image = np.asanyarray(color_frame.get_data())
    gray_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)
    cv2.namedWindow('RealSense', cv2.WINDOW_AUTOSIZE)

    detected_tags = detector.detect(gray_image)
    umi_cube_poses = []
    ee_poses = []
    base_cube_poses = []

    for tag in detected_tags:
        tag_id = tag.tag_id
        print("Detected tag ID:", tag.tag_id)

        # position = (tag.center[0], tag.center[1], 0.0)  # Replace with 3D position if available
        # marker = create_text_marker(f"ID: {tag_id}", position, frame_id="camera", marker_id=tag_id)
        # marker_pub.publish(marker)

        if tag_id in umi_tags:
            tag_pose = detector.detection_pose(tag, camera_params, umi_tag_size, z_sign=1)[0]
            # Calculate the gripper pose using the transform file
            transform = find_transform(tag_id)

            umi_cube_pose = tag_pose
            # Calculate the cube center position based on the tag's position
            umi_cube_pose[0:3, 3] = tag_pose[:3, 3] + tag_pose[0:3, 0:3] @ np.array([0, 0, umi_tag_size/2])
            umi_cube_pose[0:3, 0:3] = tag_pose[0:3, 0:3] #@ transform[0:3, 0:3]
            umi_cube_poses.append(umi_cube_pose)

            ee_pose = tag_pose @ transform
            print(transform)
            ee_poses.append(ee_pose)

        elif tag_id in base_tags:
            tag_pose = detector.detection_pose(tag, camera_params, umi_tag_size, z_sign=1)[0]
            # base_pose = tag_pose # directly the base is the zero
            # base_poses.append(base_pose)
            base_cube_pose = tag_pose
            # Calculate the cube orientation and center position based on the tag's position
            base_cube_pose[:3, 3] = tag_pose[:3, 3] + tag_pose[0:3, 0:3] @ np.array([0, 0, base_tag_size/2])
            base_cube_pose[:3, :3] = tag_pose[0:3, 0:3] @ base_tags[tag_id]
            base_cube_poses.append(base_cube_pose)

        # Draw tag
        (ptA, ptB, ptC, ptD) = tag.corners
        ptA = (int(ptA[0]), int(ptA[1]))
        ptB = (int(ptB[0]), int(ptB[1]))
        ptC = (int(ptC[0]), int(ptC[1]))
        ptD = (int(ptD[0]), int(ptD[1]))
        cv2.line(color_image, ptA, ptB, (0, 255, 0), 1)
        cv2.line(color_image, ptB, ptC, (0, 255, 0), 1)
        cv2.line(color_image, ptC, ptD, (0, 255, 0), 1)
        cv2.line(color_image, ptD, ptA, (0, 255, 0), 1)
        (cX, cY) = (int(tag.center[0]), int(tag.center[1]))
        cv2.circle(color_image, (cX, cY), 5, (0, 0, 255), -1)
        cv2.putText(
            color_image,
            f'ID: {tag.tag_id}',
            (cX - 10, cY - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 0, 0), 2)

    if len(ee_poses) > 0 and len(base_cube_poses) > 0:
        umi_cube_poses = np.array(umi_cube_poses)
        ee_poses = np.array(ee_poses)
        base_cube_poses = np.array(base_cube_poses)

        avg_umi_cam = np.mean(umi_cube_poses, axis=0)
        avg_ee_cam = np.mean(ee_poses, axis=0)
        avg_base_cam = np.mean(base_cube_poses, axis=0)

        if avg_ee_cam.shape == (4, 4) and avg_base_cam.shape == (4, 4):
            # === Compute gripper pose relative to base ===
            base_to_cam_inv = np.linalg.inv(avg_base_cam)
            ee_relative_to_base = base_to_cam_inv @ avg_ee_cam

            # === Visualization ===
            point_3d = avg_ee_cam[:3, 3]
            point_2d = K @ point_3d
            pixel_coords = point_2d[:2] / point_2d[2]

            pixel_coordinates_list.append(pixel_coords)
            if len(pixel_coordinates_list) > 100:
                pixel_coordinates_list.pop(0)

            for pt in pixel_coordinates_list:
                cv2.circle(color_image, (int(pt[0]), int(pt[1])), 2, (0, 0, 255), -1)

            cv2.circle(color_image, (int(pixel_coords[0]), int(pixel_coords[1])), 5, (0, 255, 0), -1)

            # === Publish base frame pose (optional) ===
            base_pose_msg = create_pose_msg(avg_base_cam, "april_base")
            base_pose_pub.publish(base_pose_msg)

            # === Publish umi_cube frame pose (optional) ===
            umi_cube_pose_msg = create_pose_msg(avg_umi_cam, "umi_cube")
            umi_pose_pub.publish(umi_cube_pose_msg)

            # === Publish umi_ee frame pose (optional) ===
            umi_pose_msg = create_pose_msg(avg_ee_cam, "umi_ee")
            umi_pose_pub.publish(umi_pose_msg)

            # === Publish pose relative to base ===
            pose_relative_msg = create_pose_msg(ee_relative_to_base, "april_base")
            umi_pose_world_pub.publish(pose_relative_msg)  # You can use a new topic if preferred

            # publish_pose_euler(ee_relative_to_base)

            # === Broadcast TF: "umi_cube to camera" ===
            quat = tft.quaternion_from_matrix(avg_umi_cam)
            tf_broadcaster.sendTransform(
                (avg_umi_cam[0, 3], avg_umi_cam[1, 3], avg_umi_cam[2, 3]),
                quat,
                rospy.Time.now(),
                "umi_cube",
                "camera"
            )

            # === Broadcast TF: "umi_ee to camera" ===
            quat = tft.quaternion_from_matrix(avg_ee_cam)
            tf_broadcaster.sendTransform(
                (avg_ee_cam[0, 3], avg_ee_cam[1, 3], avg_ee_cam[2, 3]),
                quat,
                rospy.Time.now(),
                "umi_ee",
                "camera"
            )

            # === Broadcast TF: "april_base to camera" ===
            quat = tft.quaternion_from_matrix(avg_base_cam)
            tf_broadcaster.sendTransform(
                (avg_base_cam[0, 3], avg_base_cam[1, 3], avg_base_cam[2, 3]),
                quat,
                rospy.Time.now(),
                "april_base",
                "camera"
            )

            # # === Publish pose relative to base ===
            # # === Broadcast TF ===
            # quat = tft.quaternion_from_matrix(ee_relative_to_base)
            # tf_broadcaster.sendTransform(
            #     (ee_relative_to_base[0, 3], ee_relative_to_base[1, 3], ee_relative_to_base[2, 3]),
            #     quat,
            #     rospy.Time.now(),
            #     "umi_ee",
            #     "april_base"
            # )

            # === Publish camera frame pose (optional) ===
            camera_to_base_msg = create_pose_msg(camera_to_world, "camera")
            camera_pose_pub.publish(camera_to_base_msg)

            # === Broadcast TF: "world to camera" ===
            quat = tft.quaternion_from_matrix(camera_to_world)
            tf_broadcaster.sendTransform(
                (camera_to_world[0, 3], camera_to_world[1, 3], camera_to_world[2, 3]),
                quat,
                rospy.Time.now(),
                "camera",
                "world"
            )


    cv2.imshow('RealSense', color_image)
    cv2.waitKey(1)
