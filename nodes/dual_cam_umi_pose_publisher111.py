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
from scipy.spatial.transform import Rotation


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

def calc_avg_pose(poses):
    avg_translation = np.mean([p[:3, 3] for p in poses], axis=0)
    # Extract rotation matrices and create a single Rotation object
    rotation_mats = [p[:3, :3] for p in poses]
    rotation_obj = Rotation.from_matrix(rotation_mats)
    # Compute mean rotation using chordal L2 minimization
    avg_rotation = rotation_obj.mean()
    avg_pose = np.eye(4)
    avg_pose[:3, 3] = avg_translation
    avg_pose[:3, :3] = avg_rotation.as_matrix() #avg_rotation
    return avg_pose

umi_tag_size = 0.050
base_tag_size = 0.040

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
    579: np.eye(3), 
    580: np.array([[0, 0, -1], [0, 1, 0], [1, 0, 0]]),   ## 90 deg around y axis
    581: np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]]),   #np.array([[0, -1, 0], [0, 0, -1], [1, 0, 0]]), 
    582: np.array([[0, 0, 1], [0, 1, 0], [-1, 0, 0]]),   ## -90 deg around y axis
    583: np.eye(3),
    584: np.eye(3)
}

# Transform from base to world frame (can be replaced with OptiTrack)
base_to_world = np.array([ [1, 0, 0, 1],
                           [0, 1, 0, 1],
                           [0, 0, 1, 1],
                           [0, 0, 0, 1]])

umi_to_gripper = np.array([[1, 0, 0, 0],
                           [0, 1, 0, umi_tag_size/2+0.025],
                           [0, 0, 1, 0.26],
                           [0, 0, 0, 1]])

# blank_image = np.zeros((480, 640, 3), dtype=np.uint8)
# cv2.imshow('Test Window', blank_image)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# # Get cameras' Serial Number
context = rs.context()
print("Connected devices:")
for device in context.devices:
    print(f"Device Name: {device.get_info(rs.camera_info.name)}")
    print(f"Serial Number: {device.get_info(rs.camera_info.serial_number)}")
# Device Name: Intel RealSense D435  ######## Serial Number: 215322079295
# Device Name: Intel RealSense D435I ######## Serial Number: 146322071247, 146322071961

##############################################################################
##############################################################################
##############################################################################

# Initialize ROS node
rospy.init_node('dual_cam_umi_gripper_pose_publisher')

# Publishers
base_pose_pub = rospy.Publisher('/base_pose_fused', PoseStamped, queue_size=10)
umi_cube_pose_pub = rospy.Publisher('/umi_pose_fused', PoseStamped, queue_size=10)
umi_ee_pose_pub = rospy.Publisher('/umi_ee_pose_fused', PoseStamped, queue_size=10)
umi_pose_world_pub = rospy.Publisher('/umi_pose_base', PoseStamped, queue_size=10)
cam1_pose_pub = rospy.Publisher('/camera1_pose', PoseStamped, queue_size=10)
cam2_pose_pub = rospy.Publisher('/camera2_pose', PoseStamped, queue_size=10)

tf_broadcaster = tf.TransformBroadcaster()

# Camera initialization
pipe1 = rs.pipeline()
config1 = rs.config()
config1.enable_device('215322079295')  
config1.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 60)

pipe2 = rs.pipeline()
config2 = rs.config()
config2.enable_device('146322071961')  
config2.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 60)

# Start pipelines
profile1 = pipe1.start(config1)
profile2 = pipe2.start(config2)

# Camera-specific parameters
color_profile1 = profile1.get_stream(rs.stream.color)
intr1 = color_profile1.as_video_stream_profile().get_intrinsics()
K1 = np.array([[intr1.fx, 0, intr1.ppx],
              [0, intr1.fy, intr1.ppy],
              [0, 0, 1]])

color_profile2 = profile2.get_stream(rs.stream.color)
intr2 = color_profile2.as_video_stream_profile().get_intrinsics()
K2 = np.array([[intr2.fx, 0, intr2.ppx],
              [0, intr2.fy, intr2.ppy],
              [0, 0, 1]])

# AprilTag configuration
options = apriltag.DetectorOptions(families='tag36h11', nthreads=4)
detector = apriltag.Detector(options)

def process_camera(pipe, camera_index):
    frames = pipe.wait_for_frames()
    color_frame = frames.get_color_frame()
    color_image = np.asanyarray(color_frame.get_data())
    gray_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)
    cv2.namedWindow(f'Camera {camera_index}', cv2.WINDOW_AUTOSIZE)
    if not color_frame:
        print(f"Camera {camera_index}: No frame captured.")
        return None, None
    
    # Detect AprilTags
    detections = []
    for tag in detector.detect(gray_image):
        pose = detector.detection_pose(tag, (intr1.fx, intr1.fy, intr1.ppx, intr1.ppy) if camera_index == 1 
                                       else (intr2.fx, intr2.fy, intr2.ppx, intr2.ppy), umi_tag_size, z_sign=1)[0]
        detections.append({
            'id': tag.tag_id,
            'pose': pose,
            'corners': tag.corners })
        
        # Draw detected tags
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
        cv2.putText(color_image, f'ID: {tag.tag_id}', (cX - 10, cY - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
        # for corner in det['corners']:
        #     cv2.circle(img1, tuple(map(int, corner)), 5, (0, 255, 0), -1)
        #     cv2.line(img1, tuple(map(int, corner)), tuple(map(int, det['pose'][:2, 3])), (0, 255, 0), 2)
        # cv2.putText(img1, str(det['id']), tuple(map(int, corner)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    return detections, color_image


pixel_coordinates_list_1 = []
pixel_coordinates_list_2 = []

base_cube_pose_1 = np.eye(4)
base_cube_pose_2 = np.eye(4)

while not rospy.is_shutdown():

    detections1, img1 = process_camera(pipe1, 1)    
    detections2, img2 = process_camera(pipe2, 2)    

    umi_poses_1 = []
    base_poses_1 = []
    umi_poses_2 = []
    base_poses_2 = []
    
    # for det in detections1 + detections2:
    #     rospy.loginfo_throttle(1.0, f"Detected tag ID: {det['id']}")
    rospy.loginfo_throttle(1.0, f"Detected tag IDs: {[det['id'] for det in detections1 + detections2]}")

    # Calculate average pose for umi and base tags wrt camera_1
    for det in detections1:
        if det['id'] in umi_tags:  
            tag_pose = det['pose']
            cube_pose = tag_pose
            cube_pose[0:3, 3] = tag_pose[:3, 3] + tag_pose[0:3, 0:3] @ np.array([0, 0, umi_tag_size/2])
            cube_pose[0:3, 0:3] = tag_pose[0:3, 0:3] @ umi_tags[det['id']]
            umi_poses_1.append(det['pose'])
        elif det['id'] in base_tags:
            tag_pose = det['pose']
            cube_pose = tag_pose
            cube_pose[0:3, 3] = tag_pose[:3, 3] + tag_pose[0:3, 0:3] @ np.array([0, 0, base_tag_size/2])
            cube_pose[0:3, 0:3] = tag_pose[0:3, 0:3] @ base_tags[det['id']]
            base_poses_1.append(det['pose'])

        # # umi_cube_pose_1 = calc_avg_pose(umi_poses_1)
        # # base_cube_pose_1 = calc_avg_pose(base_poses_1)

        # # if len(umi_cube_pose_1) > 0 and len(base_cube_pose_1) > 0:
        # if len(umi_poses_1) > 0 and len(base_poses_1) > 0:
        #     umi_cube_pose_1 = calc_avg_pose(umi_poses_1)
        #     base_cube_pose_1 = calc_avg_pose(base_poses_1)
        #     umi_cube_wrt_base_1 = np.linalg.inv(base_cube_pose_1) @ umi_cube_pose_1
        #     # umi_ee_wrt_base_1 = umi_cube_wrt_base_1 @ umi_to_gripper
            
    # Calculate average pose for umi and base tags wrt camera_2
    for det in detections2:
        if det['id'] in umi_tags:
            tag_pose = det['pose']
            cube_pose = tag_pose
            cube_pose[0:3, 3] = tag_pose[:3, 3] + tag_pose[0:3, 0:3] @ np.array([0, 0, umi_tag_size/2])
            cube_pose[0:3, 0:3] = tag_pose[0:3, 0:3] @ umi_tags[det['id']]
            umi_poses_2.append(det['pose'])
        elif det['id'] in base_tags:
            tag_pose = det['pose']
            cube_pose = tag_pose
            cube_pose[0:3, 3] = tag_pose[:3, 3] + tag_pose[0:3, 0:3] @ np.array([0, 0, base_tag_size/2])
            cube_pose[0:3, 0:3] = tag_pose[0:3, 0:3] @ base_tags[det['id']]
            base_poses_2.append(det['pose'])
        
        # # umi_cube_pose_2 = calc_avg_pose(umi_poses_2)
        # # base_cube_pose_2 = calc_avg_pose(base_poses_2)

        # # umi_avg_translation_2 = np.mean([p[:3, 3] for p in umi_poses_2], axis=0)
        # # umi_avg_rotation_2 = tft.quaternion_average([tft.quaternion_from_matrix(p) for p in umi_poses_2])
        # # umi_cube_pose_2 = np.eye(4)
        # # umi_cube_pose_2[:3, 3] = umi_avg_translation_2
        # # umi_cube_pose_2[:3, :3] = tft.quaternion_matrix(umi_avg_rotation_2)[:3, :3]
        
        # # base_avg_translation_2 = np.mean([p[:3, 3] for p in base_poses_2], axis=0)
        # # base_avg_rotation_2 = tft.quaternion_average([tft.quaternion_from_matrix(p) for p in base_poses_2])
        # # base_cube_pose_2 = np.eye(4)
        # # base_cube_pose_2[:3, 3] = base_avg_translation_2
        # # base_cube_pose_2[:3, :3] = tft.quaternion_matrix(base_avg_rotation_2)[:3, :3]
        
        # # if len(umi_cube_pose_2) > 0 and len(base_cube_pose_2) > 0:
        # if len(umi_poses_2) > 0 and len(base_poses_2) > 0:
        #     umi_cube_pose_2 = calc_avg_pose(umi_poses_2)
        #     base_cube_pose_2 = calc_avg_pose(base_poses_2)
        #     umi_cube_wrt_base_2 = np.linalg.inv(base_cube_pose_2) @ umi_cube_pose_2
        #     # umi_ee_wrt_base_2 = umi_cube_wrt_base_2 @ umi_to_gripper

    
    # if len(base_poses_1) == 0 or len(base_poses_2) == 0:
    #     rospy.logerr_throttle(3.0, "BASE not found by cameras")
    #     # rospy.logwarn_throttle(5.0, "BASE not found by camera 1. Using last updated pose values.")
    #     continue
    # else:
    #     base_cube_pose_1 = calc_avg_pose(base_poses_1)
    #     base_cube_pose_2 = calc_avg_pose(base_poses_2)
    #     camera1_to_base = np.linalg.inv(base_cube_pose_1) 
    #     camera2_to_base = np.linalg.inv(base_cube_pose_2)

    if len(base_poses_1) > 0:
        base_cube_pose_1 = calc_avg_pose(base_poses_1)
    if len(base_poses_2) > 0:
        base_cube_pose_2 = calc_avg_pose(base_poses_2)

    if len(umi_poses_1) > 0:
        umi_cube_pose_1 = calc_avg_pose(umi_poses_1)
        umi_cube_wrt_base_1 = np.linalg.inv(base_cube_pose_1) @ umi_cube_pose_1
    if len(umi_poses_2) > 0:
        umi_cube_pose_2 = calc_avg_pose(umi_poses_2)
        umi_cube_wrt_base_2 = np.linalg.inv(base_cube_pose_2) @ umi_cube_pose_2

    camera1_to_base = np.linalg.inv(base_cube_pose_1) 
    camera2_to_base = np.linalg.inv(base_cube_pose_2)

    
    # Fusion of poses from both cameras
    if len(umi_poses_1) == 0 and len(umi_poses_2) == 0:
        rospy.logwarn_throttle(3.0, "No valid UMI pose found. Waiting...")
        # continue
    elif len(umi_poses_1) == 0 and len(umi_poses_2) > 0:
        umi_cube_pose = umi_cube_wrt_base_2
    elif len(umi_poses_1) > 0 and len(umi_poses_2) == 0:
        umi_cube_pose = umi_cube_wrt_base_1
    else:
        umi_cube_pose = calc_avg_pose([umi_cube_wrt_base_1, umi_cube_wrt_base_2])

    if umi_cube_pose is not None:
        umi_ee_pose = umi_cube_pose @ umi_to_gripper

    # # Publish the pose of the base cube
    # base_pose_msg = create_pose_msg(base_cube_pose_1, "april_base")
    # base_pose_pub.publish(base_pose_msg)
    # Publish the pose of the umi cube
    umi_pose_msg = create_pose_msg(umi_cube_pose, "umi_cube")
    umi_cube_pose_pub.publish(umi_pose_msg)
    # Publish the pose of the umi ee
    umi_pose_msg = create_pose_msg(umi_ee_pose, "umi_ee")
    umi_ee_pose_pub.publish(umi_pose_msg)
    # Publish the pose of the cameras
    cam1_pose_msg = create_pose_msg(camera1_to_base, "cam1")
    cam1_pose_pub.publish(cam1_pose_msg)
    cam2_pose_msg = create_pose_msg(camera2_to_base, "cam2")
    cam2_pose_pub.publish(cam2_pose_msg)

    # Publish the TF transforms
    tf_broadcaster.sendTransform(
        (umi_cube_pose[0, 3], umi_cube_pose[1, 3], umi_cube_pose[2, 3]),    #position
        tft.quaternion_from_matrix(umi_cube_pose),                          #quaternion
        rospy.Time.now(), "umi_cube", "april_base")
    tf_broadcaster.sendTransform(
        (camera1_to_base[0, 3], camera1_to_base[1, 3], camera1_to_base[2, 3]), 
        tft.quaternion_from_matrix(camera1_to_base),                          
        rospy.Time.now(), "cam1", "april_base")
    tf_broadcaster.sendTransform(   
        (camera2_to_base[0, 3], camera2_to_base[1, 3], camera2_to_base[2, 3]),    
        tft.quaternion_from_matrix(camera2_to_base),                          
        rospy.Time.now(), "cam2", "april_base")
    tf_broadcaster.sendTransform(
        (umi_ee_pose[0, 3], umi_ee_pose[1, 3], umi_ee_pose[2, 3]), 
        tft.quaternion_from_matrix(umi_ee_pose),                          
        rospy.Time.now(), "umi_ee", "april_base")


    
    # Display camera feeds
    cv2.imshow('Camera 1', img1)
    cv2.imshow('Camera 2', img2)
    cv2.waitKey(1)

    # # Visualization
    # point_3d = umi_ee_pose[:3, 3]
    # point_2d_1 = K1 @ point_3d
    # pixel_coords_1 = point_2d_1[:2] / point_2d_1[2]
    # pixel_coordinates_list_1.append(pixel_coords_1)
    # if len(pixel_coordinates_list_1) > 100:
    #     pixel_coordinates_list_1.pop(0)
    # for pt in pixel_coordinates_list_1:
    #     cv2.circle(img1, tuple(map(int, pt)), 5, (255, 0, 0), -1)
    # cv2.circle(img1, tuple(map(int, pixel_coords_1)), 5, (255, 0, 0), -1)

    # point_2d_2 = K2 @ point_3d
    # pixel_coords_2 = point_2d_2[:2] / point_2d_2[2]
    # pixel_coordinates_list_2.append(pixel_coords_2)
    # if len(pixel_coordinates_list_2) > 100:
    #     pixel_coordinates_list_2.pop(0)
    # for pt in pixel_coordinates_list_2:
    #     cv2.circle(img2, tuple(map(int, pt)), 5, (255, 0, 0), -1)
    # cv2.circle(img2, tuple(map(int, pixel_coords_2)), 5, (255, 0, 0), -1)


# Cleanup
pipe1.stop()
pipe2.stop()
