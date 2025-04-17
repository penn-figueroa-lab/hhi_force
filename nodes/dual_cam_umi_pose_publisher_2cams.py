#!/usr/bin/env python

import cv2
import numpy as np
import pyrealsense2 as rs
import apriltag
import rospy
from geometry_msgs.msg import PoseStamped, Vector3
import tf.transformations as tft
import tf
from scipy.spatial.transform import Rotation as R

np.set_printoptions(suppress=True)

# Initialize ROS node
rospy.init_node('umi_gripper_dual_cam_pose')
pub_fused = rospy.Publisher('/umi_gripper/pose_fused', PoseStamped, queue_size=10)
pub_cam1 = rospy.Publisher('/camera1/umi_gripper/pose', PoseStamped, queue_size=10)
pub_cam2 = rospy.Publisher('/camera2/umi_gripper/pose', PoseStamped, queue_size=10)
tf_broadcaster = tf.TransformBroadcaster()

# Camera configuration
CAM1_SERIAL = '012345678901'  # Replace with actual serial numbers
CAM2_SERIAL = '987654321098'
TAG_SIZE = 0.050
GRIPPER_TAGS = {579, 580, 581, 582}
BASE_TAGS = {17, 19, 21}

# Load camera-specific calibrations
def load_calibrations():
    calibs = {
        'cam1': {
            'base_transform': np.load("cam1_to_base.npy"),
            'gripper_transforms': np.load("cam1_gripper_calib.npz")
        },
        'cam2': {
            'base_transform': np.load("cam2_to_base.npy"),
            'gripper_transforms': np.load("cam2_gripper_calib.npz")
        }
    }
    return calibs

calibrations = load_calibrations()

# Initialize cameras
def start_camera(serial):
    config = rs.config()
    config.enable_device(serial)
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    pipeline = rs.pipeline()
    profile = pipeline.start(config)
    return pipeline, profile

cam1_pipe, cam1_profile = start_camera(CAM1_SERIAL)
cam2_pipe, cam2_profile = start_camera(CAM2_SERIAL)

# AprilTag detector
options = apriltag.DetectorOptions(families='tag36h11', nthreads=4)
detector = apriltag.Detector(options)

def get_camera_params(profile):
    intr = profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
    return (intr.fx, intr.fy, intr.ppx, intr.ppy)

cam1_params = get_camera_params(cam1_profile)
cam2_params = get_camera_params(cam2_profile)

def detect_tags(gray_image, camera_params):
    return detector.detect(gray_image), camera_params

def process_camera(pipeline, calib_data, cam_id):
    try:
        frames = pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        if not color_frame:
            return None, None, None
        
        gray = cv2.cvtColor(np.asanyarray(color_frame.get_data()), cv2.COLOR_BGR2GRAY)
        detected_tags, params = detect_tags(gray, calib_data['camera_params'])
        
        base_poses = []
        gripper_poses = []
        
        for tag in detected_tags:
            if tag.tag_id in GRIPPER_TAGS:
                pose = detector.detection_pose(tag, params, TAG_SIZE)[0]
                transform = calib_data['gripper_transforms'][f'from_{tag.tag_id}']
                gripper_poses.append(pose @ transform)
            elif tag.tag_id in BASE_TAGS:
                pose = detector.detection_pose(tag, params, TAG_SIZE)[0]
                base_poses.append(pose)
        
        avg_gripper = np.mean(gripper_poses, axis=0) if gripper_poses else None
        avg_base = np.mean(base_poses, axis=0) if base_poses else None
        
        return avg_gripper, avg_base, frames.get_timestamp()
    
    except Exception as e:
        rospy.logerr(f"Camera {cam_id} error: {e}")
        return None, None, None

def fuse_poses(cam1_pose, cam2_pose, base1, base2):
    valid_poses = []
    weights = []
    
    if cam1_pose is not None and base1 is not None:
        world_pose1 = np.linalg.inv(base1) @ cam1_pose
        valid_poses.append(world_pose1)
        weights.append(1.0)  # Could use reprojection error here
    
    if cam2_pose is not None and base2 is not None:
        world_pose2 = np.linalg.inv(base2) @ cam2_pose
        valid_poses.append(world_pose2)
        weights.append(1.0)
    
    if not valid_poses:
        return None
    
    return np.average(valid_poses, axis=0, weights=weights)

def create_pose_msg(matrix, frame_id):
    msg = PoseStamped()
    msg.header.stamp = rospy.Time.now()
    msg.header.frame_id = frame_id
    msg.pose.position.x = matrix[0,3]
    msg.pose.position.y = matrix[1,3]
    msg.pose.position.z = matrix[2,3]
    q = tft.quaternion_from_matrix(matrix)
    msg.pose.orientation.x = q[0]
    msg.pose.orientation.y = q[1]
    msg.pose.orientation.z = q[2]
    msg.pose.orientation.w = q[3]
    return msg

while not rospy.is_shutdown():
    # Process both cameras
    cam1_gripper, cam1_base, ts1 = process_camera(cam1_pipe, {
        'camera_params': cam1_params,
        'gripper_transforms': calibrations['cam1']['gripper_transforms']
    }, 1)
    
    cam2_gripper, cam2_base, ts2 = process_camera(cam2_pipe, {
        'camera_params': cam2_params,
        'gripper_transforms': calibrations['cam2']['gripper_transforms']
    }, 2)
    
    # Check synchronization
    if ts1 and ts2 and abs(ts1 - ts2) > 33:  # 33ms difference
        rospy.logwarn_throttle(1, "Camera frames out of sync!")
    
    # Fuse poses
    fused_pose = fuse_poses(cam1_gripper, cam2_gripper, cam1_base, cam2_base)
    
    if fused_pose is not None:
        # Publish fused pose
        msg = create_pose_msg(fused_pose, "world")
        pub_fused.publish(msg)
        
        # Broadcast TF
        tf_broadcaster.sendTransform(
            (fused_pose[0,3], fused_pose[1,3], fused_pose[2,3]),
            tft.quaternion_from_matrix(fused_pose),
            rospy.Time.now(),
            "umi_gripper",
            "world"
        )
    
    # Publish individual camera poses (for debugging)
    if cam1_gripper is not None and cam1_base is not None:
        pub_cam1.publish(create_pose_msg(np.linalg.inv(cam1_base) @ cam1_gripper, "cam1_world"))
    
    if cam2_gripper is not None and cam2_base is not None:
        pub_cam2.publish(create_pose_msg(np.linalg.inv(cam2_base) @ cam2_gripper, "cam2_world"))

# Cleanup
cam1_pipe.stop()
cam2_pipe.stop()
cv2.destroyAllWindows()
