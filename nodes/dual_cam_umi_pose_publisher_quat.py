#!/usr/bin/env python

import cv2
import numpy as np
import pyrealsense2 as rs
import apriltag
import rospy
from geometry_msgs.msg import PoseStamped
import tf.transformations as tft
import tf
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import transforms3d as t3d
from OPF_quat2 import OPF_3d
from scipy.spatial.transform import Rotation as R, Slerp
import rospkg



### Using the original pose publisher code but with inclusion of opf filter
# Load tag-to-gripper transform from NPZ
rospack = rospkg.RosPack()
pkg_path = rospack.get_path('hhi_force')
tag_to_gripper = np.load(pkg_path + "/nodes/calibrated_transforms.npz") 

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

# def visualize_opf_particles(opf_obj):
#     fig = plt.figure()
#     ax = fig.add_subplot(111, projection='3d')

#     # Plot particles
#     particles = opf_obj.particles
#     ax.scatter(particles[:, 0], particles[:, 1], particles[:, 2], c='blue', s=1, label='Particles')

#     # Plot filtered position
#     filtered_pos = opf_obj.curr_pos
#     ax.scatter(filtered_pos[0], filtered_pos[1], filtered_pos[2], c='red', s=50, label='Filtered Pose')

#     ax.set_xlabel('X')
#     ax.set_ylabel('Y')
#     ax.set_zlabel('Z')
#     ax.set_title('OPF Particle Filter Visualization')
#     ax.legend()
#     plt.show(block=False)
#     plt.pause(0.01)
#     plt.close(fig)
# plt.ion()
# fig = plt.figure(figsize=(8,8))
# ax = fig.add_subplot(111,projection='3d')


def average_pose(pose_list):
    if not pose_list:
        return None
    positions = [pose[:3, 3] for pose in pose_list]
    avg_pos = np.mean(positions, axis=0)

    rotations = R.from_matrix([pose[:3, :3] for pose in pose_list])
    mean_rot = rotations.mean()
    avg_rot = mean_rot.as_matrix()

    T_avg = np.eye(4)
    T_avg[:3, 3] = avg_pos
    T_avg[:3, :3] = avg_rot
    return T_avg

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

# umi_to_gripper = np.array([[1, 0, 0, 0],
#                            [0, 1, 0, umi_tag_size/2+0.025],
#                            [0, 0, 1, 0.26],
#                            [0, 0, 0, 1]])


umi_to_gripper_by_tag = {
    579: np.array([  # BACK tag, +Z face → gripper is -Z from it
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 1, 0.26 + umi_tag_size],  # Z forward from +Z face
        [0, 0, 0, 1]
    ]),
    580: np.array([  # LEFT tag, -X face
        [1, 0, 0, -umi_tag_size],
        [0, 1, 0, 0],
        [0, 0, 1, 0.26],
        [0, 0, 0, 1]
    ]),
    581: np.array([  # TOP tag, +Y face
        [1, 0, 0, 0],
        [0, 1, 0, umi_tag_size],
        [0, 0, 1, 0.26],
        [0, 0, 0, 1]
    ]),
    582: np.array([  # RIGHT tag, +X face
        [1, 0, 0, umi_tag_size],
        [0, 1, 0, 0],
        [0, 0, 1, 0.26],
        [0, 0, 0, 1]
    ])
}

# umi_to_gripper_579 = np.array([
#         [1, 0, 0, 0],
#         [0, 1, 0, umi_tag_size/2 + 0.025],
#         [0, 0, 1, 0.26 + umi_tag_size],  # extra depth for back face
#         [0, 0, 0, 1]
#     ])
# umi_to_gripper = np.array([
#         [1, 0, 0, 0],
#         [0, 1, 0, umi_tag_size/2 + 0.025],
#         [0, 0, 1, 0.26],
#         [0, 0, 0, 1]
#     ])
# # blank_image = np.zeros((480, 640, 3), dtype=np.uint8)
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


def smooth_pose(prev_pose, new_pose, alpha=0.2):
    """Smooth a pose using linear interpolation for position and SLERP for rotation."""
    smoothed = np.eye(4)

    # --- Translation: linear interpolation ---
    smoothed[:3, 3] = (1 - alpha) * prev_pose[:3, 3] + alpha * new_pose[:3, 3]

    # --- Rotation: spherical interpolation ---
    times = [0, 1]
    key_rots = R.from_matrix([prev_pose[:3, :3], new_pose[:3, :3]])
    slerp = Slerp(times, key_rots)
    interpolated_rot = slerp(alpha)
    smoothed[:3, :3] = interpolated_rot.as_matrix()

    return smoothed

pixel_coordinates_list_1 = []
pixel_coordinates_list_2 = []

base_cube_pose_1 = np.eye(4)
base_cube_pose_2 = np.eye(4)

# Initialize camera-to-base transforms
last_known_base_pose_1 = np.eye(4)
last_known_base_pose_2 = np.eye(4)

obj_OPF = OPF_3d(num_particles = 5000, name="umi_gripper")
obj_OPF_base = OPF_3d(num_particles= 1000, name = "base_block")


while not rospy.is_shutdown():

    detections1, img1 = process_camera(pipe1, 1)
    detections2, img2 = process_camera(pipe2, 2)

    base_poses_1, base_poses_2 = [], []
    umi_poses_1, umi_poses_2 = [], []
    base_measurements, valid_measurements = [], []
    umi_origin_poses_1, umi_origin_poses_2 = [], []


    rospy.loginfo_throttle(1.0, f"CAM1 Detected tag IDs: {[det['id'] for det in detections1]}")
    rospy.loginfo_throttle(1.0, f"CAM2 Detected tag IDs: {[det['id'] for det in detections2]}")

    def process_base_detection(det, base_pose_storage, is_base=True):
        tag_id = det['id']
        pose = det['pose']
        rot = base_tags[tag_id]
        size = base_tag_size
        T_adj = np.eye(4)
        T_adj[:3, :3] = rot
        T_adj[:3, 3] = [0, 0, size / 2]
        T_corrected = pose @ T_adj
        quat = np.roll(t3d.quaternions.mat2quat(T_corrected[:3, :3]), -1)
        measurement = {'value': np.hstack((T_corrected[:3, 3], quat)), 'weight': 1.0}
        base_pose_storage.append(T_corrected)
        return measurement
   
    # def compute_gripper_pose(det):
    #     tag_id = det['id']
    #     pose = det['pose']
    #     try:
    #         T_tag_to_finger = tag_to_gripper[f'from_{tag_id}']  # 4x4 transform
    #     except KeyError:
    #         rospy.logwarn(f"No calibrated transform for tag {tag_id}")
    #         return None
    #     return pose @ T_tag_to_finger  
    def compute_gripper_pose(det):
        tag_id = det['id']
        pose = det['pose']

        # Tag-to-cube transform (rotation defined in umi_tags)
        rot = umi_tags[tag_id]
        T_tag_to_cube = np.eye(4)
        T_tag_to_cube[:3, :3] = rot
        T_tag_to_cube[:3, 3] = [0, 0, umi_tag_size / 2]  # move from tag center to face cube centre

        T_cube = pose @ T_tag_to_cube

        T_cube_to_finger = umi_to_gripper_by_tag.get(tag_id)
        if T_cube_to_finger is None:
            rospy.logwarn(f"No transform to gripper defined for tag {tag_id}")
            return None

        return T_cube @ T_cube_to_finger

    # # Camera 1 detections
    # for det in detections1:
    #     if det['id'] in umi_tags:
    #         Tg = compute_gripper_pose(det)
    #         if Tg is not None:
    #             umi_origin_poses_1.append(Tg)
    #     elif det['id'] in base_tags:
    #         base_measurements.append(process_base_detection(det, base_poses_1))


    #     # Camera 2 detections
    # for det in detections2:
    #     if det['id'] in umi_tags:
    #         Tg = compute_gripper_pose(det)
    #         if Tg is not None:
    #             umi_origin_poses_2.append(Tg)
    #     elif det['id'] in base_tags:
    #         base_measurements.append(process_base_detection(det, base_poses_2))
# 1. Process base detections
    for det in detections1:
        if det['id'] in base_tags:
            base_measurements.append(process_base_detection(det, base_poses_1))
    for det in detections2:
        if det['id'] in base_tags:
            base_measurements.append(process_base_detection(det, base_poses_2))

    # 2. Smooth and update base cube poses
    if base_poses_1:
        observed1 = base_poses_1[0]
        base_cube_pose_1 = smooth_pose(last_known_base_pose_1, observed1, alpha=0.2)
        last_known_base_pose_1 = base_cube_pose_1.copy()
    else:
        base_cube_pose_1 = last_known_base_pose_1.copy()

    if base_poses_2:
        observed2 = base_poses_2[0]
        base_cube_pose_2 = smooth_pose(last_known_base_pose_2, observed2, alpha=0.2)
        last_known_base_pose_2 = base_cube_pose_2.copy()
    else:
        base_cube_pose_2 = last_known_base_pose_2.copy()

    # 3. Compute camera-to-base transforms
    camera1_to_base = np.linalg.inv(base_cube_pose_1)
    camera2_to_base = np.linalg.inv(base_cube_pose_2)

    # 4. Now safely use camera-to-base to convert gripper detections
    for det in detections1:
        if det['id'] in umi_tags:
            Tg_cam = compute_gripper_pose(det)
            if Tg_cam is not None:
                Tg_base = camera1_to_base @ Tg_cam
                umi_origin_poses_1.append(Tg_base)

    for det in detections2:
        if det['id'] in umi_tags:
            Tg_cam = compute_gripper_pose(det)
            if Tg_cam is not None:
                Tg_base = camera2_to_base @ Tg_cam
                umi_origin_poses_2.append(Tg_base)

    # 5. Add measurements
    for Tg in umi_origin_poses_1 + umi_origin_poses_2:
        quat = np.roll(t3d.quaternions.mat2quat(Tg[:3, :3]), -1)
        valid_measurements.append({'value': np.hstack((Tg[:3, 3], quat)), 'weight': 1.0})
    # # Average gripper origin poses from each camera
    # T_avg_1 = average_pose(umi_origin_poses_1)
    # T_avg_2 = average_pose(umi_origin_poses_2)

    # if T_avg_1 is not None:
    #     quat1 = np.roll(t3d.quaternions.mat2quat(T_avg_1[:3, :3]), -1)
    #     valid_measurements.append({'value': np.hstack((T_avg_1[:3, 3], quat1)), 'weight': 1.0})

    # if T_avg_2 is not None:
    #     quat2 = np.roll(t3d.quaternions.mat2quat(T_avg_2[:3, :3]), -1)
    #     valid_measurements.append({'value': np.hstack((T_avg_2[:3, 3], quat2)), 'weight': 1.0})

    # if base_poses_1:
    #     observed1 = base_poses_1[0]
    #     base_cube_pose_1 = smooth_pose(last_known_base_pose_1, observed1, alpha=0.2)
    #     last_known_base_pose_1 = base_cube_pose_1.copy()
    # else:
    #     base_cube_pose_1 = last_known_base_pose_1.copy()

    # if base_poses_2:
    #     observed2 = base_poses_2[0]
    #     base_cube_pose_2 = smooth_pose(last_known_base_pose_2, observed2, alpha=0.2)
    #     last_known_base_pose_2 = base_cube_pose_2.copy()
    # else:
    #     base_cube_pose_2 = last_known_base_pose_2.copy()

    # # Compute camera-to-base transforms
    # camera1_to_base = np.linalg.inv(base_cube_pose_1)
    # camera2_to_base = np.linalg.inv(base_cube_pose_2)




    # ─── UMI OPF FILTER ───────────────────────────────
    if valid_measurements:
        obj_OPF.predict()
        obj_OPF.update_all(valid_measurements)
        obj_OPF.systematic_resample()
        obj_OPF.resample_from_index()
    else:
        obj_OPF.predict()

    # ─── EXTRACT POSES ────────────────────────────────
    Tg = np.eye(4)
    Tg[:3, 3] = obj_OPF.curr_pos
    qg = obj_OPF.curr_pos1
    quat_wxyz_g = [qg[3], qg[0], qg[1], qg[2]]
    Tg[:3, :3] = t3d.quaternions.quat2mat(quat_wxyz_g)

    # umi_cube_pose = Tg
    umi_ee_pose = Tg

    # ─── TF + POSE PUBS ───────────────────────────────
    # tf_broadcaster.sendTransform(
    #     tuple(umi_cube_pose[:3, 3]),
    #     tft.quaternion_from_matrix(umi_cube_pose),
    #     rospy.Time.now(), "umi_cube", "april_base")

    tf_broadcaster.sendTransform(
        tuple(umi_ee_pose[:3, 3]),
        tft.quaternion_from_matrix(umi_ee_pose),
        rospy.Time.now(), "umi_ee", "april_base")

    tf_broadcaster.sendTransform(
        tuple(camera1_to_base[:3, 3]),
        tft.quaternion_from_matrix(camera1_to_base),
        rospy.Time.now(), "cam1", "april_base")

    tf_broadcaster.sendTransform(
        tuple(camera2_to_base[:3, 3]),
        tft.quaternion_from_matrix(camera2_to_base),
        rospy.Time.now(), "cam2", "april_base")
    
    umi_pose_msg = create_pose_msg(umi_ee_pose, "umi_ee")
    umi_ee_pose_pub.publish(umi_pose_msg)

    camera1_to_base_msg = create_pose_msg(camera1_to_base, "cam1")
    cam1_pose_pub.publish(camera1_to_base_msg)
    
    camera2_to_base_msg = create_pose_msg(camera2_to_base, "cam2")
    cam2_pose_pub.publish(camera2_to_base_msg)


    # ─── Visual Feedback ──────────────────────────────
    cv2.imshow('Camera 1', img1)
    cv2.imshow('Camera 2', img2)
    cv2.waitKey(1)
    
# # Plotting
#     ax.clear()
#     particles = obj_OPF.particles
#     assert all(len(t) == 7 for t in obj_OPF.trajectory), "Trajectory entries must be length 7"
#     trajectory = np.array(obj_OPF.trajectory)
#     # Plot particles
#     ax.scatter(particles[:, 0], particles[:, 1], particles[:, 2], c='blue', s=1, label='Particles')
#     # Plot estimated pose
#     ax.scatter(obj_OPF.curr_pos[0], obj_OPF.curr_pos[1], obj_OPF.curr_pos[2], c='red', s=50, label='Estimated Pose')
#     # Plot trajectory
#     ax.plot(trajectory[:, 0], trajectory[:, 1], trajectory[:, 2], c='green', label='Trajectory')
#     ax.set_xlabel('X')
#     ax.set_ylabel('Y')
#     ax.set_zlabel('Z')
#     ax.set_title('OPF Particle Visualization')
#     ax.legend()
#     plt.draw()
#     plt.pause(0.01)  # pause to update the figure
#     fig.canvas.flush_events()
# Cleanup
pipe1.stop()
pipe2.stop()
