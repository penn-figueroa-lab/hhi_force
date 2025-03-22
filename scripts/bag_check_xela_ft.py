#!/usr/bin/env python3

import numpy as np
import pandas as pd
import bagpy
import rosbag  
from bagpy import bagreader
from pathlib import Path
import inspect
import glob
import rospy
import os
import matplotlib.pyplot as plt
from scipy.integrate import simps
from scipy.interpolate import interp1d
from tf.transformations import euler_from_quaternion, quaternion_from_euler, quaternion_matrix
import tf.transformations as tf_transform

def skew_sym(v):
    return np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])

def interpolate(data, time, time_match):
    time_match_limited = time_match[(time_match >= time.min()) & (time_match <= time.max())]
    if data.ndim == 1:
        data = interp1d(time, data)(time_match_limited)
    else:
        data = np.array([interp1d(time, data[i])(time_match_limited) for i in range(data.shape[0])])
    return time_match_limited, data

def get_robot_ee_data(data):
    robot_ee_time = np.array(data['Time']) #-data['Time'][0]
    robot_ee_pos = np.array([data['position.x'], data['position.y'], data['position.z']])
    robot_ee_rot = np.array([data['orientation.x'], data['orientation.y'], data['orientation.z'], data['orientation.w']])
    return robot_ee_time, robot_ee_pos, robot_ee_rot

def get_wrench_data(data):
    wrench_time = np.array(data['Time']) #-data['Time'][0]
    wrench_force = np.array([data['wrench.force.x'], data['wrench.force.y'], -data['wrench.force.z']])
    wrench_torque = np.array([data['wrench.torque.x'], data['wrench.torque.y'], data['wrench.torque.z']])
    return wrench_time, wrench_force, wrench_torque

def get_bus_data(data):
    bus_time = np.array(data['Time']) #-data['Time'][0]
    bus_force = np.array([data['wrench.wrench.force.x'], data['wrench.wrench.force.y'], data['wrench.wrench.force.z']])
    bus_torque = np.array([data['wrench.wrench.torque.x'], data['wrench.wrench.torque.y'], data['wrench.wrench.torque.z']])
    bus_lin_acc = np.array([data['externalImu.linear_acceleration.x'], data['externalImu.linear_acceleration.y'], data['externalImu.linear_acceleration.z']])
    bus_ang_vel = np.array([data['externalImu.angular_velocity.x'], data['externalImu.angular_velocity.y'], data['externalImu.angular_velocity.z']])
    bus_orientaion = np.array([data['externalImu.orientation.x'], data['externalImu.orientation.y'], data['externalImu.orientation.z'], data['externalImu.orientation.w']])
    bus_lin_acc_zero_hold = bus_lin_acc[:,:-1]
    bus_lin_vel = np.diff(bus_lin_acc_zero_hold, axis=1)
    return bus_time, bus_force, bus_torque, bus_lin_vel, bus_ang_vel, bus_orientaion

def get_xela_data(bag_file):
    xela_time = []
    sensor_forces = {'x': [], 'y': [], 'z': []}
    with rosbag.Bag(bag_file, 'r') as bag:
        # Iterate over messages in the /xServTopic
        for topic, msg, t in bag.read_messages(topics=['/xServTopic']):
            # time = t.to_sec()
            # xela_time.append(time)
            # Assuming msg.sensors is a list of SensorFull messages
            for sensor in msg.sensors:
                time = sensor.time
                xela_time.append(time)
                forces = sensor.forces                
                # print(len(forces))  # result: 24                
                for force in forces:
                    sensor_forces['x'].append(force.x)
                    sensor_forces['y'].append(force.y)
                    sensor_forces['z'].append(force.z)
    # Convert lists to numpy arrays
    xela_time = np.array(xela_time) #-xela_time[0]
    for key, values in sensor_forces.items():
        sensor_forces[key] = np.array(values)
    # Since all sensors are under sensor_pos 1, we'll separate them manually
    num_sensors = 24
    sensor_forces['x'] = sensor_forces['x'].reshape(-1, num_sensors).transpose()
    sensor_forces['y'] = sensor_forces['y'].reshape(-1, num_sensors).transpose()
    sensor_forces['z'] = sensor_forces['z'].reshape(-1, num_sensors).transpose()

    return xela_time, sensor_forces

# # # # Plot the data
def plot_robot_end_effector_positions(robot_ee_time, robot_ee_pos, robot_ee_data):
    """Plot Robot End Effector Positions."""
    fig3, ax = plt.subplots(nrows=3, ncols=1, num='robot_ee_pos_vs_time', sharex=True)
    ax[0].set_title('Robot End Effector Position')
    ax[2].set_xlabel('Time (s)')
    for i in range(3):
        ax[i].plot(robot_ee_time, robot_ee_pos[i], label=f'Position_{list(robot_ee_data.columns)[i+1]}')
        ax[i].legend()
        ax[i].set_ylabel(f'Position_{list(robot_ee_data.columns)[i]} (m)')
    plt.tight_layout()

def plot_force_torque(wrench_time, wrench_force, wrench_torque, wrench_data):
    """Plot FT Sensor Force and Torque Graphs."""
    fig4, ax = plt.subplots(nrows=2, ncols=1, num='force_torque_vs_time', sharex=True)
    ax[0].set_title('Force and Torque')
    ax[1].set_xlabel('Time (s)')
    ax[0].set_ylabel('Force (N)')
    ax[1].set_ylabel('Torque (N*m)')
    for j in range(3):
        ax[0].plot(wrench_time, wrench_force[j], label=f'Force {list(wrench_data.columns)[j+5]}')
        ax[1].plot(wrench_time, wrench_torque[j], label=f'Torque {list(wrench_data.columns)[j+5]}')
    ax[0].legend()
    ax[1].legend()
    plt.tight_layout()

def plot_xela_sensor_forces_summed(xela_time, xela_forces_sum, reverse_neg_z=False):
    """Plot Xela Sensor Forces Summed."""
    fig2, ax = plt.subplots(nrows=3, ncols=1, num='xela_forces_sum_vs_time', sharex=True)
    ax[0].set_title('Xela Sensor Forces Summed')
    ax[2].set_xlabel('Time (s)')
    for i in range(3):
        if reverse_neg_z:
            ax[i].plot(xela_time, xela_forces_sum[i], label=f'Force_{list(xela_forces.keys())[i]}')
        else:
            ax[i].plot(xela_time, xela_forces_sum_flipped[i], label=f'Force_{list(xela_forces.keys())[i]}')
        ax[i].set_ylabel(f'Force_{list(xela_forces.keys())[i]} (N)')
        ax[i].legend()




# test = 'Test1_S1_tactileFTBAG'
# test = 'Test1_s2_tactileFTBAG'
# test = 'Test2_s1_tactileFTBAG'
test = 'Test2_s2_tactileFTBAG'
bag_names = glob.glob(str(Path(inspect.getsourcefile(lambda:0)).parent.parent/'data'/"*.bag"))
for bag_name in bag_names:
    if test in bag_name:
        b = bagreader(bag_name)
        print(b.topic_table)

  
# bus_data = pd.read_csv(b.message_by_topic('/bus0/ft_sensor0/ft_sensor_readings/reading'))
# _, _, _, _, _, bus_orientaion = get_bus_data(bus_data)
# bus_R_matrix = np.array([quaternion_matrix([bus_orientaion[i][0], bus_orientaion[i][1], bus_orientaion[i][2], bus_orientaion[i][3]]) for i in range(len(bus_orientaion))])
# # print("Bus R Matrix: ", bus_R_matrix[0])
wrench_data = pd.read_csv(b.message_by_topic('/bus0/ft_sensor0/ft_sensor_readings/wrench'))
wrench_time, wrench_force, wrench_torque = get_wrench_data(wrench_data)

xela_time, xela_forces = get_xela_data(bag_name)
# print(xela_time.shape, xela_forces['x'].shape, xela_forces['y'].shape, xela_forces['z'].shape)
xela_forces_sum = np.array([np.sum(xela_forces['x'], axis=0), np.sum(xela_forces['y'], axis=0), np.sum(xela_forces['z'], axis=0)])
# print(xela_forces_sum.shape)

# # Some post-processing
for i in range(24):
    # Remove Bias
    xela_forces['x'][i] = [val - xela_forces['x'][i][0] for val in xela_forces['x'][i]]
    xela_forces['y'][i] = [val - xela_forces['y'][i][0] for val in xela_forces['y'][i]]
    xela_forces['z'][i] = [val - xela_forces['z'][i][0] for val in xela_forces['z'][i]]
    # Flip negative in 'z' direction:
    xela_forces_flipped = xela_forces.copy()
    num_negative = sum(1 for val in xela_forces['z'][i] if val < -4)  # Count negative values
    # if num_negative / len(xela_forces['z'][i]) > 0.8:  # Check if more than 80% are negative
    if num_negative > 10: 
        xela_forces_flipped['z'][i] = [-val for val in xela_forces['z'][i]]  # Flip the sign of all values
        print("\n"f"Flipped negative values in sensor {i}")

xela_forces_sum_flipped = np.array([np.sum(xela_forces_flipped['x'], axis=0), np.sum(xela_forces_flipped['y'], axis=0), np.sum(xela_forces_flipped['z'], axis=0)])

def plot_forces(reverse_neg_z=False, show_ft=False):
    """Plot Xela Sensor Forces."""   
    y_min, y_max, y_ranges = np.array([0, 0, 0]), np.array([0, 0, 0]), np.array([0, 0, 0])
    for i in range(3):
        y_min[i] = min([min(xela_forces[list(xela_forces.keys())[i]][j]) for j in range(xela_forces['x'].shape[0])])
        y_max[i] = max([max(xela_forces[list(xela_forces.keys())[i]][j]) for j in range(xela_forces['x'].shape[0])])
        y_ranges[i] = y_max[i] - y_min[i]    
    height_ratios = [y_range / sum(y_ranges) for y_range in y_ranges]

    fig1, ax = plt.subplots(nrows=3, ncols=1, num='xela_forces_vs_time', sharex=True, gridspec_kw={'height_ratios': height_ratios})
    ax[0].set_title('Xela Sensor Forces')
    ax[2].set_xlabel('Time (s)')
    for i in range(3):
        for j in range(xela_forces['x'].shape[0]):
            if reverse_neg_z:
                ax[i].plot(xela_time, xela_forces_flipped[list(xela_forces.keys())[i]][j], label=f'Sensor {j}')
            else:
                ax[i].plot(xela_time, xela_forces[list(xela_forces.keys())[i]][j], label=f'Sensor {j}')
        if reverse_neg_z:
            ax[i].plot(xela_time, xela_forces_sum_flipped[i], color='black', linestyle='--', linewidth=1.5, label='Summed Forces')
        else:
            ax[i].plot(xela_time, xela_forces_sum[i], color='black', linestyle='--', linewidth=1.5, label='Summed Forces') 
        if show_ft:
            ax[i].plot(wrench_time, wrench_force[i], color='black', linestyle=':', linewidth=0.5, label='FT Sensor')
        ax[i].set_ylabel(f'Force_{list(xela_forces.keys())[i]} (N)')
        y_range_min = min(y_min[i], min(xela_forces_sum[i]))
        y_range_max = max(y_max[i], max(xela_forces_sum[i]))
        # scale ax[i] based on the data range
        ax[i].set_ylim([y_range_min - 0.2 * abs(y_range_min), y_range_max + 0.2 * abs(y_range_max)])
    ax[i].set_xlim([min(xela_time[0], wrench_time[0]), max(xela_time[-1], wrench_time[-1])])
    ax[1].legend(loc='center left', bbox_to_anchor=(1, 0.5))

def plot_xela_vs_ft_sensor_forces(wrench_time, wrench_force, reverse_neg_z=False):
    """Plot Xela forces vs. FT Sensor forces."""
    fig5, ax = plt.subplots(nrows=3, ncols=1, num='xela_vs_ft_sensor_forces', sharex=True)
    ax[0].set_title('Xela vs. FT Sensor Forces')
    ax[2].set_xlabel('Time (s)')
    for i in range(3):
        ax[i].plot(wrench_time, wrench_force[i], label=f'FT Sensor {list(wrench_data.columns)[i+5]}')
        if reverse_neg_z:
            ax[i].plot(xela_time, xela_forces_sum_flipped[i], label=f'Xela Force {list(xela_forces.keys())[i]}')
        else:
            ax[i].plot(xela_time, xela_forces_sum[i], label=f'Xela Force {list(xela_forces.keys)[i]}')
        ax[i].legend()
        ax[i].set_ylabel(f'Force_{list(xela_forces.keys())[i]} (N)')
        ax[i].set_xlim([min(xela_time[0], wrench_time[0]), max(xela_time[-1], wrench_time[-1])])
    plt.tight_layout()

# Transform the FT forces to the world frame
if '/iiwa/task_states' in b.topic_table['Topics'].values:
    print("Robot EE Data Available")
    robot_ee_data = pd.read_csv(b.message_by_topic('/iiwa/task_states'))
    robot_ee_time, robot_ee_pos, robot_ee_rot = get_robot_ee_data(robot_ee_data)

    print("Robot EE Data:   Start Time: ", robot_ee_time[0], "End Time: ", robot_ee_time[-1], "Duration: ", robot_ee_time[-1] - robot_ee_time[0])
    print("Wrench Data:   Start Time: ", wrench_time[0], "End Time: ", wrench_time[-1], "Duration: ", wrench_time[-1] - wrench_time[0])
    print("Xela Data:   Start Time: ", xela_time[0], "End Time: ", xela_time[-1], "Duration: ", xela_time[-1] - xela_time[0])

    start_time = min(wrench_time[0], xela_time[0])
    print("Start Time: ", start_time)
    # robot_ee_time = robot_ee_time - start_time
    xela_time = xela_time - start_time
    wrench_time = wrench_time - start_time
    robot_ee_time = robot_ee_time - start_time
    xela_time -= xela_time[0]

    print("FT Sensor Force: ", wrench_force.shape)
    print("FT Sensor Torque: ", wrench_torque.shape)
    print("Robot EE Rot: ", robot_ee_rot.shape) #(4, 32569)
    # interpolate 
    robot_ee_time, robot_ee_rot = interpolate(robot_ee_rot, robot_ee_time, wrench_time)
    robot_ee_rot = robot_ee_rot.T
    robot_ee_rot = np.append(robot_ee_rot, robot_ee_rot[-1].reshape(1,4), axis=0)
    print("Robot EE Rot: ", robot_ee_rot.shape) #(24417, 4)

    FT_sensor_rot = tf_transform.quaternion_matrix([0, 0, 0.707, 0.707])[:3,:3]
    FT_sensor_trans = np.array([0.01, 0.01, 0.3231]) #np.array([0, 0, 0.16])
    ee_to_ft_matrix = np.vstack((np.hstack((FT_sensor_rot, np.zeros((3,3)))), \
                            np.hstack((-skew_sym(-FT_sensor_trans), FT_sensor_rot))))
    print("EE to FT Matrix: ", ee_to_ft_matrix.shape)

    force = wrench_force.T
    torque = wrench_torque.T
    wrench_wrt_robot = np.zeros((len(force), 6))
    wrench_wrt_world = np.zeros((len(force), 3))
    for t in range(len(wrench_force.T)):
        wrench_wrt_robot[t] = ee_to_ft_matrix @ np.hstack((force[t], torque[t]))
        wrench_wrt_world[t] = tf_transform.quaternion_matrix(robot_ee_rot[t])[:3,:3] @ wrench_wrt_robot[t][:3]
    wrench_wrt_world = wrench_wrt_world.T

    plot_xela_vs_ft_sensor_forces(wrench_time, wrench_wrt_world, reverse_neg_z=True)

else:
    print("Robot EE Data Not Available")
    plot_xela_vs_ft_sensor_forces(wrench_time, wrench_force, reverse_neg_z=True)


# plot_robot_end_effector_positions(robot_ee_time, robot_ee_pos, robot_ee_data)
# plot_force_torque(wrench_time, wrench_force, wrench_torque, wrench_data)
# plot_xela_sensor_forces(xela_time, xela_forces, reverse_neg_z=False)
# plot_xela_sensor_forces(xela_time, xela_forces, reverse_neg_z=True)
# plot_xela_sensor_forces_summed(xela_time, xela_forces_sum, reverse_neg_z=True)


plot_forces(reverse_neg_z=False, show_ft=False)
plt.show()
