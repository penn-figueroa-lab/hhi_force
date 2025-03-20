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
    bus_lin_acc_zero_hold = bus_lin_acc[:,:-1]
    bus_lin_vel = np.diff(bus_lin_acc_zero_hold, axis=1)
    return bus_time, bus_force, bus_torque, bus_lin_vel, bus_ang_vel

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


# test = 'Test1_S1_tactileFTBAG'
# test = 'Test1_s2_tactileFTBAG'
# test = 'Test1_S1_tactileFTBAG'
test = 'Test2_s2_tactileFTBAG'
bag_names = glob.glob(str(Path(inspect.getsourcefile(lambda:0)).parent.parent/'data'/"*.bag"))
for bag_name in bag_names:
    if test in bag_name:
        b = bagreader(bag_name)
        print(b.topic_table)

# robot_ee_data = pd.read_csv(b.message_by_topic('/iiwa/task_states'))
# robot_ee_time, robot_ee_pos, robot_ee_rot = get_robot_ee_data(robot_ee_data)

# bus_data = pd.read_csv(b.message_by_topic('/bus0/ft_sensor0/ft_sensor_readings/reading'))
# print(bus_data.columns)
wrench_data = pd.read_csv(b.message_by_topic('/bus0/ft_sensor0/ft_sensor_readings/wrench'))
wrench_time, wrench_force, wrench_torque = get_wrench_data(wrench_data)

xela_time, xela_forces = get_xela_data(bag_name)
# print(xela_time.shape, xela_forces['x'].shape, xela_forces['y'].shape, xela_forces['z'].shape)

# Summation of all x, y, z forces
xela_forces_sum = np.array([np.sum(xela_forces['x'], axis=0), np.sum(xela_forces['y'], axis=0), np.sum(xela_forces['z'], axis=0)])
# print(xela_forces_sum.shape)

# Print columns:
# print("Robot_ee: ", robot_ee_data.columns)
print("Wrench: ", wrench_data.columns)
print("Xela: ", xela_forces.keys())

# print("Robot EE Data:   Start Time: ", robot_ee_time[0], "End Time: ", robot_ee_time[-1], "Duration: ", robot_ee_time[-1] - robot_ee_time[0])
print("Wrench Data:   Start Time: ", wrench_time[0], "End Time: ", wrench_time[-1], "Duration: ", wrench_time[-1] - wrench_time[0])
print("Xela Data:   Start Time: ", xela_time[0], "End Time: ", xela_time[-1], "Duration: ", xela_time[-1] - xela_time[0])
start_time = min(wrench_time[0], xela_time[0])
print("Start Time: ", start_time)
# robot_ee_time = robot_ee_time - start_time
xela_time = xela_time - start_time
wrench_time = wrench_time - start_time

# # Some post-processing
for i in range(24):
    # # Remove Bias
    # xela_forces['x'][i] = [val - xela_forces['x'][i][0] for val in xela_forces['x'][i]]
    # xela_forces['y'][i] = [val - xela_forces['y'][i][0] for val in xela_forces['y'][i]]
    # xela_forces['z'][i] = [val - xela_forces['z'][i][0] for val in xela_forces['z'][i]]

    # Flip negative in 'z' direction:
    xela_forces_flipped = xela_forces.copy()
    num_negative = sum(1 for val in xela_forces['z'][i] if val < -4)  # Count negative values
    # if num_negative / len(xela_forces['z'][i]) > 0.8:  # Check if more than 80% are negative
    if num_negative > 10: 
        xela_forces_flipped['z'][i] = [-val for val in xela_forces['z'][i]]  # Flip the sign of all values
        print("\n"f"Flipped negative values in sensor {i+1}")

xela_forces_sum_2 = np.array([np.sum(xela_forces_flipped['x'], axis=0), np.sum(xela_forces_flipped['y'], axis=0), np.sum(xela_forces_flipped['z'], axis=0)])


# Interpolate the data
# wrench_time_matched, wrench_force = interpolate(wrench_force, wrench_time, xela_time)
# wrench_time, wrench_torque = interpolate(wrench_torque, wrench_time, xela_time)
# robot_ee_time, robot_ee_pos = interpolate(robot_ee_pos, robot_ee_time, xela_time)

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

def plot_xela_sensor_forces(xela_time, xela_forces, reverse_neg_z=False):
    """Plot Xela Sensor Forces."""  
        
    fig1, ax = plt.subplots(nrows=3, ncols=1, num='xela_forces_vs_time', sharex=True)
    ax[0].set_title('Xela Sensor Forces')
    ax[2].set_xlabel('Time (s)')
    for i in range(3):
        for j in range(xela_forces['x'].shape[0]):
            if reverse_neg_z:
                ax[i].plot(xela_time, xela_forces_flipped[list(xela_forces.keys())[i]][j], label=f'Sensor {j}')
            else:
                ax[i].plot(xela_time, xela_forces[list(xela_forces.keys())[i]][j], label=f'Sensor {j}')
        ax[i].set_ylabel(f'Force_{list(xela_forces.keys())[i]} (N)')
    plt.tight_layout()

def plot_xela_sensor_forces_summed(xela_time, xela_forces_sum, neglect_negative_z=False):
    """Plot Xela Sensor Forces Summed."""
    fig2, ax = plt.subplots(nrows=3, ncols=1, num='xela_forces_sum_vs_time', sharex=True)
    ax[0].set_title('Xela Sensor Forces Summed')
    ax[2].set_xlabel('Time (s)')
    for i in range(3):
        if neglect_negative_z:
            ax[i].plot(xela_time, xela_forces_sum[i], label=f'Force_{list(xela_forces.keys())[i]}')
        else:
            ax[i].plot(xela_time, xela_forces_sum_2[i], label=f'Force_{list(xela_forces.keys())[i]}')
        ax[i].set_ylabel(f'Force_{list(xela_forces.keys())[i]} (N)')
        ax[i].legend()
    plt.tight_layout()

def plot_xela_vs_ft_sensor_forces(xela_time, xela_forces_sum, wrench_time, wrench_force, neglect_negative_z=False):
    """Plot Xela forces vs. FT Sensor forces."""
    fig5, ax = plt.subplots(nrows=3, ncols=1, num='xela_vs_ft_sensor_forces', sharex=True)
    ax[0].set_title('Xela vs. FT Sensor Forces')
    ax[2].set_xlabel('Time (s)')
    for i in range(3):
        ax[i].plot(wrench_time, wrench_force[i], label=f'FT Sensor {list(wrench_data.columns)[i+5]}')
        if neglect_negative_z:
            ax[i].plot(xela_time, xela_forces_sum[i], label=f'Xela Force {list(xela_forces.keys())[i]}')
        else:
            ax[i].plot(xela_time, xela_forces_sum_2[i], label=f'Xela Force {list(xela_forces.keys)[i]}')
        ax[i].legend()
        ax[i].set_ylabel(f'Force_{list(xela_forces.keys())[i]} (N)')
        ax[i].set_xlim([xela_time[0], xela_time[-1]])
        # ax[i].set_ylim([-100,100])
    plt.tight_layout()


# plot_robot_end_effector_positions(robot_ee_time, robot_ee_pos, robot_ee_data)
# plot_force_torque(wrench_time, wrench_force, wrench_torque, wrench_data)
# plot_xela_sensor_forces(xela_time, xela_forces, reverse_neg_z=False)
plot_xela_sensor_forces(xela_time, xela_forces, reverse_neg_z=True)
# plot_xela_sensor_forces_summed(xela_time, xela_forces_sum, neglect_negative_z=True)
plot_xela_vs_ft_sensor_forces(xela_time, xela_forces_sum, wrench_time, wrench_force, neglect_negative_z=True)
plt.show()
