#!/usr/bin/env python3

import pandas as pd
import bagpy
from bagpy import bagreader
from pathlib import Path
import inspect
import glob
import rospy
import os

def play_bag_files():
    # test = 'Shafagh_Proposed_4.18'
    # bag_files_path = Path(inspect.getsourcefile(lambda:0)).parent.parent / 'data' / test 
    # Get a list of .bag files in the directory
    bag_files = ['/home/yifei/kuka_ws/bags/6d_exp/2024-08-04-18-54-52_ft_human_6d.bag']
    
    # Iterate over each .bag file in the directory and play them
    for bag_file in bag_files:
        bag_file_path = bag_file
        rospy.loginfo("Playing {}".format(bag_file_path))
        os.system("rosbag play {} --clock".format(bag_file_path))

if __name__ == '__main__':
    rospy.init_node('bag_play_script')
    play_bag_files()

    # shutdown the node after playing all the bag files
    # rospy.signal_shutdown("Finished playing the bag file")

    # test = 'Tianyu_Planner_1'
    # bag_names = glob.glob(str(Path(inspect.getsourcefile(lambda:0)).parent.parent / 'data' / test / "*.bag"))
    # for bag_name in bag_names:
    #     b = bagreader(bag_name)
    #     print(b.topic_table)
 

