#!/usr/bin/env python3
import os
import pickle
import time
import numpy as np
import utils as Utils
import cv2

import rospy
from sensor_msgs.msg import Image
from geometry_msgs.msg import Twist
from nav_msgs.msg import Path , Odometry
from grid_map_msgs.msg import GridMap

from Grid import MapProcessor
from vertiencoder.model.dt_models import get_model
from vertiencoder.utils.helpers import get_conf

import torch
import torch.nn as nn
import torchvision.transforms.functional as F

MAX_VEL = 1.0
MIN_VEL = -1.0


class Deployment_Tverti():
    def __init__(self):
        #Robot Limits
        self.max_vel = MAX_VEL
        self.min_vel = MIN_VEL
        self.robot_length = 0.54

        # Load the statistics
        with open('stats.pkl', 'rb') as f:
            self.stats = pickle.load(f)
        
        self.mp = MapProcessor()
        cfg = get_conf("/home/aniket/Token_prediction/Deployment/Tverti-wheeler/vertiencoder/conf/deployment.yaml")
        self.model = get_model(cfg) # Load the model
        self.model.cuda()
        print("Model loaded")
        print(self.model)


        self.robot_pose = None
        self.patch_stack = torch.zeros(20, 1, 40, 40).cuda()
        self.action_stack = torch.zeros(20, 2).cuda()

        rospy.Subscriber('/elevation_mapping/elevation_map_raw', GridMap, self.gridMap_callback, queue_size=1)
        rospy.Subscriber("/dlio/odom_node/odom", Odometry, self.odom_cb, queue_size=1)
        self.cmd_vel_pub = rospy.Publisher('/cmd_vel', Twist, queue_size=1)    # Publish to the cmd_vel topic

    def gridMap_callback(self, gridmap_msg):
        self.mp.update_map(gridmap_msg)
            
    def odom_cb(self, odom_msg):
        x, y, z = odom_msg.pose.pose.position.x, odom_msg.pose.pose.position.y, odom_msg.pose.pose.position.z
        roll, pitch, yaw = Utils.quaternion_to_angle(odom_msg.pose.pose.orientation)
        self.robot_pose = [x, y, z, roll, pitch, yaw]
    
    def get_action(self):

        current_patch = torch.tensor(self.mp.get_elev_footprint(self.robot_pose, (40,40))).cuda().unsqueeze(0)
        current_patch = current_patch - self.robot_pose[2]
        current_patch = torch.clip(current_patch, -0.5, 0.5)
        current_patch = (current_patch - self.stats['footprint_mean']) / self.stats['footprint_std']
        # print(f"min:{current_patch.min()}, max:{current_patch.max()}, mean:{current_patch.mean()}, robot:{self.robot_pose[2]}")
        cv2.imshow("current_patch", cv2.normalize(cv2.resize(current_patch.cpu().numpy()[0], (500, 500)),None, norm_type=cv2.NORM_MINMAX))
        cv2.waitKey(1)

        # self.patch_stack[:-1, :] = self.patch_stack[1:, :].clone()
        self.patch_stack = self.patch_stack.roll(-1, 0)
        self.patch_stack[-1] = current_patch

        action = self.model(self.patch_stack.unsqueeze(0), self.action_stack.unsqueeze(0))
        action = action.squeeze(0)
        # self.action_stack[:-1] = self.action_stack[1:].clone()
        self.action_stack = self.action_stack.roll(-1, 0)
        self.action_stack[-1] = action
        # print(f"action:{self.action_stack}")
        action = action.cpu().detach().squeeze()
        action = action * self.stats['cmd_vel_std'] + self.stats['cmd_vel_mean']
        return torch.clip(action, self.min_vel, self.max_vel).numpy()

if __name__ == '__main__':
    rospy.init_node("bc_TVERTI", anonymous=True)       # Initialize the node
    deploy = Deployment_Tverti()
    rate = rospy.Rate(10)     
    while not rospy.is_shutdown():
        cur_time = time.time()
        if deploy.robot_pose is None:
            print("waiting for robot pose")
        elif deploy.mp.map_elevation is None:
            print("waiting for map data")
        else:
            action = deploy.get_action()
            print(f"action:{action},\tTime: {int((time.time() - cur_time)*1000)}")

        rate.sleep()


        # rock      0.18,   0.15
        # ground    0.0,    0.003