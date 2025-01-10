#!/usr/bin/env python3
import os
import sys
import pickle
import time
import numpy as np
import utils as Utils
import cv2
from typing import List, Optional, Tuple

import rospy
from sensor_msgs.msg import Image
from geometry_msgs.msg import Twist
from nav_msgs.msg import Path , Odometry
from grid_map_msgs.msg import GridMap
from std_msgs.msg import Float32MultiArray

from Grid import MapProcessor
from vertiencoder.model.dt_models import get_model
from vertiencoder.utils.helpers import get_conf

import torch
import torch.nn as nn
import torchvision.transforms.functional as F
torch.set_float32_matmul_precision('high')


MAX_VEL = 1.0
MIN_VEL = -1.0
publish_itrs = 80
ackermann = False

class Deployment_Tverti():
    def __init__(self):
        #Robot Limits
        self.max_vel = MAX_VEL
        self.min_vel = MIN_VEL
        self.robot_length = 0.54

        self.stats = None
        # Load the statistics
        with open('/home/aniket/Token_prediction/Deployment/Tverti-wheeler/vertiencoder/checkpoint/last_hopes/stats_tal.pkl', 'rb') as f:
            self.stats = pickle.load(f)

        # print(f"Stats loaded {self.stats.keys()}")
        self.mp = MapProcessor()

        cfg = get_conf("/home/aniket/Token_prediction/Deployment/Tverti-wheeler/vertiencoder/conf/deployment.yaml")
        self.model = get_model(cfg) # Load the model
        self.model.cuda()
        self.model.eval()
        self.model.requires_grad_(False)
        t_Params = sum([p.numel() for p in self.model.parameters() if p.requires_grad == True])
        print("Model loaded")
        print(self.model)
        print(f"Train Model: {self.model.training}, Trainable Params: {t_Params}")
        self.model(torch.zeros((1,20,1,40,40), dtype=torch.float32).cuda(), torch.zeros((1,20,2), dtype=torch.float32).cuda(), torch.zeros((1,20,6), dtype=torch.float32).cuda())
        print("Model initialized")
        #General
        self.robot_pose: Optional[List] = None 
        self.prev_robot_pose_a: Optional[List] = None
        self.prev_robot_pose_m: Optional[List] = None

        #loop Specific
        self.gt_path: List = []
        self.pred_path_a: List = []
        self.pred_path_m: List = []
        self.no_iter: int = 0 
        self.init: int = 0

        self.gt_cmds: List = []
        
        #Model SSL Specific
        self.pre_del_pose_stack: torch.Tensor = torch.zeros((20, 6), dtype=torch.float32).cuda()
        self.patch_stack: torch.Tensor = torch.zeros((20, 1, 40, 40), dtype=torch.float32).cuda()
        self.action_stack: torch.Tensor = torch.zeros((20, 2) , dtype=torch.float32).cuda()

        #Subscribers and Publishers
        rospy.Subscriber('/elevation_mapping/elevation_map_raw', GridMap, self.gridMap_callback, queue_size=5, buff_size=6*sys.getsizeof(np.zeros((360,360))))
        rospy.Subscriber("/dlio/odom_node/odom", Odometry, self.odom_cb, queue_size = 100, buff_size=100*sys.getsizeof(Odometry()))
        # rospy.Subscriber("/natnet_ros/crawler/odom", Odometry, self.odom_cb, queue_size=1000, buff_size=100*sys.getsizeof(Odometry()) )
        rospy.Subscriber("/cmd_vel1", Float32MultiArray, self.cmd_vel_cb, queue_size = 100, buff_size=100*8*sys.getsizeof(float()))

        self.path_pub_pred_a = rospy.Publisher("/paths_pred_ack", Path, queue_size = 100 )
        self.path_pub_gt = rospy.Publisher("/paths_gt", Path, queue_size = 100)
        self.path_pub_pred_m = rospy.Publisher("/paths_pred_ssl", Path, queue_size = 100)
        # self.cmd_vel_pub = rospy.Publisher('/cmd_vel1', Float32MultiArray, queue_size=1)    # Publish to the cmd_vel topic

    def gridMap_callback(self, gridmap_msg):
        self.mp.update_map(gridmap_msg)
            
    def odom_cb(self, odom_msg):
        self.robot_pose = Utils.odometry_to_particle(odom_msg)
    
    def cmd_vel_cb(self, cmd_vel_msg):
        self.gt_cmds.append([cmd_vel_msg.data[1], cmd_vel_msg.data[0]])

    def ackermann_predict(self):
        if self.prev_robot_pose_a is None:
            self.prev_robot_pose_a = torch.tensor(self.robot_pose)
            self.prev_robot_pose_a[[3, 4]] = 0.0 
            return None
        
        throttle, steering = torch.tensor(self.gt_cmds).mean(dim=0)
        
        throttle *= 1.7 #throttle -1:1 maps to -1.7m/s:1.7m/s
        steering *= 0.55 #steering -1:1 maps to -0.55rad:0.55rad
        
        pose_diff = Utils.ackermann(throttle, -steering)
        cur_pose_pred = Utils.to_world_torch(self.prev_robot_pose_a, pose_diff).squeeze()
        # print(f"pose_diff_ack xyt: {pose_diff[[0,1,5]]}, cur_pose_pred_ack: {cur_pose_pred[[1,2,5]]}")
        self.prev_robot_pose_a = cur_pose_pred

        return cur_pose_pred.tolist()

    def model_predict(self):
        if self.prev_robot_pose_m is None:
            self.prev_robot_pose_m = self.robot_pose
            return None       
        
        current_patch = torch.tensor(self.mp.get_elev_footprint(self.robot_pose, (40,40))).cuda().unsqueeze(0)
        current_patch = torch.clip(current_patch - self.robot_pose[2], -0.5, 0.5)
        current_patch = current_patch / self.stats['footprint_max']
        
        self.patch_stack = self.patch_stack.roll(-1, 0)
        self.patch_stack[-1] = current_patch
        
        cmd = torch.tensor(self.gt_cmds).mean(dim=0) #/self.stats['cmd_vel_max']
        self.action_stack = self.action_stack.roll(-1, 0)
        self.action_stack[-1] = cmd.cuda()
        
        pose_diff = self.model(self.patch_stack.unsqueeze(0), self.action_stack.unsqueeze(0), self.pre_del_pose_stack.unsqueeze(0))
        
        self.pre_del_pose_stack = self.pre_del_pose_stack.roll(-1, 0)
        self.pre_del_pose_stack[-1] = pose_diff.clone()
        
        pose_diff = (pose_diff.squeeze(0).cpu() * self.stats['pose_diff_max'])
        
        cur_pose_pred = Utils.to_world_torch(self.prev_robot_pose_m, pose_diff).squeeze()
        self.prev_robot_pose_m = cur_pose_pred

        return cur_pose_pred.tolist() 

    def initialize_stack(self):
        if self.prev_robot_pose_m is None:
                self.prev_robot_pose_m = self.robot_pose
                return
            
        #update Current Patch
        current_patch = torch.tensor(self.mp.get_elev_footprint(self.robot_pose, (40,40)), dtype=torch.float32).cuda().unsqueeze(0)
        current_patch = torch.clip((current_patch - self.robot_pose[2]), -0.5, 0.5)
        current_patch = current_patch / self.stats['footprint_max']
        self.patch_stack = self.patch_stack.roll(-1, 0)
        self.patch_stack[-1] = current_patch

        # update Pose_diff stack
        del_pose = Utils.to_robot_torch(self.prev_robot_pose_m, self.robot_pose).squeeze() / 0.1
        del_pose = del_pose / self.stats['pose_diff_max']
        self.prev_robot_pose_m = self.robot_pose
        self.prev_robot_pose_a = self.robot_pose

        self.pre_del_pose_stack = self.pre_del_pose_stack.roll(-1, 0)
        self.pre_del_pose_stack[-1] = del_pose.cuda()

        #update action stack
        cmd = torch.tensor(self.gt_cmds).mean(dim=0) #/ self.stats['cmd_vel_max']
        self.action_stack = self.action_stack.roll(-1, 0)
        self.action_stack[-1] = cmd.cuda()
        
        self.init += 1
        print(f"Initialization {self.init}/22")

        if self.init == 22:
            print("Initialization complete")
            # for patch in range(20):
            #     cv2.imshow("current_patch", cv2.normalize(cv2.resize(self.patch_stack[patch].cpu().numpy()[0], (500, 500)), None, norm_type=cv2.NORM_MINMAX))
            #     cv2.waitKey(100)
            #     print(f"pose_diff: {self.pre_del_pose_stack[patch][[0,1,5]]}, cmd: {self.action_stack[patch]}")
            # cv2.destroyAllWindows()
                         
    def loop(self, _)-> None:
        cur_time = time.time()
        if self.robot_pose is None:
            print("waiting for robot pose")
            self.gt_cmds = []
            return
        if self.mp.map_elevation is None:
            print("waiting for map data")
            self.gt_cmds = []
            return
        if len(self.gt_cmds) == 0:
            # print("waiting for ground truth commands")
            self.gt_cmds = []
            return 
        if len(self.gt_path) == 0:
            self.pred_path_a = [self.robot_pose]
            self.pred_path_m = [self.robot_pose]
            self.gt_path = [self.robot_pose]
            self.gt_cmds = []
            return
        if self.init < 22:
            self.initialize_stack()
            self.gt_cmds = []
            return
        if self.gt_path[-1] == self.robot_pose and len(self.gt_path) != 1:
            #print("waiting for robot to move")
            self.gt_cmds = []
            return 
        
        # if ackermann:
        pred_pose_a = self.ackermann_predict()
        # else:
        pred_pose_m = self.model_predict()

        if pred_pose_a is not None and pred_pose_m is not None:
            self.pred_path_a.append(pred_pose_a)
            self.pred_path_m.append(pred_pose_m)
            self.gt_path.append(self.robot_pose)
            self.no_iter += 1
            print(f"pred_pose_a: {torch.tensor(pred_pose_a)[[0,1,5]]}, pred_pose_m: {torch.tensor(pred_pose_m)[[0,1,5]]}, cmd: {torch.tensor(self.gt_cmds).mean(dim=0)}")
            # print(f"Time taken {int((time.time() - cur_time)*1000)}")
        if self.no_iter == publish_itrs:
            Utils.visualize(self.path_pub_gt, self.gt_path)
            Utils.visualize(self.path_pub_pred_a, self.pred_path_a)
            Utils.visualize(self.path_pub_pred_m, self.pred_path_m)
            self.no_iter = 0
            self.prev_robot_pose_a = self.robot_pose
            self.prev_robot_pose_m = self.robot_pose
            self.pred_path_a = [self.robot_pose]
            self.pred_path_m = [self.robot_pose]
            self.gt_path = [self.robot_pose]
        self.gt_cmds = []

if __name__ == '__main__':
    rospy.init_node("bc_TVERTI", anonymous=True)       # Initialize the node
    deploy = Deployment_Tverti()
    rospy.Timer(rospy.Duration(0.1), deploy.loop)
    try:
        while not rospy.is_shutdown():
            rospy.spin()
    except KeyboardInterrupt:
        print("Shutting down")
        cv2.destroyAllWindows()
        pass