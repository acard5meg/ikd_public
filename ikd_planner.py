#!/usr/bin/env python3

import os
import sys

## ALL PATHING IS BASED ON THIS FILES LOCATION WITHIN THE DIRECTORY

current = os.path.dirname(os.path.realpath(__file__))
vertiencoder_path = os.path.join(current, "verti-wheeler_pipeline/Tverti-wheeler/vertiencoder")
model_path = os.path.join(vertiencoder_path, "model")
utils_path = os.path.join(vertiencoder_path, "utils")
deployment_path = os.path.join(vertiencoder_path, "conf")
checkpoint_path = os.path.join(vertiencoder_path, "checkpoint")
pickle_path = os.path.join(current, "validation")
sys.path.append(model_path)
sys.path.append(utils_path)

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

from dt_models import get_model
from helpers import get_conf

import torch
import torch.nn as nn

# FOR GRAPH/PATH PLANNING
from collections import deque
from dijkstra_planner import dij_planner
from a_star_heur1 import a_star1_planner
from a_star_heur2 import a_star2_planner

# THIS GIVES WARNING ABOUT DISTUTILS BEING INSTALLED BEFORE SETUPTOOLS
# CURRENTLY GIVES WARNING BUT CAUTIONS MAY CAUSE ERROR/DISTUTILS IS DEPRACATED

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
        with open(os.path.join(checkpoint_path, "last_hopes/stats_tal.pkl"), 'rb') as f:
            self.stats = pickle.load(f)

        self.mp = MapProcessor()

        cfg = get_conf(os.path.join(deployment_path, "deployment.yaml"))
        self.model = get_model(cfg) # Load the model
        self.model.cuda()
        self.model.eval()
        self.model.requires_grad_(False)
  
        t_Params = sum([p.numel() for p in self.model.parameters() if p.requires_grad == True])
        print("Model loaded")
        print(self.model)
        print(f"Train Model: {self.model.training}, Trainable Params: {t_Params}")
        
        self.model(torch.zeros((1,20,1,40,40), dtype=torch.float32).cuda(), \
                   torch.zeros((1,20,2), dtype=torch.float32).cuda(), \
                    torch.zeros((1,20,6), dtype=torch.float32).cuda())
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

        # Circumvent GridMap being updated mid path plan algo
        self.is_planning = False
        self.map_hold = []

        # Used to build graphs out of current pose for path planning
        self.change_vals = []
        for i in range(-55, 56, 11):
            self.change_vals.append(Utils.ackermann(1, i/100))

        rospy.Subscriber('/elevation_mapping/elevation_map_raw', GridMap, self.gridMap_callback, queue_size=5, buff_size=6*sys.getsizeof(np.zeros((360,360))))
        rospy.Subscriber("/dlio/odom_node/odom", Odometry, self.odom_cb, queue_size = 100, buff_size=100*sys.getsizeof(Odometry()))
        # rospy.Subscriber("/natnet_ros/crawler/odom", Odometry, self.odom_cb, queue_size=1000, buff_size=100*sys.getsizeof(Odometry()) )
        rospy.Subscriber("/cmd_vel1", Float32MultiArray, self.cmd_vel_cb, queue_size = 100, buff_size=100*8*sys.getsizeof(float()))

        self.path_pub_pred_a = rospy.Publisher("/paths_pred_ack", Path, queue_size = 100 )
        self.path_pub_gt = rospy.Publisher("/paths_gt", Path, queue_size = 100)
        self.path_pub_pred_m = rospy.Publisher("/paths_pred_ssl", Path, queue_size = 100)
        # self.cmd_vel_pub = rospy.Publisher('/cmd_vel1', Float32MultiArray, queue_size=1)    # Publish to the cmd_vel topic

    def gridMap_callback(self, gridmap_msg):

        # Flag is set to true whenever building graph for path planning to 
        # prevent the map from updating and the edge weights between
        # poses to not be based on the same gridmap
        if not self.is_planning:
            self.mp.update_map(gridmap_msg)
            # print("GRIDMAP UPDATE")
        else:
            self.map_hold.append(gridmap_msg)
            # print("GRIDMAP HOLD")
            
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
        self.prev_robot_pose_a = cur_pose_pred

        return cur_pose_pred.tolist()

    def model_predict(self):
        if self.prev_robot_pose_m is None:
            self.prev_robot_pose_m = self.robot_pose
            return None       
        
        current_patch = torch.tensor(self.mp.get_elev_footprint(self.robot_pose,\
                                                                 (40,40))).cuda().unsqueeze(0)

        current_patch = torch.clip(current_patch - self.robot_pose[2], -0.5, 0.5)

        current_patch = current_patch / self.stats['footprint_max']
        
        self.patch_stack = self.patch_stack.roll(-1, 0)
        self.patch_stack[-1] = current_patch

        cmd = torch.tensor(self.gt_cmds).mean(dim=0) #/self.stats['cmd_vel_max']

        self.action_stack = self.action_stack.roll(-1, 0)
        self.action_stack[-1] = cmd.cuda()
        
        pose_diff = self.model(self.patch_stack.unsqueeze(0), \
                               self.action_stack.unsqueeze(0), \
                                self.pre_del_pose_stack.unsqueeze(0))
                
        self.pre_del_pose_stack = self.pre_del_pose_stack.roll(-1, 0)
        self.pre_del_pose_stack[-1] = pose_diff.clone()
        
        pose_diff = (pose_diff.squeeze(0).cpu() * self.stats['pose_diff_max'])

        cur_pose_pred = Utils.to_world_torch(self.prev_robot_pose_m, pose_diff).squeeze()
        self.prev_robot_pose_m = cur_pose_pred

        return cur_pose_pred.tolist() 
    
    
    def build_graph(self, steps = 2, final_x = 3.414258, final_y = 0.039283, normalize = False, elevation = False):
        """
        Builds the graph used in the path planning algorithm
        Uses values from Ackerman model to get updated poses
        Nodes are poses and edges are sum of traversibility map
        (another option for edges are difference between front and back wheels
        in the traversibility map)

        steps: Number of steps from start want method to calculate, steps >= 5
        severely limits performance

        Goal coordinates taken from rock1.bag
        final_x : value of goal x coordinate 
        final_y: value of goal y coordinate

        # May be slightly closer to end
        # CURRENT POSE: [3.7533748149871826, -0.019408313557505608]

        normalize : boolean, whether to normalize edge weights with starting traversibility map sum

        Issue with Euclidean distance for A* heuristic is the 
        edge weight is 1000+ and the distance is ~1 so using distance as a
        heuristic isn't useful. One thought was to normalize the edge weights
        using the current pose's traversibility map as the denominator.

        I believe using Dijkstra is faster than A* because the size of the map
        is small and we're only looking a couple steps out. Thus, the overhead
        of creating the additional data structure in A* to hold the heuristic value
        slows down the algorithm. 

        elevation : boolean, whether to use the wheel difference in the elevation footprint
                    as the edge weight or the sum of the traversibility footprint

        Returns: dictionary as a weighted graph
        {1x6 tuple-pose : {1x6 tuple-pose : float-weight}}
        2 special keys: 'start', 'end'
        """
        # x, y, z, roll, pitch, yaw = curr_pose

        # flag to prevent gridMap_callback from updating the traversibility map
        # mid algo. When the flag is true the gridMap_callback method stores
        # the message in a list. At the end of the build_graph method if the 
        # list has length greater than 0 the update_map method is called and
        # the traversibility map is updated to the most recent map message data
        # the list is then cleared
        self.is_planning = True

        print("STARTING PATH PLAN")
        
        curr_pose = self.robot_pose.copy()

        weight_dict = {}
        poses = deque([tuple(curr_pose)])
        closest_pose, closest_dist = curr_pose.copy(), ((curr_pose[0]-final_x)**2 + (curr_pose[1]-final_y)**2)**(1/2)
        
        # Used to normalize the edge weights if using A* with the Euclidean distance
        # as the heuristic
        # Also checking elevation because the value of the wheel difference is small
        # so I don't believe we need to normalize for the A* heuristic
        if not normalize or elevation:
            normalizing_weight = 1
        else:
            normalizing_weight = torch.sum(torch.tensor(self.mp.get_trav_footprint(curr_pose, (40,40)),\
                                        dtype=torch.float32).cuda().unsqueeze(0))

        # if want 1 step dictionary has 1 key (curr_pose) with 11 values (possible poses)
        # if want 2 steps dictionary needs 12 keys -> the current pose, and the 11 
        # possible next poses 
        add_one = 0
        if steps > 1:
            add_one = 1


        # Graph is built to have the following dictionary structure
        # key : current pose - tuple
        # value : dictionary - key : estimated pose, tuple
        #                      value : edge weight, float 
        # two special keys
        # 'start' <- beginning pose
        # 'end' <- pose closest to final_x, final_y in Euclidian distance 
        while len(weight_dict) < 11 ** (steps - 1) + add_one:

            build_pose = poses.popleft()

            weight_dict.update({build_pose : {}})

            for move in self.change_vals:
                new_pose = Utils.to_world_torch(build_pose, move).squeeze().tolist()

                # BOUND CHECKING FOR NEW POSES
                if not self.mp.is_on_map(new_pose):
                    print("OFF MAP")
                    continue

                poses.append(tuple(new_pose))
                
                # Constructing the edge weight based on whether we use the elevation map
                if not elevation:
                    edge_weight = torch.sum(torch.tensor(self.mp.get_trav_footprint(new_pose, (40,40)),\
                                            dtype=torch.float32).cuda().unsqueeze(0))
                else:                    
                    elev_map = torch.tensor(self.mp.get_elev_footprint(new_pose, (40,40)),\
                                            dtype=torch.float32).cuda().unsqueeze(0)
                    
                    back_l = torch.mean(elev_map[ : , : 4, : 4]).item()
                    back_r = torch.mean(elev_map[ : , : 4, elev_map.shape[2]-4 : ]).item()
                    front_l = torch.mean(elev_map[ : , elev_map.shape[2]-4 : , : 4]).item()
                    front_r = torch.mean(elev_map[ : , elev_map.shape[2]-4 : , elev_map.shape[2] - 4 : ]).item()

                    edge_weight = abs(back_l - back_r) + abs(front_l - front_r) + abs(front_l - back_l) + abs(front_r + back_r)
                
                edge_weight /= normalizing_weight

                weight_dict[tuple(build_pose)].update({tuple(new_pose) : edge_weight.item()})

                if ((new_pose[0]-final_x)**2 + (new_pose[1]-final_y)**2)**(1/2) < closest_dist:
                    closest_pose = new_pose.copy()

        weight_dict['start'] = tuple(curr_pose)
        weight_dict['end'] = tuple(closest_pose)

        print("ENDING PATH PLAN")
        if len(self.map_hold) > 0:
            self.mp.update_map(self.map_hold[-1])
            self.map_hold.clear()
        self.is_planning = False

        return weight_dict
    

    def path_planner(self, planner = 1, steps = 2, final_x = 3.414258, final_y = 0.039283, elevation = False):
        """
        Method to return path

        planner : 1 - Dijkstra
                  2 - A* Normalized edge weight, Euclidean distance heuristic
                  3 - A* Non-normalized edge weight, product of distance, edge weight heuristic

        steps, final_x, final_y, elevation: same definition as those given in build_graph function above

        Returns list of lists for paths

        Total runtime for 100,000 iterations at different number of steps edge weight is sum traversibility map

        2 steps
        DIJ time:  8.053734540939331
        A* unnormalize:  13.503831386566162
        A* normalize:  12.777570962905884
        3 steps
        DIJ time:  84.82603025436401
        A* unnormalize:  139.15969705581665
        A* normalize:  133.01054000854492
        4 steps
        DIJ time:  1464.0915904045105
        A* unnormalize:  1660.4615585803986
        A* normalize:  1582.9033591747284
        5 steps
        DIJ time:  23290.10035252571
        A* unnormalize:  22398.738894701004
        """

        if planner == 2:
            normalize = True
        else:
            normalize = False

        weight_dict = self.build_graph(planner, steps , final_x, final_y, normalize, elevation)

        if planner == 1:
            return dij_planner(weight_dict)
        elif planner == 2:
            return a_star1_planner(weight_dict)
        else:
            return a_star2_planner(weight_dict)
        

    def initialize_stack(self):
        if self.prev_robot_pose_m is None:
                self.prev_robot_pose_m = self.robot_pose
                return
            
        # update Current Patch
        current_patch = torch.tensor(self.mp.get_elev_footprint(self.robot_pose, (40,40)),\
                                      dtype=torch.float32).cuda().unsqueeze(0)
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
            
            print("waiting for ground truth commands")

            self.gt_cmds = []
            return 
        if len(self.gt_path) == 0:

            print("waiting fror ground truth path")

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

        ### running the path planner and print the path in SE2
        pred_x_t1 = self.path_planner()
        print(pred_x_t1)

        if pred_pose_a is not None and pred_pose_m is not None:
            self.pred_path_a.append(pred_pose_a)
            self.pred_path_m.append(pred_pose_m)
            self.gt_path.append(self.robot_pose)
            self.no_iter += 1
            print(f"pred_pose_a: {torch.tensor(pred_pose_a)[[0,1,5]]}, ", end="")
            print(f"pred_pose_m: {torch.tensor(pred_pose_m)[[0,1,5]]}, ", end="")
            print(f"cmd: {torch.tensor(self.gt_cmds).mean(dim=0)}")
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

    rospy.init_node("bc_TVERTI", anonymous=True)
    deploy = Deployment_Tverti()
    rospy.Timer(rospy.Duration(0.1), deploy.loop)
    
    try:
        while not rospy.is_shutdown():
            rospy.spin()
    except KeyboardInterrupt:
        print("Shutting down")
        cv2.destroyAllWindows()
        pass
    