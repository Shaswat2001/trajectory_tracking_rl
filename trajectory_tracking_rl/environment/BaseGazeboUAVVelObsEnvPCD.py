import gym
import rclpy
from rclpy.node import Node
from std_msgs.msg import String 
from math import pi
from gym import spaces
import numpy as np
import threading
import math
import time
from geometry_msgs.msg import Point
from visualization_msgs.msg import Marker
import open3d as o3d
from trajectory_tracking_rl.environment.utils.CltSrvClasses import UavClientAsync,UavVelClientAsync, ResetSimClientAsync, GetUavPoseClientAsync, PauseGazeboClient, UnPauseGazeboClient
from trajectory_tracking_rl.environment.utils.PubSubClasses import StaticFramePublisher, LidarSubscriber, PCDSubscriber

class BaseGazeboUAVVelObsEnvPCD(gym.Env):
    
    def __init__(self): 
        
        self.pcd_subscriber = PCDSubscriber()
        self.uam_publisher = UavVelClientAsync()
        self.get_uav_pose_client = GetUavPoseClientAsync()
        self.pause_sim = PauseGazeboClient()
        self.unpause_sim = UnPauseGazeboClient()
        self.reset_sim = ResetSimClientAsync()

        self.executor = rclpy.executors.MultiThreadedExecutor()
        self.executor.add_node(self.uam_publisher)
        self.executor.add_node(self.pcd_subscriber)
        self.executor.add_node(self.get_uav_pose_client)
        self.executor.add_node(self.pause_sim)
        self.executor.add_node(self.unpause_sim)
        self.executor.add_node(self.reset_sim)

        self.executor_thread = threading.Thread(target=self.executor.spin, daemon=True)
        self.executor_thread.start()

        self.state = None
        self.state_size = 606
        self.action_max = np.array([0.1,0.1,0.1])
        
        self.q = None
        self.q_des = None

        self.max_time = 10
        self.dt = 0.04
        self.current_time = 0

        self.q_vel_bound = np.array([3,3,3,1.5,1.5,1.5,1.5,1.5,1.5,1.5])
        self.max_q_bound = np.array([1.5,1.5,1.5])
        self.min_q_bound = np.array([-1.5,-1.5,-1.5])

        self.max_q_safety = np.array([8,8,8])
        self.min_q_safety = np.array([-8,-8,2])
        # self.max_q_safety = None
        # self.min_q_safety = None

        self.max_safety_engage = np.array([5.5,5.5,5.5])
        self.min_safety_engage = np.array([-5.5,-5.5,0.8])

        self.safe_action_max = np.array([8,8,8])
        self.safe_action_min = np.array([-8,-8,2])

        self.action_space = spaces.Box(-self.action_max,self.action_max,dtype=np.float64)

    def step(self, action):
        
        action = action[0]
        self.vel[:3] = self.vel[:3] + action[:3]
        self.vel = np.clip(self.vel,self.min_q_bound,self.max_q_bound)
        self.publish_simulator(self.vel)

        self.get_uav_pose()

        downsampled_pcd,pcd_data,pcd_range,self.check_contact = self.get_lidar_data()
        heading = self.get_desired_heading()
        # self.check_contact = self.collision_sub.get_collision_info()

        # print(f"New pose : {new_q}")
        # print(f"New velocity : {new_q_vel}")
        # self.q,self.qdot = self.controller.solve(new_q,new_q_vel)

        self.const_broken = self.constraint_broken()
        self.pose_error = self.get_error()
        reward,done = self.get_reward(pcd_data,pcd_range,heading)
        constraint = self.get_constraint()
        info = self.get_info(constraint)

        if done:

            # theta_v = math.atan2(self.vel[1],self.vel[0])
            # if theta_v < 0:
            #     theta_v += 2 * math.pi
            # i = 0
        
            print(f"The constraint is broken : {self.const_broken}")
            print(f"The end pose of UAV is : {self.pose[:3]}")
            # print(f"The final heading of UAV is : {theta_v}")

        pose_diff = self.q_des - self.pose
        # prp_state = lidar
        prp_state = np.concatenate((pose_diff,self.vel,downsampled_pcd))
        prp_state = prp_state.reshape(1,-1)
        self.current_time += 1

        return prp_state, reward, done, info

    def get_reward(self,pcd_data,pcd_range,heading):
        
        done = False
        pose_error = self.pose_error
        distance = np.linalg.norm(self.pose - self.starting_pose)
        reward = 0
        if not self.const_broken:
            self.previous_pose = self.pose

            collision_rwd,heading_reward = self.collision_reward(pcd_data,self.vel,heading)

            if pose_error < 0.1:
                reward = 10
                done = True
            else:
                # if np.min(pcd_range) > 0.5:
                #     reward = 2*np.min(pcd_range)
                # else:
                #     reward = -5*(1 - np.min(pcd_range))

                # reward += 2*distance

                # reward += 4*collision_rwd

                reward = 5*np.min(pcd_range) - 15*pose_error

                # reward -= 3*pose_error

        else:
            reward = -300
            done = True

        if self.current_time > self.max_time:
            done = True
            reward -= 2

        return reward,done
    
    def get_constraint(self):
        
        constraint = 0
        if self.const_broken:

            for i in range(self.vel.shape[0]):
                if self.vel[i] > self.max_q_bound[i]:
                    constraint+= (self.vel[i] - self.max_q_bound[i])*10
                elif self.vel[i] < self.min_q_bound[i]:
                    constraint+= abs(self.vel[i] - self.min_q_bound[i])*10

            if constraint < 0:
                constraint = 10
        else:

            for i in range(self.vel.shape[0]):
                constraint+= (abs(self.vel[i]) - self.max_q_bound[i])*10

        return constraint

    def get_info(self,constraint):

        info = {}
        info["constraint"] = constraint
        info["safe_reward"] = -constraint
        info["safe_cost"] = 0
        info["negative_safe_cost"] = 0
        info["engage_reward"] = -10

        if np.any(self.vel > self.max_q_safety) or np.any(self.vel < self.min_q_safety):
            info["engage_reward"] = 10
            
        if constraint > 0:
            info["safe_cost"] = 1
            info["negative_safe_cost"] = -1

        return info

    def constraint_broken(self):
        
        if self.check_contact:
            return True
        
        return False
    
    def get_error(self):

        pose_error =  np.linalg.norm(self.pose - self.q_des) 

        return pose_error
        
    def reset(self,pose = np.array([0,0,2]),pose_des = None,max_time = 20,publish_path = False):

        #initial conditions
        self.pose = pose
        self.starting_pose = pose

        self.vel = np.random.uniform(low=[-1.5,-1.5,0],high=[1.5,1.5,0])
        # self.vel = np.array([0,0,0])
        self.previous_pose = pose
        # self.qdot = np.array([0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01]) #initial velocity [x; y; z] in inertial frame - m/s
        
        if pose_des is None:
            self.q_des = np.random.randint([-1,-1,1],[2,2,4])

            while np.all(self.q_des == pose):
                self.q_des = np.random.randint([-1,-1,1],[2,2,4])
        else:
            self.q_des = pose_des

        for i in range(3):

            if self.q_des[i] < 0:
                self.vel[i] = -abs(self.vel[i])
            else:
                self.vel[i] = abs(self.vel[i])

        # print(f"The target heading is : {self.angle}")

        pose_string = list(pose)[0:3]
        pose_string += [0,0,0]
        pose_string = f"{pose_string}"[1:-1]

        self.publish_simulator(self.vel)
        self.reset_sim.send_request(pose_string)
        time.sleep(0.1)

        self.pcd_subscriber.contact = False
        downsampled_pcd,_,_,self.check_contact = self.get_lidar_data()
        heading = self.get_desired_heading()

        print(f"The target heading is : {heading}")
        print(f"The velocity is : {self.vel}")
        pose_diff = self.q_des - self.pose
        # pose_diff = np.clip(self.q_des - self.man_pos,np.array([-1,-1,-1]),np.array([1,1,1]))
        prp_state = np.concatenate((pose_diff,self.vel,downsampled_pcd))
        # prp_state = lidar
        prp_state = prp_state.reshape(1,-1)
        self.current_time = 0
        self.const_broken = False
        self.max_time = max_time
        time.sleep(0.1)

        return prp_state
    
    def reset_test(self,q_des,max_time,algorithm):


        self.pose = np.array([-6.0,-6.0,1])
        self.vel = np.array([0,0,0])
        self.previous_pose = self.pose
        self.algorithm = algorithm

        self.q_des = q_des
        self.max_time = max_time
        print(f"The target pose is : {self.q_des}")

        self.publish_simulator(self.pose)
        downsampled_pcd,_,_,self.check_contact = self.get_lidar_data()
        # print(f"the man pose : {self.man_pos}")
        pose_diff = self.q_des - self.pose
        # pose_diff = np.clip(self.q_des - self.man_pos,np.array([-1,-1,-1]),np.array([1,1,1]))
        prp_state = np.concatenate((pose_diff,downsampled_pcd))
        prp_state = prp_state.reshape(1,-1)
        self.current_time = 0
        self.const_broken = False
        time.sleep(0.1)

        return prp_state
    
    def collision_reward(self,lidar,vel,heading):

        theta_v = self.get_orientation(vel)
        i = 0

        if np.linalg.norm(vel) != 0:
            direction_heading = vel/np.linalg.norm(vel)
        else:
            direction_heading = vel

        heading_reward = np.dot(direction_heading[:2],heading)
    
        reward = 0

        while i < lidar.shape[0]:
            distance = 10
            start_theta = self.get_orientation(lidar[i])
            final_theta = start_theta

            distance = min(np.linalg.norm(lidar[i,:]),distance)

            while abs(start_theta - final_theta) < 0.2 and i < lidar.shape[0]:
                
                distance = min(np.linalg.norm(lidar[i,:]),distance)
                final_theta = self.get_orientation(lidar[i])

                i += 1

            if start_theta <= theta_v <= final_theta and distance < 1:
                reward = -10*np.linalg.norm(vel)
                return reward,heading_reward
            else:
                value = min(abs(theta_v - start_theta),abs(theta_v - final_theta))
                reward = max(reward,value)

        return reward,heading_reward
    
    def collision_switch(self,lidar,vel):

        theta_v = self.get_orientation(vel)
        i = 0
    
        while i < lidar.shape[0]:
            start_theta = self.get_orientation(lidar[i])
            final_theta = start_theta

            while abs(start_theta - final_theta) < 0.2 and i < lidar.shape[0]:
                
                final_theta = self.get_orientation(lidar[i])

                i += 1

            if start_theta <= theta_v <= final_theta:
                return True
            
        return False
    
    def publish_simulator(self,q):
    
        uav_vel = list(q)[0:3]
        uav_vel = f"{uav_vel}"[1:-1]

        uav_msg = String()
        uav_msg.data = uav_vel

        self.uam_publisher.send_request(uav_msg)

        self.unpause_sim.send_request()

        time.sleep(self.dt)

        self.pause_sim.send_request()

    def get_desired_heading(self):

        heading = self.q_des - self.pose
        heading = heading/np.linalg.norm(heading)

        return heading[:2]

    def get_uav_pose(self):
    
        self.pose = self.get_uav_pose_client.send_request()

    def get_lidar_data(self):

        data,distance,contact = self.pcd_subscriber.get_state()
        max_points = self.pcd_subscriber.max_points

        if np.all(data == np.zeros((max_points,3))):
            return data.flatten(),data,distance,contact
        
        downsampled_pcd = np.zeros((max_points,3))
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(data)
        pcd = pcd.voxel_down_sample(0.08)

        if distance.shape[0] == 0:
            distance = np.full(shape=(max_points),fill_value=1.0)

        xyz_load = np.asarray(pcd.points)

        if xyz_load.shape[0] == 0:            
            return downsampled_pcd.flatten(),data,distance,contact
        
        downsampled_pcd[:xyz_load.shape[0],:] = xyz_load[:min(xyz_load.shape[0],max_points),:]
        downsampled_pcd[xyz_load.shape[0]:,:] = xyz_load[-1,:]
        
        return downsampled_pcd.flatten(),data,distance,contact
    
    def get_orientation(self,point):

        theta = math.atan2(point[1],point[0])
        if theta < 0:
            theta += 2 * math.pi

        return theta