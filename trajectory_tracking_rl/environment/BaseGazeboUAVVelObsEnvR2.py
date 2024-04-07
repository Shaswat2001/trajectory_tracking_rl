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

from trajectory_tracking_rl.environment.utils.CltSrvClasses import UavClientAsync,UavVelClientAsync, ResetSimClientAsync, GetUavPoseClientAsync, PauseGazeboClient, UnPauseGazeboClient
from trajectory_tracking_rl.environment.utils.PubSubClasses import StaticFramePublisher, LidarSubscriber, PathPublisherDDPG, PathPublisherSAC, PathPublisherSoftQ, PathPublisherTD3

class BaseGazeboUAVVelObsEnvR2(gym.Env):
    
    def __init__(self): 
        
        self.lidar_subscriber = LidarSubscriber()
        self.uam_publisher = UavVelClientAsync()
        self.get_uav_pose_client = GetUavPoseClientAsync()
        self.pause_sim = PauseGazeboClient()
        self.unpause_sim = UnPauseGazeboClient()
        self.reset_sim = ResetSimClientAsync()

        self.executor = rclpy.executors.MultiThreadedExecutor()
        self.executor.add_node(self.uam_publisher)
        self.executor.add_node(self.lidar_subscriber)
        self.executor.add_node(self.get_uav_pose_client)
        self.executor.add_node(self.pause_sim)
        self.executor.add_node(self.unpause_sim)
        self.executor.add_node(self.reset_sim)

        self.executor_thread = threading.Thread(target=self.executor.spin, daemon=True)
        self.executor_thread.start()

        self.state = None
        self.state_size = 362
        self.action_max = np.array([0.1,0.1])
        
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
        self.vel[:2] = self.vel[:2] + action[:2]
        self.vel = np.clip(self.vel,self.min_q_bound,self.max_q_bound)
        self.publish_simulator(self.vel)

        self.get_uav_pose()

        lidar,self.check_contact = self.get_lidar_data()
        # self.check_contact = self.collision_sub.get_collision_info()

        # print(f"New pose : {new_q}")
        # print(f"New velocity : {new_q_vel}")
        # self.q,self.qdot = self.controller.solve(new_q,new_q_vel)

        self.const_broken = self.constraint_broken()
        self.pose_error = self.get_error()
        reward,done = self.get_reward(lidar)
        constraint = self.get_constraint()
        info = self.get_info(constraint)

        if done:
            print(f"The constraint is broken : {self.const_broken}")
            print(f"The position error at the end : {self.pose_error}")
            print(f"The end pose of UAV is : {self.pose[:3]}")

        pose_diff = self.q_des - self.pose
        # prp_state = lidar
        prp_state = np.concatenate((self.vel[:2],lidar))
        prp_state = prp_state.reshape(1,-1)
        self.current_time += 1

        return prp_state, reward, done, info

    def get_reward(self,lidar):
        
        done = False
        pose_error = self.pose_error
        distance = np.linalg.norm(self.pose - self.starting_pose)
        reward = 0
        if not self.const_broken:
            self.previous_pose = self.pose

            collision_rwd = self.collision_reward(lidar)

            if np.min(lidar) > 1:
                reward = 3*np.min(lidar)
            else:
                reward = -5*np.min(lidar)

            reward += 2*distance

            reward += 4*collision_rwd
        
        else:
            reward = -200
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
        
    def reset(self,pose = np.array([0,-2,2]),pose_des = None,max_time = 110,publish_path = False):

        #initial conditions
        self.pose = pose
        self.starting_pose = pose
        self.vel = np.array([0,0,0])
        self.previous_pose = pose
        # self.qdot = np.array([0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01]) #initial velocity [x; y; z] in inertial frame - m/s
        
        if pose_des is None:
            self.q_des = np.random.randint([-1,-1,2],[2,2,3])

            while np.all(self.q_des == pose):
                self.q_des = np.random.randint([-1,-1,2],[2,2,3])
        else:
            self.q_des = pose_des

        print(f"The target pose is : {self.q_des}")

        pose_string = list(pose)[0:3]
        pose_string += [0,0,0]
        pose_string = f"{pose_string}"[1:-1]

        self.publish_simulator(self.vel)
        self.reset_sim.send_request(pose_string)
        time.sleep(0.1)

        self.lidar_subscriber.contact = False
        
        lidar,self.check_contact = self.get_lidar_data()
        # print(f"the man pose : {self.man_pos}")
        pose_diff = self.q_des - self.pose
        # pose_diff = np.clip(self.q_des - self.man_pos,np.array([-1,-1,-1]),np.array([1,1,1]))
        prp_state = np.concatenate((self.vel[:2],lidar))
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
        lidar,self.check_contact = self.get_lidar_data()
        # print(f"the man pose : {self.man_pos}")
        pose_diff = self.q_des - self.pose
        # pose_diff = np.clip(self.q_des - self.man_pos,np.array([-1,-1,-1]),np.array([1,1,1]))
        prp_state = np.concatenate((pose_diff,lidar))
        prp_state = prp_state.reshape(1,-1)
        self.current_time = 0
        self.const_broken = False
        time.sleep(0.1)

        return prp_state
    
    def collision_reward(self,lidar):

        theta_v = math.atan2(self.vel[1],self.vel[0])
        if theta_v < 0:
            theta_v += 2 * math.pi
        i = 0

        start_theta = 0
        reward = 0
        while i < lidar.shape[0]:

            if lidar[i] < 1:
                start_theta = i*np.pi/180
                i += 1
                while i < lidar.shape[0]:
                    
                    if lidar[i] < 1:
                        i += 1
                    else:
                        break

                final_theta = i*np.pi/180

                if start_theta <= theta_v <= final_theta:
                    reward = -10*np.linalg.norm(self.vel)
                    return reward
                else:
                    value = min(abs(theta_v - start_theta),abs(theta_v - final_theta))
                    reward = min(reward,value)
            else:
                i += 1

        return reward
    
    def publish_simulator(self,q):
    
        uav_vel = list(q)[0:3]
        uav_vel = f"{uav_vel}"[1:-1]

        uav_msg = String()
        uav_msg.data = uav_vel

        self.uam_publisher.send_request(uav_msg)

        self.unpause_sim.send_request()

        time.sleep(self.dt)

        self.pause_sim.send_request()

    def get_uav_pose(self):
    
        self.pose = self.get_uav_pose_client.send_request()

    def get_lidar_data(self):

        data,contact = self.lidar_subscriber.get_state()
        return data,contact