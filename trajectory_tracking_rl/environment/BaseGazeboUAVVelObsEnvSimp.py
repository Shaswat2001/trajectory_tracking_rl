import gym
import rclpy
from rclpy.node import Node
from std_msgs.msg import String 
from math import pi
from gym import spaces
import numpy as np
import threading
import time
from geometry_msgs.msg import Point
from visualization_msgs.msg import Marker

from trajectory_tracking_rl.environment.utils.CltSrvClasses import UavClientAsync
from trajectory_tracking_rl.environment.utils.PubSubClasses import StaticFramePublisher, LidarSubscriber, PathPublisherDDPG, PathPublisherSAC, PathPublisherSoftQ, PathPublisherTD3

class BaseGazeboUAVVelObsEnvSimp(gym.Env):
    
    def __init__(self): 
        
        self.uam_publisher = UavClientAsync()
        self.lidar_subscriber = LidarSubscriber()

        self.executor = rclpy.executors.MultiThreadedExecutor()
        self.executor.add_node(self.uam_publisher)
        self.executor.add_node(self.lidar_subscriber)

        self.executor_thread = threading.Thread(target=self.executor.spin, daemon=True)
        self.executor_thread.start()

        self.state = None
        self.state_size = 363
        self.action_max = np.array([0.1,0.1])
        
        self.q = None
        self.q_des = None

        self.max_time = 10
        self.dt = 0.07
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
        self.pose = np.array([self.dt*self.vel[i] + self.pose[i] for i in range(self.vel.shape[0])])
        self.pose = np.clip(self.pose,np.array([-12,-12,0.5]),np.array([12,12,3]))
        self.publish_simulator(self.pose)
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
        pose_diff = np.clip(pose_diff,np.array([-1,-1,-1]),np.array([1,1,1]))
        prp_state = np.concatenate((pose_diff,lidar))
        prp_state = prp_state.reshape(1,-1)
        self.current_time += 1

        return prp_state, reward, done, info

    def get_reward(self,lidar):
        
        done = False
        pose_error = self.pose_error
        reward = 0
        if not self.const_broken:
            self.previous_pose = self.pose

            if pose_error < 0.1:
                done = True
                reward = 30
            else:
                reward = -(pose_error*25) + 9*np.min(lidar)
        
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
        
        # if np.any(self.vel[:3] > self.max_q_bound[:3]) or np.any(self.vel[:3] < self.min_q_bound[:3]):
        #     return True
        
        return False
    
    def get_error(self):

        pose_error =  np.linalg.norm(self.pose - self.q_des) 

        return pose_error
        
    def reset(self,pose = np.array([0,0,2]),pose_des = None,max_time = 10,publish_path = False):

        #initial conditions
        self.pose = pose
        self.vel = np.array([0,0,0])
        self.previous_pose = pose
        # self.qdot = np.array([0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01]) #initial velocity [x; y; z] in inertial frame - m/s
        
        if pose_des is None:
            self.q_des = np.random.randint([-1,-1,2],[2,2,3])
        else:
            self.q_des = pose_des

        print(f"The target pose is : {self.q_des}")

        self.publish_simulator(self.pose)
        
        lidar,self.check_contact = self.get_lidar_data()
        # print(f"the man pose : {self.man_pos}")
        pose_diff = self.q_des - self.pose
        pose_diff = np.clip(pose_diff,np.array([-1,-1,-1]),np.array([1,1,1]))
        prp_state = np.concatenate((pose_diff,lidar))
        prp_state = prp_state.reshape(1,-1)
        self.current_time = 0
        self.const_broken = False

        if np.all(self.q_des[:2] == np.array([-1,-1])) or np.all(self.q_des[:2] == np.array([1,1])):
            self.max_time = 15
        else:
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
        
    def publish_simulator(self,q):
        
        uav_pos_ort = list(q)[0:3]
        uav_pos_ort += [0,0,0]
        uav_pos_ort = f"{uav_pos_ort}"[1:-1]

        uav_msg = String()
        uav_msg.data = uav_pos_ort

        self.uam_publisher.send_request(uav_msg)

    def get_lidar_data(self):

        data,contact = self.lidar_subscriber.get_state()
        return data,contact