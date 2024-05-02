import gym
import rclpy
from rclpy.node import Node
from std_msgs.msg import String 
from math import pi
from gym import spaces
import numpy as np
import threading
import time
import math
import open3d as o3d

from trajectory_tracking_rl.environment.utils.CltSrvClasses import UavClientAsync,UavVelClientAsync, ResetSimClientAsync, GetUavPoseClientAsync, PauseGazeboClient, UnPauseGazeboClient
from trajectory_tracking_rl.environment.utils.PubSubClasses import StaticFramePublisher, LidarSubscriber, PCDSubscriber

class BaseGazeboUAVVel3DTrajectoryTracking(gym.Env):
    
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
        self.state_size = 9
        self.action_max = np.array([0.1,0.1,0.1])
        
        self.pose = None
        self.q_des = None
        self.check_contact = False

        self.max_points = 9
        self.current_target = 1
        self.max_time = 25
        self.max_trajectory_step = 6
        self.dt = 0.04
        self.current_time = 0
        self.current_subtraj_time = 0

        self.max_q_bound = np.array([1.5,1.5,1.5])
        self.min_q_bound = np.array([-1.5,-1.5,-1.5])

        self.max_q_safety = np.array([8,8,8])
        self.min_q_safety = np.array([-8,-8,2])

        self.max_safety_engage = np.array([5.5,5.5,5.5])
        self.min_safety_engage = np.array([-5.5,-5.5,0.8])

        self.safe_action_max = np.array([8,8,8])
        self.safe_action_min = np.array([-8,-8,2])

        self.action_space = spaces.Box(-self.action_max,self.action_max,dtype=np.float64)

    def step(self, action):
        
        action = action[0]
        self.vel = self.vel + action
        self.vel = np.clip(self.vel,self.min_q_bound,self.max_q_bound)
        self.publish_simulator(self.vel)

        self.get_uav_pose()

        current_target_pose,self.current_target = self.get_closest_point()
        # current_target_pose = self.trajectory[self.current_target]
        # if current_target < self.current_target and self.current_target > 6:
        #     self.current_target += 1
        #     self.current_target = min(29,self.current_target)
        #     current_target_pose = self.trajectory[self.current_target]
        # else:
        #     self.current_target = current_target
        # current_target_pose = self.trajectory[self.current_target]

        _,self.lidar_data,self.distance,_ = self.get_lidar_data()
        
        self.const_broken = self.constraint_broken()
        self.pose_error = self.get_error(current_target_pose)
        self.global_error = self.get_error(self.q_des)
        self.max_deviation = self.calculate_distance_from_line()

        self.current_time += 1
        self.current_subtraj_time += 1
        reward,done = self.get_reward()
        constraint = self.get_constraint()
        info = self.get_info(constraint)

        if done:            
            print(f"The constraint is broken : {self.const_broken}")
            print(f"The position error at the end : {self.global_error}")
            print(f"The end pose of UAV is : {self.pose[:3]}")

        pose_diff = self.get_subregion()
        prp_state = pose_diff
        # prp_state = np.concatenate((pose_diff,lidar))
        prp_state = prp_state.reshape(1,-1)

        # if self.const_broken:

        #     self.get_safe_pose()
        #     self.publish_simulator(self.previous_pose)
        #     self.pose = self.previous_pose

        #     self.vel = self.vel - action[:3]
            # self.publish_simulator(self.vel)

        return prp_state, reward, done, info
    
    def calculate_distance_from_line(self):

        current_target_pose = self.trajectory[self.current_target]
        initial_pose = self.trajectory[self.current_target - 1]

        # if current_target_pose[0] == initial_pose[0]:
        #     distance = abs(self.pose[0] - initial_pose[0])

        
        A = np.linalg.norm(current_target_pose - initial_pose)
        B = np.linalg.norm(current_target_pose - self.pose)
        C = np.linalg.norm(np.dot(current_target_pose - self.pose,current_target_pose - initial_pose))

        distance = math.sqrt(abs((B**2)*(A**2) - C**2)/A**2)

        return distance

    def get_reward(self):
        
        done = False
        pose_error = self.pose_error
        global_error = self.global_error
        deviation = self.max_deviation
        reward = 0
        if not self.const_broken:
            self.previous_pose = self.pose

            if pose_error < 0.1:
                # self.current_target+=1
                # self.current_subtraj_time = 0
                reward = 10
            else:

                reward = -deviation*4 - pose_error*2 - global_error*3

        else:
            reward = -300
            # done = True

        # if self.current_subtraj_time == 3:
        #     # print(f"Maximum error : {pose_error}")
        #     # print(f"Maximum deviation : {deviation}")
        #     self.current_target+=1
        #     self.current_subtraj_time = 0
        
        if self.current_target > self.max_points or self.current_time > self.max_time:
            done = True
            reward -= 2

        # if self.current_target > self.max_points:
        #     # print("HI")
        #     done = True
        #     reward -= 2

        return reward,done
    

    def get_closest_point(self):

        distance = np.linalg.norm(self.trajectory - self.pose, axis = 1)

        ahead_indices = np.where(np.dot(self.trajectory - self.pose, self.pose) > 0)[0]

        if len(ahead_indices) == 0:
            closest_point_arg = np.argmin(distance)
            return self.trajectory[closest_point_arg],closest_point_arg
        
        else:
            closest_point_arg = ahead_indices[np.argmin(distance[ahead_indices])]
            # Return the closest point
            return self.trajectory[closest_point_arg],closest_point_arg
        
    
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
    
    def get_subregion(self):

        points = np.zeros((3,3))
        points_list = self.trajectory[self.current_target:min(self.current_target+3,self.max_points+1),:]
        points[:len(points_list)] = points_list
        points[len(points_list):,:] = points[len(points_list)-1]
        points -= self.pose
        
        return points.flatten()

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
    
    def get_error(self,target):

        pose_error =  np.linalg.norm(self.pose - target) 

        return pose_error
        
    def reset(self,pose = np.array([0,0,2]),pose_des = None,max_time = 25):

        # self.pose = np.round(np.random.uniform(low = -0.2,high=0.2,size=(2,)),1)
        # self.pose = np.append(self.pose,2.0)
        self.pose = pose
        self.vel = np.array([0,0,0])
        self.previous_pose = self.pose
        
        if pose_des is None:
            self.q_des = np.random.randint([-1,-1,1],[2,2,4])

            while np.all(self.q_des == pose):
                self.q_des = np.random.randint([-1,-1,1],[2,2,4])
        else:
            self.q_des = pose_des

        # trajectory1 = self.generate_sample_trajectory(np.array([0,0,2]),np.array([1,1,2]))
        # trajectory2 = self.generate_sample_trajectory(np.array([1,1,2]),np.array([2,1,2]))
        # self.trajectory = np.concatenate((trajectory1[:-1,:],trajectory2))
        self.trajectory = self.generate_sample_trajectory(pose,self.q_des)

        _,self.lidar_data,self.distance,_ = self.get_lidar_data()

        print(f"The target pose is : {self.q_des}")

        pose_string = list(self.pose)[0:3]
        pose_string += [0,0,0]
        pose_string = f"{pose_string}"[1:-1]

        self.publish_simulator(self.vel)
        self.reset_sim.send_request(pose_string)
        time.sleep(0.1)

        self.current_time = 0
        self.const_broken = False
        self.check_contact = False
        self.current_target = 1
        self.current_subtraj_time = 0
        self.max_time = max_time

        pose_diff = self.get_subregion()
        prp_state = pose_diff
        prp_state = prp_state.reshape(1,-1)

        time.sleep(0.1)

        return prp_state
    
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

    def get_intermediate_state(self):

        lidar,_,_,_ = self.get_lidar_data()
        heading = self.get_desired_heading()
        pose_diff = self.q_des - self.pose
        # pose_diff = np.clip(self.q_des - self.man_pos,np.array([-1,-1,-1]),np.array([1,1,1]))
        prp_state = np.concatenate((pose_diff,lidar))
        # prp_state = lidar
        prp_state = prp_state.reshape(1,-1)

        return prp_state

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
    
    def generate_sample_trajectory(self,q_start,q_des):

        fraction = np.linspace(0,1,self.max_points+1)
        points = q_start + fraction[:,np.newaxis]*(q_des - q_start)

        return np.round(points,3)

    def get_desired_heading(self):

        heading = self.q_des - self.pose
        heading = heading/np.linalg.norm(heading)

        return heading[:2]