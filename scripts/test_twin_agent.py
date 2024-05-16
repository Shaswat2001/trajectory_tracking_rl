#!/usr/bin/env python3

import os
import pickle
import argparse
import sys
import numpy as np
import rclpy
import time
import matplotlib.pyplot as plt
# sys.path.insert(0, '/Users/shaswatgarg/Documents/WaterlooMASc/StateSpaceUAV')

from trajectory_tracking_rl.agent import DDPG,TD3,SAC,SoftQ,RCRL,SEditor,USL,SAAC,IDEA1,IDEA2,IDEA3,IDEA4
from trajectory_tracking_rl.pytorch_model import GaussianPolicyNetwork, PolicyNetwork,QNetwork,VNetwork,PhasicPolicyNetwork,PhasicQNetwork,ConstraintNetwork,MultiplierNetwork,SafePolicyNetwork,RealNVP,FeatureExtractor

from trajectory_tracking_rl.replay_buffer.Uniform_RB import ReplayBuffer,VisionReplayBuffer
from trajectory_tracking_rl.replay_buffer.Auxiliary_RB import AuxReplayBuffer
from trajectory_tracking_rl.replay_buffer.Constraint_RB import ConstReplayBuffer,CostReplayBuffer

from trajectory_tracking_rl.exploration.OUActionNoise import OUActionNoise

from trajectory_tracking_rl.exploration.OUActionNoise import OUActionNoise
from trajectory_tracking_rl.environment.BaseGazeboUAVVelEnv import BaseGazeboUAVVelEnv
from trajectory_tracking_rl.environment.BaseGazeboUAVVelObsEnvSimp import BaseGazeboUAVVelObsEnvSimp
from trajectory_tracking_rl.environment.BaseGazeboUAVVelObsEnvR2 import BaseGazeboUAVVelObsEnvR2
from trajectory_tracking_rl.environment.BaseGazeboUAVTrajectoryTracking import BaseGazeboUAVTrajectoryTracking
from trajectory_tracking_rl.environment.BaseGazeboUAVVelTrajectoryTracking import BaseGazeboUAVVelTrajectoryTracking
from trajectory_tracking_rl.environment.BaseGazeboUAVVel3DTrajectoryTracking import BaseGazeboUAVVel3DTrajectoryTracking
from trajectory_tracking_rl.environment.BaseGazeboUAVVelObsEnvPCD import BaseGazeboUAVVelObsEnvPCD

from trajectory_tracking_rl.teacher import TeacherController

def build_parse():

    parser = argparse.ArgumentParser(description="RL Algorithm Variables")

    parser.add_argument("Environment",nargs="?",type=str,default="uam_vel_gazebo_tracking",help="Name of OPEN AI environment")
    parser.add_argument("input_shape",nargs="?",type=int,default=[],help="Shape of environment state")
    parser.add_argument("n_actions",nargs="?",type=int,default=[],help="shape of environment action")
    parser.add_argument("max_action",nargs="?",type=float,default=[],help="Max possible value of action")
    parser.add_argument("min_action",nargs="?",type=float,default=[],help="Min possible value of action")

    parser.add_argument("Algorithm",nargs="?",type=str,default="DDPG",help="Name of RL algorithm")
    parser.add_argument('tau',nargs="?",type=float,default=0.005)
    parser.add_argument('gamma',nargs="?",default=0.99)
    parser.add_argument('actor_lr',nargs="?",type=float,default=0.0001,help="Learning rate of Policy Network")
    parser.add_argument('critic_lr',nargs="?",type=float,default=0.0001,help="Learning rate of the Q Network")
    parser.add_argument('mult_lr',nargs="?",type=float,default=0.0001,help="Learning rate of the LAG constraint")

    parser.add_argument("mem_size",nargs="?",type=int,default=100000,help="Size of Replay Buffer")
    parser.add_argument("batch_size",nargs="?",type=int,default=64,help="Batch Size used during training")
    parser.add_argument("n_episodes",nargs="?",type=int,default=50000,help="Total number of episodes to train the agent")
    parser.add_argument("target_update",nargs="?",type=int,default=2,help="Iterations to update the target network")
    parser.add_argument("vision_update",nargs="?",type=int,default=5,help="Iterations to update the vision network")
    parser.add_argument("delayed_update",nargs="?",type=int,default=100,help="Iterations to update the second target network using delayed method")
    parser.add_argument("enable_vision",nargs="?",type=bool,default=False,help="Whether you want to integrate sensor data")
    
    # SOFT ACTOR PARAMETERS
    parser.add_argument("temperature",nargs="?",type=float,default=0.2,help="Entropy Parameter")
    parser.add_argument("log_std_min",nargs="?",type=float,default=np.log(1e-4),help="")
    parser.add_argument("log_std_max",nargs="?",type=float,default=np.log(4),help="")
    parser.add_argument("aux_step",nargs="?",type=int,default=8,help="How often the auxiliary update is performed")
    parser.add_argument("aux_epoch",nargs="?",type=int,default=6,help="How often the auxiliary update is performed")
    parser.add_argument("target_entropy_beta",nargs="?",type=float,default=-3,help="")
    parser.add_argument("target_entropy",nargs="?",type=float,default=-3,help="")

    # MISC VARIABLES 
    parser.add_argument("save_rl_weights",nargs="?",type=bool,default=True,help="save reinforcement learning weights")
    parser.add_argument("save_results",nargs="?",type=bool,default=True,help="Save average rewards using pickle")

    # USL 
    parser.add_argument("eta",nargs="?",type=float,default=0.05,help="USL eta")
    parser.add_argument("delta",nargs="?",type=float,default=0.1,help="USL delta")
    parser.add_argument("Niter",nargs="?",type=int,default=20,help="Iterations")
    parser.add_argument("cost_discount",nargs="?",type=float,default=0.99,help="Iterations")
    parser.add_argument("kappa",nargs="?",type=float,default=5,help="Iterations")
    parser.add_argument("cost_violation",nargs="?",type=int,default=20,help="Save average rewards using pickle")

    # Safe RL parameters
    parser.add_argument("safe_iterations",nargs="?",type=int,default=5,help="Iterations to run Safe RL once engaged")
    parser.add_argument("safe_max_action",nargs="?",type=float,default=[],help="Max possible value of safe action")
    parser.add_argument("safe_min_action",nargs="?",type=float,default=[],help="Min possible value of safe action")

    # Environment Teaching parameters
    parser.add_argument("safe_iterations",nargs="?",type=int,default=5,help="Iterations to run Safe RL once engaged")
    parser.add_argument("teach_alg",nargs="?",type=str,default="alp_gmm",help="How to change the environment")

    # Environment parameters List
    parser.add_argument("max_obstacles",nargs="?",type=int,default=10,help="Maximum number of obstacles need in the environment")
    parser.add_argument("obs_region",nargs="?",type=float,default=6,help="Region within which obstacles should be added")

    # ALP GMM parameters
    parser.add_argument('gmm_fitness_fun',nargs="?", type=str, default="aic")
    parser.add_argument('warm_start',nargs="?", type=bool, default=False)
    parser.add_argument('nb_em_init',nargs="?", type=int, default=1)
    parser.add_argument('min_k', nargs="?", type=int, default=2)
    parser.add_argument('max_k', nargs="?", type=int, default=11)
    parser.add_argument('fit_rate', nargs="?", type=int, default=250)
    parser.add_argument('alp_buffer_size', nargs="?", type=int, default=500)
    parser.add_argument('random_task_ratio', nargs="?", type=int, default=0.2)
    parser.add_argument('alp_max_size', nargs="?", type=int, default=None)

    args = parser.parse_args("")

    return args

def train(args1,args2,env1,env2,agent1,agent2,teacher):

    velocity_traj = []
    s = env1.reset(pose = np.array([0,0,2]), pose_des = np.array([5,0,2]),max_time = 800)
    action = np.zeros((3))

    agent1.load("uam_vel_gazebo_tracking_3d")
    agent2.load("uam_vel_gazebo_obs_pcd")
    start_time = time.time()  
    obs_switch = 0
    # for _ in range(200):
    while True:
        # s = s.reshape(1,s.shape[0])
        start_time = time.time()

        action = agent1.choose_action(s,"testing")

        if (np.min(env1.distance) < 1.0 and env2.collision_switch(env1.lidar_data,action[0])):
            s = env1.get_intermediate_state()
            print("MAKING SWITCH")
            action = agent2.choose_action(s,"testing")
            # action = np.append(action,0)

        print(f"Time in seconds : {time.time() - start_time}")
        action[0,-1] = 0.0
        next_state,rwd,done,info = env1.step(action)
        velocity_traj.append(list(env1.pose))
        # print(next_state)   
        if done:
            break
            
        s = next_state

    velocity_traj = np.array(velocity_traj)
    plt.plot(velocity_traj[:,0],velocity_traj[:,1])
    plt.plot(env1.trajectory[:,0],env1.trajectory[:,1])
    plt.show()
    # f = open("config/saves/velocity_nine.pkl","wb")
    # pickle.dump(velocity_traj,f)
    # f.close()

if __name__=="__main__":

    rclpy.init(args=None)

    args1 = build_parse()
    args2 = build_parse()

    env1 = BaseGazeboUAVVel3DTrajectoryTracking()
    env2 = BaseGazeboUAVVelObsEnvPCD()
    
    args1.state_size = env1.state_size
    args1.input_shape = env1.state_size
    args1.n_actions = env1.action_space.shape[0]
    args1.max_action = env1.action_space.high
    args1.min_action = env1.action_space.low
    args1.safe_max_action = env1.safe_action_max
    args1.safe_min_action = -env1.safe_action_max

    args2.state_size = env2.state_size
    args2.input_shape = env2.state_size
    args2.n_actions = env2.action_space.shape[0]
    args2.max_action = env2.action_space.high
    args2.min_action = env2.action_space.low
    args2.safe_max_action = env2.safe_action_max
    args2.safe_min_action = -env2.safe_action_max


    agent1 = DDPG.DDPG(args = args1,policy = PolicyNetwork,critic = QNetwork,replayBuff = ReplayBuffer,exploration = OUActionNoise)
    agent2 = DDPG.DDPG(args = args2,policy = PolicyNetwork,critic = QNetwork,replayBuff = ReplayBuffer,exploration = OUActionNoise)

    teacher = None

    train(args1,args2,env1,env2,agent1,agent2,teacher)