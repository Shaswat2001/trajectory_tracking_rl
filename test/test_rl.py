import sys
sys.path.insert(1, "/Users/shaswatgarg/Documents/WaterlooMASc/StateSpaceUAV/")

import argparse
from audioop import avg
import gym
import numpy as np
from Agent import DDPG,TD3,SAC,SoftQ,PSAC_CEPO,RCRL,SEditor,USL,SAAC,IDEA4
import torch
from mpl_toolkits.mplot3d import Axes3D
from trajectory_tracking_rl.pytorch_model import GaussianPolicyNetwork, PolicyNetwork,QNetwork,VNetwork,PhasicPolicyNetwork,PhasicQNetwork,ConstraintNetwork,MultiplierNetwork,RealNVP
from ReplayBuffer.Uniform_RB import ReplayBuffer
from ReplayBuffer.Auxiliary_RB import AuxReplayBuffer
from ReplayBuffer.Constraint_RB import CostReplayBuffer
from Exploration.OUActionNoise import OUActionNoise
import matplotlib.pyplot as plt
from controllers.PID import CascadeController
from Environment.Quadrotor.QuadrotorObsEnv import QuadrotorObsEnv
from Environment.Quadrotor.BaseQuadrotorEnv import BaseQuadrotorEnv
from Environment.Quadrotor.QuadrotorTeachEnv import QuadrotorTeachEnv

from Environment.AerialManipulator.BaseUAMEnv import BaseUAMEnv
from Environment.AerialManipulator.UAMObsEnv import UAMObsEnv
from Teacher import TeacherController

from scripts.main import build_parse
import pickle

np.warnings.filterwarnings('ignore', category=np.VisibleDeprecationWarning)   

def testRL(args,env,agent):

    position,velocity,angle,angle_vel,motor_thrust,body_torque = test(args,env,agent)
    plot_results(position,velocity,angle,angle_vel,motor_thrust,body_torque,args,env)

def test(args,env,agent,teacher):

    agent.load(args.Environment)
    if teacher is not None:
            teacher.generate_env_param()
    s = env.reset()
    reward = 0
    itr = 0
    constraint_broke = 0
    while True:
        action = agent.choose_action(s,"eval")
        next_state,rwd,done,info = env.step(action)
        if info["constraint"] > 0:
                constraint_broke+=1
        itr+=1
        reward+=rwd
        # print(next_state)
        if done:
            break
            
        s = next_state

    if constraint_broke > 0:
        print("CONSTRAINT WAS BROKEN")
        
    angle = env.controller.orientation_list
    motor_thrust = env.controller.motor_thrust_list
    body_torque = env.controller.torque_list
    angle_vel = env.controller.ang_vel_list
    velocity = env.controller.lnr_vel_list
    position = env.controller.pos_list

    return position,velocity,angle,angle_vel,motor_thrust,body_torque


def plot_results(position,velocity,angle,angle_vel,motor_thrust,body_torque,args,env):

    fig =  plt.figure(num=None, figsize=(16, 8), dpi=80, facecolor='w', edgecolor='k')

    # 3D Flight path
    axes = fig.add_subplot(2, 4, 1, projection='3d')
    if "obs" in args.Environment or "teach" in args.Environment:
        env.grid.visualize_grid(axes)
    axes.plot(position[0], position[1], position[2])
    axes.set_title('Flight Path')
    axes.set_xlabel('x (m)')
    axes.set_ylabel('y (m)')
    axes.set_zlabel('z (m)')

    # Lateral position plots
    axes = fig.add_subplot(2, 4, 2)
    axes.plot(position[0], label= 'x')
    axes.plot(position[1], label= 'y')
    axes.set_title('Lateral Postion')
    axes.set_xlabel('time (s)')
    axes.set_ylabel('position (m)')
    axes.legend()

    # Vertical position plot
    axes = fig.add_subplot(2, 4, 3)
    axes.plot(position[2], label= 'z')
    axes.set_title('Vertical Position')
    axes.set_xlabel('time (s)')
    axes.set_ylabel('altitude (m)')

    # Lateral velocity plots
    axes = fig.add_subplot(2, 4, 4)
    axes.plot(velocity[0], label= 'd(x)/dt')
    axes.plot(velocity[1], label= 'd(y)/dt')
    axes.plot(velocity[2], label= 'd(z)/dt')
    axes.set_title('Linear Velocity')
    axes.set_xlabel('time (s)')
    axes.set_ylabel('velocity (m/s)')
    axes.legend()

    # Motor speed plots
    axes = fig.add_subplot(2, 4, 5)
    axes.plot(motor_thrust[0], label= 'motor 1')
    axes.plot(motor_thrust[1], label= 'motor 2')
    axes.plot(motor_thrust[2], label= 'motor 3')
    axes.plot(motor_thrust[3], label= 'motor 4')
    axes.set_title('Motor Thrust')
    axes.set_xlabel('time (s)')
    axes.set_ylabel('Motor Thrust (N)')
    axes.legend()

    # Body torque over time
    axes = fig.add_subplot(2, 4, 6)
    axes.plot(body_torque[0], label= 'x')
    axes.plot(body_torque[1], label= 'y')
    axes.plot(body_torque[2], label= 'z')
    axes.set_title('Body Torque')
    axes.set_xlabel('time (s)')
    axes.set_ylabel('torque (n-m)')
    axes.legend()

    # Angles over time
    axes = fig.add_subplot(2, 4, 7)
    axes.plot(angle[0], label= 'phi')
    axes.plot(angle[1], label= 'theta')
    axes.plot(angle[2], label= 'psi')
    axes.set_title('Euler Angles')
    axes.set_xlabel('time (s)')
    axes.set_ylabel('angle (deg)')
    axes.legend()

    # Angular velocity over time
    axes = fig.add_subplot(2, 4, 8)
    axes.plot(angle_vel[0], label= 'd(phi)/dt')
    axes.plot(angle_vel[1], label= 'd(theta)/dt')
    axes.plot(angle_vel[2], label= 'd(psi)/dt')
    axes.set_title('Angular Velocity')
    axes.set_xlabel('time (s)')
    axes.set_ylabel('angular velocity (deg/s)')
    axes.legend()

    plt.show()
    
def main(algorithm = None):

    args = build_parse()

    if algorithm is not None:
        args.Algorithm = algorithm

    param_bound = dict()
    param_bound["n_obstacles"] = [0,args.max_obstacles]
    param_bound["obs_centre"] = [0,args.obs_region,3]

    if "quadrotor_obs" == args.Environment:
        env = QuadrotorObsEnv(controller=CascadeController)
    elif "quadrotor_teach" == args.Environment:
        env = QuadrotorTeachEnv(controller=CascadeController,load_obstacle=False)
    elif "UAM" == args.Environment:
        env = BaseUAMEnv(controller=CascadeController)
    elif "UAM_obs" == args.Environment:
        env = UAMObsEnv(controller=CascadeController)
    else:
        env = BaseQuadrotorEnv(controller=CascadeController)

    args.input_shape = env.state_size
    args.n_actions = env.action_space.shape[0]
    args.max_action = env.action_space.high
    args.min_action = env.action_space.low

    if args.Algorithm == "DDPG":
        agent = DDPG.DDPG(args = args,policy = PolicyNetwork,critic = QNetwork,replayBuff = ReplayBuffer,exploration = OUActionNoise)
    elif args.Algorithm == "TD3":
        agent = TD3.TD3(args = args,policy = PolicyNetwork,critic = QNetwork,replayBuff = ReplayBuffer,exploration = OUActionNoise)
    elif args.Algorithm == "SAC":
        agent = SAC.SAC(args = args,policy = GaussianPolicyNetwork,critic = QNetwork,valueNet=VNetwork,replayBuff = ReplayBuffer,exploration = OUActionNoise)
    elif args.Algorithm == "SoftQ":
        agent = SoftQ.SoftQ(args = args,policy = PolicyNetwork,critic = QNetwork,replayBuff = ReplayBuffer,exploration = OUActionNoise)
    elif args.Algorithm == "PSAC":
        agent = PSAC_CEPO.PSAC_CEPO(args = args,policy = PhasicPolicyNetwork,critic = PhasicQNetwork,valueNet=VNetwork,replayBuff = ReplayBuffer,auxBuff=AuxReplayBuffer,exploration = OUActionNoise)
    elif args.Algorithm == "RCRL":
        agent = RCRL.RCRL(args = args,policy = PolicyNetwork,critic = QNetwork,multiplier=MultiplierNetwork,replayBuff = CostReplayBuffer,exploration = OUActionNoise)
    elif args.Algorithm == "SEditor":
        agent = SEditor.SEditor(args = args,policy = PolicyNetwork,critic = QNetwork,replayBuff = CostReplayBuffer,exploration = OUActionNoise)
    elif args.Algorithm == "USL":
        agent = USL.USL(args = args,policy = PolicyNetwork,critic = QNetwork,replayBuff = CostReplayBuffer,exploration = OUActionNoise)
    elif args.Algorithm == "SAAC":
        agent = SAAC.SAAC(args = args,policy = GaussianPolicyNetwork,critic = QNetwork,valueNet=VNetwork, replayBuff = CostReplayBuffer,exploration = OUActionNoise)
    elif args.Algorithm == "IDEA4":
        agent = IDEA4.IDEA4(args = args,policy = PolicyNetwork,critic = QNetwork,nvp=RealNVP,replayBuff = CostReplayBuffer,exploration = OUActionNoise)

    if "teach" in args.Environment:
        teacher = TeacherController.TeacherController(param_bound,env,args)
    else:
        teacher = None

    position,velocity,angle,angle_vel,motor_thrust,body_torque = test(args,env,agent,teacher)
    plot_results(position,velocity,angle,angle_vel,motor_thrust,body_torque,args,env)

if __name__ == "__main__":
    main()