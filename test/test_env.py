import sys
sys.path.insert(1, "/Users/shaswatgarg/Documents/WaterlooMASc/StateSpaceUAV/")

from trajectory_tracking_rl.environment.Quadrotor.QuadrotorObsEnv import QuadrotorObsEnv
from trajectory_tracking_rl.environment.Quadrotor.QuadrotorTeachEnv import QuadrotorTeachEnv
from trajectory_tracking_rl.environment.Quadrotor.BaseQuadrotorEnv import BaseQuadrotorEnv
from trajectory_tracking_rl.rl_aerial_manipulation.environment.PyBulletEnv.UAM.BaseUAMEnv import BaseUAMEnv
from trajectory_tracking_rl.rl_aerial_manipulation.environment.PyBulletEnv.UAM.BaseUAMObsEnv import UAMObsEnv
from scripts.main import build_parse
from trajectory_tracking_rl.controllers.PID import CascadeController
import numpy as np
import argparse
import pybullet as p
from trajectory_tracking_rl.teacher import TeacherController

parser = argparse.ArgumentParser(description="RL Algorithm Variables")
parser.add_argument("Environment",nargs="?",type=str,default="UAM",help="Name of OPEN AI environment")
args = parser.parse_args("")
args.Environment = "UAM"

if "UAM" in args.Environment:
    env = BaseUAMEnv(controller=CascadeController)
elif "obs" in args.Environment:
    env = QuadrotorObsEnv(controller = CascadeController)
elif "teach" in args.Environment:
    env = QuadrotorTeachEnv(controller=CascadeController,load_obstacle=False)
else:
    env = BaseQuadrotorEnv(controller = CascadeController)

env = UAMObsEnv(controller=CascadeController)

def test_env_basics():

    pass

if __name__ == "__main__":
    param = {"n_obstacles":[0,10],"obs_centre":np.array([0,6,3])}

    # if "teach" in args.Environment:
    #     teacher = TeacherController.TeacherController(param,env)
    #     teacher.generate_env_param()

    # print(env.grid.visualize_grid())
    state = env.reset()
    print(f"desired pose is : {env.pos_des}")
    while True:

        # pass
        print(f"current_state is : {state.shape}")
        # action = env.action_space.sample()
        # action = [0,0,0,0,0,0,0,0,0,0.5,0.5]

        p.resetBasePositionAndOrientation(env.robot_id, [0,0,1],[0,0,0,1])
        
        p.setJointMotorControl2(env.robot_id, env.joint_indices["joint_shoulder"], p.POSITION_CONTROL, targetPosition=0)
        p.setJointMotorControl2(env.robot_id, env.joint_indices["joint_elbow"], p.POSITION_CONTROL, targetPosition=0)
        # print(f"action taken is {action}")
        p.stepSimulation()

        contact_info = p.getContactPoints(env.robot_id,env.robot_id)
        print(f"the contact is detected : {contact_info}")

        print(p.getLinkState(env.robot_id,0)[:2])
        print(p.getLinkState(env.robot_id,5)[:2])

        # if len(contact_info) > 0:
        #     pass
        # new_state,reward,done,_ = env.step(action)

        # state = new_state


