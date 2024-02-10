import sys
sys.path.insert(1, "/Users/shaswatgarg/Documents/WaterlooMASc/StateSpaceUAV/")

from audioop import avg
import numpy as np
from Agent import DDPG,TD3,SAC,SoftQ,PSAC_CEPO,RCRL,SEditor,USL,SAAC
import torch
from trajectory_tracking_rl.pytorch_model import GaussianPolicyNetwork, PolicyNetwork,QNetwork,VNetwork,PhasicPolicyNetwork,PhasicQNetwork,ConstraintNetwork,MultiplierNetwork, RealNVP
import matplotlib.pyplot as plt
from controllers.PID import CascadeController
from Environment.Quadrotor.QuadrotorObsEnv import QuadrotorObsEnv

    
def main():

    env = QuadrotorObsEnv(controller=CascadeController)
    n_actions = env.action_space.shape[0]
    one = np.ones(1)
    action1 = np.concatenate((env.action_space.sample(),one))
    action2 = np.concatenate((env.action_space.sample(),one))
    nvp = RealNVP(n_actions+1)
    action = torch.tensor(action1,dtype=torch.float32)
    action1 = torch.tensor(action2,dtype=torch.float32)

    sfpred,pred = nvp(action,action1)
    print(sfpred)

if __name__ == "__main__":
    main()