from sre_parse import State
import numpy as np
import torch
from trajectory_tracking_rl.pytorch_utils import hard_update,soft_update
import os


class SAC:

    def __init__(self,args,policy,critic,valueNet,replayBuff,exploration,eval_rl=False):

        self.args = args
        self.learning_step = 0 
        self.eval_rl = eval_rl
        if not self.eval_rl:
            self.replay_buffer = replayBuff(input_shape = args.input_shape,mem_size = args.mem_size,n_actions = args.n_actions,batch_size = args.batch_size)
            self.noiseOBJ = exploration(mean=np.zeros(args.n_actions), std_deviation=float(0.2) * np.ones(args.n_actions))
        
        self.PolicyNetwork = policy(args.input_shape,args.n_actions,args.max_action,args.log_std_min,args.log_std_max)
        self.PolicyOptimizer = torch.optim.Adam(self.PolicyNetwork.parameters(),lr=args.actor_lr)
        self.Qnetwork = critic(args.input_shape,args.n_actions)
        self.QOptimizer = torch.optim.Adam(self.Qnetwork.parameters(),lr=args.critic_lr)
        self.VNetwork = valueNet(args.input_shape)
        self.VOptimizer = torch.optim.Adam(self.VNetwork.parameters(),lr=args.critic_lr)
        self.TargetVNetwork = valueNet(args.input_shape)

        hard_update(self.TargetVNetwork,self.VNetwork)
        if self.eval_rl:
            self.load()

    def choose_action(self,state,stage="training"):
        
        state = torch.tensor(state,dtype=torch.float32)
        obs = self.PolicyNetwork(state) 
        action = obs.pi.detach().numpy()
        # action = torch.round(torch.clip(action,self.args.min_action,self.args.max_action),decimals=2).detach().numpy()
        action = np.clip(action,self.args.min_action,self.args.max_action)

        return action

    def learn(self):
        
        self.learning_step+=1
        
        if self.learning_step<self.args.batch_size:
            return

        state,action,reward,next_state,done = self.replay_buffer.shuffle()
        state = torch.Tensor(state)
        next_state = torch.Tensor(next_state)
        action  = torch.Tensor(action)
        reward = torch.Tensor(reward)
        done = torch.Tensor(done)

        obs = self.PolicyNetwork(state)
        target = self.Qnetwork(state,obs.pi)
        y = target - self.args.temperature*obs.log_prob_pi
        v_Val = self.VNetwork(state)
        v_loss = torch.mean(torch.square(v_Val-y))
        self.VOptimizer.zero_grad()
        v_loss.mean().backward()
        self.VOptimizer.step()

        target_vVal = self.TargetVNetwork(next_state)
        y = reward + self.args.gamma*target_vVal
        critic_value = self.Qnetwork(state,action)
        critic_loss = torch.mean(torch.square(y - critic_value),dim=1)
        self.QOptimizer.zero_grad()
        critic_loss.mean().backward()
        self.QOptimizer.step()

        actions = self.PolicyNetwork(state)
        critic_value = self.Qnetwork(state,actions.pi)
        actor_loss = -(critic_value- actions.log_prob_pi*self.args.temperature).mean()
        self.PolicyOptimizer.zero_grad()
        actor_loss.mean().backward()
        self.PolicyOptimizer.step()
        
        if self.learning_step%self.args.target_update == 0:                
            soft_update(self.TargetVNetwork,self.VNetwork,self.args.tau)

    def add(self,s,action,rwd,constraint,next_state,done):
        self.replay_buffer.store(s,action,rwd,next_state,done)

    def save(self,env):
        print("-------SAVING NETWORK -------")

        os.makedirs("config/saves/training_weights/"+ env + "/sac_weights", exist_ok=True)
        torch.save(self.VNetwork.state_dict(),"config/saves/training_weights/"+env+"/sac_weights/VWeights.pth")
        torch.save(self.TargetVNetwork.state_dict(),"config/saves/training_weights/"+env+"/sac_weights/TargetVWeights.pth")
        torch.save(self.Qnetwork.state_dict(),"config/saves/training_weights/"+env+"/sac_weights/criticWeights.pth")
        torch.save(self.PolicyNetwork.state_dict(),"config/saves/training_weights/"+env+"/sac_weights/actorWeights.pth")

    def load(self,env):
        print("-------LOADING NETWORK -------")
        
        self.PolicyNetwork.load_state_dict(torch.load("config/saves/training_weights/"+env+"/sac_weights/actorWeights.pth",map_location=torch.device('cpu')))
        self.VNetwork.load_state_dict(torch.load("config/saves/training_weights/"+env+"/sac_weights/VWeights.pth",map_location=torch.device('cpu')))
        self.TargetVNetwork.load_state_dict(torch.load("config/saves/training_weights/"+env+"/sac_weights/TargetVWeights.pth",map_location=torch.device('cpu')))
        self.Qnetwork.load_state_dict(torch.load("config/saves/training_weights/"+env+"/sac_weights/criticWeights.pth",map_location=torch.device('cpu')))