import numpy as np
import torch
from trajectory_tracking_rl.pytorch_utils import hard_update,soft_update
import os

class TD3:

    def __init__(self,args,policy,critic,replayBuff,exploration):

        self.args = args
        self.learning_step = 0 
        self.replay_buffer = replayBuff(input_shape = args.input_shape,mem_size = args.mem_size,n_actions = args.n_actions,batch_size = args.batch_size)
        self.noiseOBJ = exploration(mean=np.zeros(args.n_actions), std_deviation=float(0.2) * np.ones(args.n_actions))
        
        self.PolicyNetwork = policy(args.input_shape,args.n_actions,args.max_action)
        self.PolicyOptimizer = torch.optim.Adam(self.PolicyNetwork.parameters(),lr=args.actor_lr)
        self.TargetPolicyNetwork = policy(args.input_shape,args.n_actions,args.max_action)

        self.Qnetwork1 = critic(args.input_shape,args.n_actions)
        self.QOptimizer1 = torch.optim.Adam(self.Qnetwork1.parameters(),lr=args.critic_lr)
        self.TargetQNetwork1 = critic(args.input_shape,args.n_actions)

        self.Qnetwork2 = critic(args.input_shape,args.n_actions)
        self.QOptimizer2 = torch.optim.Adam(self.Qnetwork2.parameters(),lr=args.critic_lr)
        self.TargetQNetwork2 = critic(args.input_shape,args.n_actions)

        hard_update(self.TargetPolicyNetwork,self.PolicyNetwork)
        hard_update(self.TargetQNetwork1,self.Qnetwork1)
        hard_update(self.TargetQNetwork2,self.Qnetwork2)

    def choose_action(self,state,stage = "training"):
        
        state = torch.Tensor(state)
        if stage == "training":
            action = self.PolicyNetwork(state).detach().numpy()
            action += self.noiseOBJ()
        else:
            action = self.TargetPolicyNetwork(state).detach().numpy()

        action = np.clip(action,self.args.min_action,self.args.max_action)

        return action

    def learn(self):
        
        self.learning_step+=1
        if self.learning_step<self.args.batch_size:
            return

        state,action,reward,next_state,done = self.replay_buffer.shuffle()
        state = torch.Tensor(state)
        action  = torch.Tensor(action)
        reward = torch.Tensor(reward)
        next_state = torch.Tensor(next_state)
        done = torch.Tensor(done)

        target_critic_action = self.TargetPolicyNetwork(next_state)
        Q1 = self.TargetQNetwork1(next_state,target_critic_action)
        Q2 = self.TargetQNetwork2(next_state,target_critic_action)
        y = reward + self.args.gamma*torch.minimum(Q1,Q2)
        critic_value1 = self.Qnetwork1(state,action)
        critic_loss1 = torch.mean(torch.square(y - critic_value1),dim=1)
        self.QOptimizer1.zero_grad()
        critic_loss1.mean().backward(retain_graph=True)
        self.QOptimizer1.step()

        critic_value2 = self.Qnetwork2(state,action)
        critic_loss2 = torch.mean(torch.square(y - critic_value2),dim=1)
        self.QOptimizer2.zero_grad()
        critic_loss2.mean().backward(retain_graph=True)
        self.QOptimizer2.step()

        actions = self.PolicyNetwork(state)
        critic_value = self.Qnetwork1(state,actions)
        actor_loss = -critic_value.mean()
        self.PolicyOptimizer.zero_grad()
        actor_loss.mean().backward()
        self.PolicyOptimizer.step()
        

        if self.learning_step%self.args.target_update == 0:                
            soft_update(self.TargetPolicyNetwork,self.PolicyNetwork,self.args.tau)
            soft_update(self.TargetQNetwork1,self.Qnetwork1,self.args.tau)
            soft_update(self.TargetQNetwork2,self.Qnetwork2,self.args.tau)

    def add(self,s,action,rwd,constraint,next_state,done):
        self.replay_buffer.store(s,action,rwd,next_state,done)
    
    def save(self,env):
        print("-------SAVING NETWORK -------")

        os.makedirs("config/saves/training_weights/"+ env + "/td3_weights", exist_ok=True)
        torch.save(self.PolicyNetwork.state_dict(),"config/saves/training_weights/"+env+"/td3_weights/actorWeights.pth")
        torch.save(self.TargetPolicyNetwork.state_dict(),"config/saves/training_weights/"+env+"/td3_weights/TargetactorWeights.pth")

    def load(self,env):

        self.PolicyNetwork.load_state_dict(torch.load("config/saves/training_weights/"+env+"/td3_weights/actorWeights.pth",map_location=torch.device('cpu')))
        self.TargetPolicyNetwork.load_state_dict(torch.load("config/saves/training_weights/"+env+"/td3_weights/TargetactorWeights.pth",map_location=torch.device('cpu')))