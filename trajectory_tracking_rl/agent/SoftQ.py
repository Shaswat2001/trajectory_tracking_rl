import numpy as np
import torch
from trajectory_tracking_rl.pytorch_utils import hard_update,soft_update
import os


class SoftQ:

    def __init__(self,args,policy,critic,replayBuff,exploration):

        self.args = args
        self.learning_step = 0 
        self.replay_buffer = replayBuff(input_shape = args.input_shape,mem_size = args.mem_size,n_actions = args.n_actions,batch_size = args.batch_size)
        self.noiseOBJ = exploration(mean=np.zeros(args.n_actions), std_deviation=float(0.2) * np.ones(args.n_actions))
        
        self.PolicyNetwork = policy(args.input_shape,args.n_actions,args.max_action)
        self.PolicyOptimizer = torch.optim.Adam(self.PolicyNetwork.parameters(),lr=args.actor_lr)
        self.TargetPolicyNetwork = policy(args.input_shape,args.n_actions,args.max_action)

        self.Qnetwork = critic(args.input_shape,args.n_actions)
        self.QOptimizer = torch.optim.Adam(self.Qnetwork.parameters(),lr=args.critic_lr)
        self.TargetQNetwork = critic(args.input_shape,args.n_actions)

        hard_update(self.TargetPolicyNetwork,self.PolicyNetwork)
        hard_update(self.TargetQNetwork,self.Qnetwork)

    def choose_action(self,state,stage="training"):
        
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
        target_q = self.TargetQNetwork(next_state,target_critic_action)
        target_v = torch.logsumexp(target_q, dim=1,keepdim=True)
        y = reward + self.args.gamma*target_v
        critic_value = self.Qnetwork(state,action)
        critic_loss = torch.mean(torch.square(y - critic_value),dim=1)
        self.QOptimizer.zero_grad()
        critic_loss.mean().backward()
        self.QOptimizer.step()

        actions = self.PolicyNetwork(state)
        critic_value = self.Qnetwork(state,actions)
        actor_loss = -critic_value.mean()
        self.PolicyOptimizer.zero_grad()
        actor_loss.mean().backward()
        self.PolicyOptimizer.step()
        

        if self.learning_step%self.args.target_update == 0:                
            soft_update(self.TargetPolicyNetwork,self.PolicyNetwork,self.args.tau)
            soft_update(self.TargetQNetwork,self.Qnetwork,self.args.tau)

    def add(self,s,action,rwd,constraint,next_state,done):
        self.replay_buffer.store(s,action,rwd,next_state,done)

    def save(self,env):
        print("-------SAVING NETWORK -------")

        os.makedirs("config/saves/training_weights/"+ env + "/softq_weights", exist_ok=True)
        torch.save(self.PolicyNetwork.state_dict(),"config/saves/training_weights/"+env+"/softq_weights/actorWeights.pth")
        torch.save(self.Qnetwork.state_dict(),"config/saves/training_weights/"+env+"/softq_weights/QWeights.pth")
        torch.save(self.TargetPolicyNetwork.state_dict(),"config/saves/training_weights/"+env+"/softq_weights/TargetactorWeights.pth")
        torch.save(self.TargetQNetwork.state_dict(),"config/saves/training_weights/"+env+"/softq_weights/TargetQWeights.pth")

    def load(self,env):

        self.PolicyNetwork.load_state_dict(torch.load("config/saves/training_weights/"+env+"/softq_weights/actorWeights.pth",map_location=torch.device('cpu')))
        self.Qnetwork.load_state_dict(torch.load("config/saves/training_weights/"+env+"/softq_weights/QWeights.pth",map_location=torch.device('cpu')))
        self.TargetPolicyNetwork.load_state_dict(torch.load("config/saves/training_weights/"+env+"/softq_weights/TargetactorWeights.pth",map_location=torch.device('cpu')))
        self.TargetQNetwork.load_state_dict(torch.load("config/saves/training_weights/"+env+"/softq_weights/TargetQWeights.pth",map_location=torch.device('cpu')))