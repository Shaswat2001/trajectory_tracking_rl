import numpy as np
import torch
import os
from trajectory_tracking_rl.pytorch_utils import hard_update,soft_update

class DDPG:
    '''
    DDPG Algorithm 
    '''
    def __init__(self,args,policy,critic,replayBuff,exploration):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.args = args # Argument values given by the user
        self.learning_step = 0 # counter to keep track of learning
        # Replay Buffer provided by the user
        self.replayBuff = replayBuff
        self.policy = policy
        self.critic = critic
        self.exploration = exploration

        self.reset()

    def choose_action(self,state,stage="training"):
        
        state = torch.Tensor(state).to(self.device)
        if stage == "training":
            action = self.PolicyNetwork(state).to("cpu").detach().numpy()
            action += self.noiseOBJ()
        else:
            action = self.TargetPolicyNetwork(state).to("cpu").detach().numpy()

        action = np.clip(action,self.args.min_action,self.args.max_action)

        return action

    def learn(self):
        
        self.learning_step+=1
        if self.learning_step<self.args.batch_size:
            return
        state,action,reward,next_state,done = self.replay_buffer.shuffle()

        state = torch.Tensor(state).to(self.device)
        next_state = torch.Tensor(next_state).to(self.device)
        
        action  = torch.Tensor(action).to(self.device)
        reward = torch.Tensor(reward).to(self.device)
        next_state = torch.Tensor(next_state).to(self.device)
        done = torch.Tensor(done).to(self.device)
        
        target_critic_action = self.TargetPolicyNetwork(next_state)
        target = self.TargetQNetwork(next_state,target_critic_action)
        y = reward + self.args.gamma*target*(1-done)
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

    def reset(self):

        self.replay_buffer = self.replayBuff(input_shape = self.args.state_size,mem_size = self.args.mem_size,n_actions = self.args.n_actions,batch_size = self.args.batch_size)
        # Exploration Technique
        self.noiseOBJ = self.exploration(mean=np.zeros(self.args.n_actions), std_deviation=float(0.1) * np.ones(self.args.n_actions))
        
        self.PolicyNetwork = self.policy(self.args.input_shape,self.args.n_actions,self.args.max_action).to(self.device)
        self.PolicyOptimizer = torch.optim.Adam(self.PolicyNetwork.parameters(),lr=self.args.actor_lr)
        self.TargetPolicyNetwork = self.policy(self.args.input_shape,self.args.n_actions,self.args.max_action).to(self.device)

        self.Qnetwork = self.critic(self.args.input_shape,self.args.n_actions).to(self.device)
        self.QOptimizer = torch.optim.Adam(self.Qnetwork.parameters(),lr=self.args.critic_lr)
        self.TargetQNetwork = self.critic(self.args.input_shape,self.args.n_actions).to(self.device)

        hard_update(self.TargetPolicyNetwork,self.PolicyNetwork)
        hard_update(self.TargetQNetwork,self.Qnetwork)
    
    def save(self,env):
        print("-------SAVING NETWORK -------")

        os.makedirs("config/saves/training_weights/"+ env + "/ddpg_weights", exist_ok=True)
        torch.save(self.PolicyNetwork.state_dict(),"config/saves/training_weights/"+ env + "/ddpg_weights/actorWeights.pth")
        torch.save(self.Qnetwork.state_dict(),"config/saves/training_weights/"+ env + "/ddpg_weights/QWeights.pth")
        torch.save(self.TargetPolicyNetwork.state_dict(),"config/saves/training_weights/"+ env + "/ddpg_weights/TargetactorWeights.pth")
        torch.save(self.TargetQNetwork.state_dict(),"config/saves/training_weights/"+ env + "/ddpg_weights/TargetQWeights.pth")

    def load(self,env):

        print("-------LOADING NETWORK -------")

        self.PolicyNetwork.load_state_dict(torch.load("config/saves/training_weights/"+ env + "/ddpg_weights/actorWeights.pth"))
        self.Qnetwork.load_state_dict(torch.load("config/saves/training_weights/"+ env + "/ddpg_weights/QWeights.pth"))
        self.TargetPolicyNetwork.load_state_dict(torch.load("config/saves/training_weights/"+ env + "/ddpg_weights/TargetactorWeights.pth"))
        self.TargetQNetwork.load_state_dict(torch.load("config/saves/training_weights/"+ env + "/ddpg_weights/TargetQWeights.pth"))