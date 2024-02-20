import numpy as np
import torch
import os
from trajectory_tracking_rl.pytorch_utils import hard_update,soft_update

class DDPG:
    '''
    DDPG Algorithm 
    '''
    def __init__(self,args,policy,critic,replayBuff,exploration,vision):

        self.args = args # Argument values given by the user
        self.learning_step = 0 # counter to keep track of learning
        # Replay Buffer provided by the user
        self.vision = vision
        self.replayBuff = replayBuff
        self.exploration = exploration
        self.policy = policy
        self.critic = critic

        self.reset()

    def choose_action(self,state,stage="training"):
        
        if self.vision is not None:
            prp_state,rgb_state,depth_state = state
            state = self.vision(torch.Tensor(depth_state),torch.Tensor(rgb_state),torch.Tensor(prp_state))
        else:
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

        if self.vision is not None:
            prp_state,rgb_state,depth_state = state
            next_prp_state,next_rgb_state, next_depth_state = next_state
            state = self.vision(torch.Tensor(depth_state),torch.Tensor(rgb_state),torch.Tensor(prp_state))
            next_state = self.vision(torch.Tensor(next_depth_state),torch.Tensor(next_rgb_state),torch.Tensor(next_prp_state))
        else:
            state = torch.Tensor(state)
            next_state = torch.Tensor(next_state)
        
        action  = torch.Tensor(action)
        reward = torch.Tensor(reward)
        next_state = torch.Tensor(next_state)
        done = torch.Tensor(done)
        
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

        if self.vision is not None and self.learning_step%self.args.vision_update == 0:
            action_vision = self.PolicyNetwork(state)
            critic_value_vision = self.Qnetwork(state,action_vision)
            vision_loss = -critic_value_vision.mean()
            self.VisionOptimizer.zero_grad()
            vision_loss.mean().backward()
            self.VisionOptimizer.step()

        if self.learning_step%self.args.target_update == 0:                
            soft_update(self.TargetPolicyNetwork,self.PolicyNetwork,self.args.tau)
            soft_update(self.TargetQNetwork,self.Qnetwork,self.args.tau)

    def add(self,s,action,rwd,constraint,next_state,done):
        self.replay_buffer.store(s,action,rwd,next_state,done)

    def reset(self):

        self.replay_buffer = self.replayBuff(input_shape = self.args.state_size,mem_size = self.args.mem_size,n_actions = self.args.n_actions,batch_size = self.args.batch_size)
        # Exploration Technique
        self.noiseOBJ = self.exploration(mean=np.zeros(self.args.n_actions), std_deviation=float(0.08) * np.ones(self.args.n_actions))
        
        self.PolicyNetwork = self.policy(self.args.input_shape,self.args.n_actions,self.args.max_action)
        self.PolicyOptimizer = torch.optim.Adam(self.PolicyNetwork.parameters(),lr=self.args.actor_lr)
        self.TargetPolicyNetwork = self.policy(self.args.input_shape,self.args.n_actions,self.args.max_action)

        self.Qnetwork = self.critic(self.args.input_shape,self.args.n_actions)
        self.QOptimizer = torch.optim.Adam(self.Qnetwork.parameters(),lr=self.args.critic_lr)
        self.TargetQNetwork = self.critic(self.args.input_shape,self.args.n_actions)

        if self.vision is not None:
            self.VisionOptimizer = torch.optim.Adam(self.vision.parameters(),lr=self.args.critic_lr)

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

        self.PolicyNetwork.load_state_dict(torch.load("config/saves/training_weights/"+ env + "/ddpg_weights/actorWeights.pth",map_location=torch.device('cpu')))
        self.Qnetwork.load_state_dict(torch.load("config/saves/training_weights/"+ env + "/ddpg_weights/QWeights.pth",map_location=torch.device('cpu')))
        self.TargetPolicyNetwork.load_state_dict(torch.load("config/saves/training_weights/"+ env + "/ddpg_weights/TargetactorWeights.pth",map_location=torch.device('cpu')))
        self.TargetQNetwork.load_state_dict(torch.load("config/saves/training_weights/"+ env + "/ddpg_weights/TargetQWeights.pth",map_location=torch.device('cpu')))