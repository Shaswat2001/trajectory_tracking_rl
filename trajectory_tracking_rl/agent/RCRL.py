import numpy as np
import torch
import os

class RCRL:
    '''
    RCRL Algorithm 
    '''
    def __init__(self,args,policy,critic,multiplier,replayBuff,exploration):
        
        self.args = args # Argument values given by the user
        self.learning_step = 0 # counter to keep track of learning
        # Replay Buffer provided by the user
        self.replay_buffer = replayBuff(input_shape = args.input_shape,mem_size = args.mem_size,n_actions = args.n_actions,batch_size = args.batch_size)
        # Exploration Technique
        self.noiseOBJ = exploration(mean=np.zeros(args.n_actions), std_deviation=float(0.2) * np.ones(args.n_actions))
        
        self.PolicyNetwork = policy(args.input_shape,args.n_actions,args.max_action)
        self.PolicyOptimizer = torch.optim.Adam(self.PolicyNetwork.parameters(),lr=args.actor_lr)

        self.Qnetwork = critic(args.input_shape,args.n_actions)
        self.QOptimizer = torch.optim.Adam(self.Qnetwork.parameters(),lr=args.critic_lr)

        self.safetyQnetwork = critic(args.input_shape,args.n_actions)
        self.safetyQOptimizer = torch.optim.Adam(self.safetyQnetwork.parameters(),lr=args.critic_lr)

        self.multiplier = multiplier(args.input_shape)
        self.mOptimizer = torch.optim.Adam(self.multiplier.parameters(),lr=args.critic_lr)

    def choose_action(self,state,stage="training"):
        
        state = torch.tensor(state,dtype=torch.float32)
        obs = self.PolicyNetwork(state) 
        action = obs.detach().numpy()
        if stage == "training":
            action += self.noiseOBJ()
        # action = torch.round(torch.clip(action,self.args.min_action,self.args.max_action),decimals=2).detach().numpy()
        action = np.clip(action,self.args.min_action,self.args.max_action)

        return action

    def learn(self):
        
        self.learning_step+=1
        if self.learning_step<self.args.batch_size:
            return
        state,action,reward,constraint,next_state,done = self.replay_buffer.shuffle()
        state = torch.Tensor(state)
        action  = torch.Tensor(action)
        reward = torch.Tensor(reward)
        constraint = torch.Tensor(constraint)
        next_state = torch.Tensor(next_state)
        done = torch.Tensor(done)

        # critic function update
        target_next_action = self.PolicyNetwork(next_state)
        target = self.Qnetwork(next_state,target_next_action)
        y = reward + self.args.gamma*target
        critic_value = self.Qnetwork(state,action)
        critic_loss = torch.mean(torch.square(y - critic_value),dim=1)
        self.QOptimizer.zero_grad()
        critic_loss.mean().backward()
        self.QOptimizer.step()

        # safety critic function update
        starget_next_action = self.PolicyNetwork(next_state)
        starget = self.safetyQnetwork(next_state,starget_next_action)
        y = (1-self.args.gamma)*constraint + self.args.gamma*torch.max(starget,constraint)
        scritic_value = self.safetyQnetwork(state,action)
        scritic_loss = torch.mean(torch.square(y - scritic_value),dim=1)
        self.safetyQOptimizer.zero_grad()
        scritic_loss.mean().backward()
        self.safetyQOptimizer.step()

        # actor update
        actions = self.PolicyNetwork(state)
        critic_value = self.Qnetwork(state,actions)
        safety_critic_value = self.safetyQnetwork(state,actions)
        multiplier_val = self.multiplier(state)
        actor_loss = -critic_value + multiplier_val*safety_critic_value
        actor_loss = actor_loss.mean()
        self.PolicyOptimizer.zero_grad()
        actor_loss.mean().backward(retain_graph=True)
        self.PolicyOptimizer.step()

        #multiplier update
        action_m = self.PolicyNetwork(state)
        critic_value_m = self.Qnetwork(state,action_m)
        safety_critic_value_m = self.safetyQnetwork(state,action_m)
        mult_loss = critic_value_m - multiplier_val*safety_critic_value_m
        mult_loss = torch.mean(mult_loss)
        self.mOptimizer.zero_grad()
        mult_loss.mean().backward()
        self.mOptimizer.step()

    def add(self,s,action,rwd,constraint,next_state,done):
        self.replay_buffer.store(s,action,rwd,constraint,next_state,done)

    def save(self,env):
        print("-------SAVING NETWORK -------")

        os.makedirs("config/saves/training_weights/"+ env + "/rcrl_weights", exist_ok=True)
        torch.save(self.PolicyNetwork.state_dict(),"config/saves/training_weights/"+ env + "/rcrl_weights/actorWeights.pth")
        torch.save(self.Qnetwork.state_dict(),"config/saves/training_weights/"+ env + "/rcrl_weights/QWeights.pth")

    def load(self,env):

        self.PolicyNetwork.load_state_dict(torch.load("config/saves/training_weights/"+ env + "/rcrl_weights/actorWeights.pth"))
        self.Qnetwork.load_state_dict(torch.load("config/saves/training_weights/"+ env + "/rcrl_weights/QWeights.pth"))
