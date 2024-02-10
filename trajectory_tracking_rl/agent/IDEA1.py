import numpy as np
import torch
from trajectory_tracking_rl.pytorch_utils import hard_update
import os

class IDEA1:
    '''
    Testing Idea 1 
    '''
    def __init__(self,args,policy,critic,replayBuff,exploration):
        
        self.args = args # Argument values given by the user
        self.learning_step = 0 # counter to keep track of learning
        # Replay Buffer provided by the user
        self.replay_buffer = replayBuff(input_shape = args.input_shape,mem_size = args.mem_size,n_actions = args.n_actions,batch_size = args.batch_size)
        self.safe_replay_buffer = replayBuff(input_shape = args.input_shape,mem_size = args.mem_size,n_actions = args.n_actions,batch_size = args.batch_size)
        # Exploration Technique
        self.noiseOBJ = exploration(mean=np.zeros(args.n_actions), std_deviation=float(0.2) * np.ones(args.n_actions))
        
        self.PolicyNetwork = policy(args.input_shape,args.n_actions,args.max_action)
        self.PolicyOptimizer = torch.optim.Adam(self.PolicyNetwork.parameters(),lr=args.actor_lr)

        self.SafePolicyNetwork = policy(args.input_shape,args.n_actions,args.safe_max_action)
        self.SafePolicyOptimizer = torch.optim.Adam(self.SafePolicyNetwork.parameters(),lr=args.actor_lr)

        self.Qnetwork = critic(args.input_shape,args.n_actions)
        self.QOptimizer = torch.optim.Adam(self.Qnetwork.parameters(),lr=args.critic_lr)

        self.SafeQnetwork = critic(args.input_shape,args.n_actions)
        self.SafeQOptimizer = torch.optim.Adam(self.SafeQnetwork.parameters(),lr=args.critic_lr)
        self.safe_policy_called = 0

        self.bound_min = torch.tensor(self.args.min_action,dtype=torch.float32)
        self.bound_max = torch.tensor(self.args.max_action,dtype=torch.float32)

        hard_update(self.SafePolicyNetwork,self.PolicyNetwork)
        hard_update(self.SafeQnetwork,self.Qnetwork)

    def choose_action(self,state,stage="training"):
        
        engage_safety = self.is_state_unsafe(state)
        state = torch.tensor(state,dtype=torch.float32)
        # print(f"the state is : {state[:3]}")
        if engage_safety:
            
            if self.safe_policy_called == 3:
                action = self.PolicyNetwork(state).detach().numpy()
                self.safe_policy_called = 0

            else:
                # print("Safety engaged")
                action = self.SafePolicyNetwork(state)
                action = self.get_safe_action(state,action).detach().numpy()
            # print(f"the safety action is : {action[:3]}")
            # print(f"the state is : {state[:3]}")
                self.safe_policy_called += 1
            
        else:
            action = self.PolicyNetwork(state).detach().numpy()
            if stage == "training":
                action += self.noiseOBJ()

        action = np.clip(action,self.args.min_action,self.args.max_action)

        return action
    
    def get_safe_action(self,state,action,verbose=False):
        
        state =  torch.reshape(state,shape=(1,state.shape[0]))
        action = torch.reshape(action,shape=(1,action.shape[0]))
        pred = self.SafeQnetwork(state,action)
        if pred <= self.args.delta:
            return torch.clamp(action,self.bound_min,self.bound_max)[0]
        else:
            for i in range(self.args.Niter):
                if np.any(np.abs(action.data.numpy()) > self.args.max_action):
                    break
                action.retain_grad()
                self.SafeQnetwork.zero_grad()
                pred = self.SafeQnetwork(state,action)
                pred.backward(retain_graph=True)
                if verbose and i % 5 == 0:
                    print(f'a{i} = {action.data.numpy()}, C = {pred.item()}')
                if pred <= self.args.delta:
                    break
                Z = np.max(np.abs(action.grad.data.numpy().flatten()))
                action = action - self.args.eta * action.grad / (Z + 1e-8)
            #print(i,pred.item())
            return torch.clamp(action,self.bound_min,self.bound_max)[0]
    
    def is_state_unsafe(self,state):

        engage_safety = False
        if np.any(np.abs(state[:3]) > np.array([5.5,5.5,5.5])) or self.safe_policy_called != 0:
            engage_safety = True

        return engage_safety
    
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

        self.train_policy(state,action,reward,next_state,done)
        # self.train_safe_policy(state,action,reward,constraint,next_state,done)

        if self.safe_replay_buffer.current_mem:
            state,action,reward,constraint,next_state,done = self.safe_replay_buffer.shuffle()
            state = torch.Tensor(state)
            action  = torch.Tensor(action)
            reward = torch.Tensor(reward)
            constraint = torch.Tensor(constraint)
            next_state = torch.Tensor(next_state)
            done = torch.Tensor(done)

            # self.train_policy(state,action,reward,next_state,done)
            self.train_safe_policy(state,action,reward,constraint,next_state,done)

    def train_policy(self,state,action,reward,next_state,done):

        # critic function update
        target_next_action = self.PolicyNetwork(next_state)
        target = self.Qnetwork(next_state,target_next_action)
        y = reward + self.args.gamma*target*(1-done)
        critic_value = self.Qnetwork(state,action)
        critic_loss = torch.mean(torch.square(y - critic_value),dim=1)
        self.QOptimizer.zero_grad()
        critic_loss.mean().backward()
        self.QOptimizer.step()

        # actor update
        actions = self.PolicyNetwork(state)
        critic_value = self.Qnetwork(state,actions)
        actor_loss = -critic_value.mean()
        # actor_loss = actor_loss.mean()
        self.PolicyOptimizer.zero_grad()
        actor_loss.mean().backward()
        self.PolicyOptimizer.step()

    def train_safe_policy(self,state,action,reward,constraint,next_state,done):

        # safety critic function update
        starget_next_action = self.SafePolicyNetwork(next_state)
        starget = self.SafeQnetwork(next_state,starget_next_action)
        y = constraint + self.args.gamma*starget
        scritic_value = self.SafeQnetwork(state,action)
        scritic_loss = torch.mean(torch.square(y - scritic_value),dim=1)
        self.SafeQOptimizer.zero_grad()
        scritic_loss.mean().backward()
        self.SafeQOptimizer.step()

        #safe actor update
        sactions = self.SafePolicyNetwork(state)
        scritic_value = self.SafeQnetwork(state,sactions)
        sactor_loss = -scritic_value.mean()
        # actor_loss = actor_loss.mean()
        self.SafePolicyOptimizer.zero_grad()
        sactor_loss.mean().backward()
        self.SafePolicyOptimizer.step()

    def add(self,s,action,rwd,constraint,next_state,done):

        if self.is_state_unsafe(s):
            self.safe_replay_buffer.store(s,action,rwd,constraint,next_state,done)
        else:
            self.replay_buffer.store(s,action,rwd,constraint,next_state,done)

    def save(self,env):
        print("-------SAVING NETWORK -------")

        os.makedirs("config/saves/training_weights/"+ env + "/idea1_weights", exist_ok=True)
        torch.save(self.PolicyNetwork.state_dict(),"config/saves/training_weights/"+ env + "/idea1_weights/actorWeights.pth")
        torch.save(self.Qnetwork.state_dict(),"config/saves/training_weights/"+ env + "/idea1_weights/QWeights.pth")
        torch.save(self.SafePolicyNetwork.state_dict(),"config/saves/training_weights/"+ env + "/idea1_weights/safeactorWeights.pth")

    def load(self,env):

        self.PolicyNetwork.load_state_dict(torch.load("config/saves/training_weights/"+ env + "/idea1_weights/actorWeights.pth"))
        self.Qnetwork.load_state_dict(torch.load("config/saves/training_weights/"+ env + "/idea1_weights/QWeights.pth"))
        self.SafePolicyNetwork.load_state_dict(torch.load("config/saves/training_weights/"+ env + "/idea1_weights/safeactorWeights.pth"))
