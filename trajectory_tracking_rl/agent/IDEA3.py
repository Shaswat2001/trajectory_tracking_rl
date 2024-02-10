import numpy as np
import torch
from trajectory_tracking_rl.pytorch_utils import hard_update,soft_update
import os


class IDEA3:
    '''
    IDEA3 Algorithm 
    '''
    def __init__(self,args,policy,critic,replayBuff,exploration):

        self.args = args # Argument values given by the user
        self.learning_step = 0 # counter to keep track of learning
        # Replay Buffer provided by the user
        self.replay_buffer = replayBuff(input_shape = args.input_shape,mem_size = args.mem_size,n_actions = args.n_actions,batch_size = args.batch_size)
        # Exploration Technique
        self.noiseOBJ = exploration(mean=np.zeros(args.n_actions), std_deviation=float(0.2) * np.ones(args.n_actions))
        
        self.PolicyNetwork = policy(args.input_shape,args.n_actions,args.max_action)
        self.PolicyOptimizer = torch.optim.Adam(self.PolicyNetwork.parameters(),lr=args.actor_lr)
        self.TargetPolicyNetwork = policy(args.input_shape,args.n_actions,args.max_action)

        self.Qnetwork = critic(args.input_shape,args.n_actions)
        self.QOptimizer = torch.optim.Adam(self.Qnetwork.parameters(),lr=args.critic_lr)
        self.TargetQNetwork = critic(args.input_shape,args.n_actions)

        self.SafeQnetwork = critic(args.input_shape,args.n_actions)
        self.SafeQOptimizer = torch.optim.Adam(self.SafeQnetwork.parameters(),lr=args.critic_lr)
        self.TargetSafeQNetwork = critic(args.input_shape,args.n_actions)

        self.bound_min = torch.tensor(self.args.min_action,dtype=torch.float32)
        self.bound_max = torch.tensor(self.args.max_action,dtype=torch.float32)
        self.safe_policy_called = 0

        hard_update(self.TargetPolicyNetwork,self.PolicyNetwork)
        hard_update(self.TargetQNetwork,self.Qnetwork)
        hard_update(self.TargetSafeQNetwork,self.SafeQnetwork)

    def choose_action(self,state,stage="training"):
        
        engage_safety = self.is_state_unsafe(state)
        state = torch.tensor(state,dtype=torch.float32)

        action,_ = self.PolicyNetwork(state)
        # print(f"the state is : {state[:3]}")
        if engage_safety:
            
            if self.safe_policy_called == 3:
                self.safe_policy_called = 0

            else:
                action = self.get_safe_action(state,action)
                self.safe_policy_called += 1
            
        action = action.detach().numpy()

        if stage == "training":
            action += self.noiseOBJ()

        action = np.clip(action,self.args.min_action,self.args.max_action)

        return action
    
    def get_safe_action(self,state,action,verbose=False):
        
        state =  torch.reshape(state,shape=(1,state.shape[0]))
        action = torch.reshape(action,shape=(1,action.shape[0]))
        pred = self.TargetSafeQNetwork(state,action)
        if pred <= self.args.delta:
            return torch.clamp(action,self.bound_min,self.bound_max)[0]
        else:
            for i in range(self.args.Niter):
                if np.any(np.abs(action.data.numpy()) > self.args.max_action):
                    break
                action.retain_grad()
                self.TargetSafeQNetwork.zero_grad()
                pred = self.TargetSafeQNetwork(state,action)
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
        
        target_next_action,_ = self.TargetPolicyNetwork(next_state)
        target = self.TargetQNetwork(next_state,target_next_action)
        y = reward + self.args.gamma*target*(1-done)
        critic_value = self.Qnetwork(state,action)
        critic_loss = torch.mean(torch.square(y - critic_value),dim=1)
        self.QOptimizer.zero_grad()
        critic_loss.mean().backward()
        self.QOptimizer.step()

        starget_next_action,_ = self.TargetPolicyNetwork(next_state)
        starget = self.TargetSafeQNetwork(next_state,starget_next_action)
        y = constraint + self.args.gamma*starget
        scritic_value = self.SafeQnetwork(state,action)
        scritic_loss = torch.mean(torch.square(y - scritic_value),dim=1)
        self.SafeQOptimizer.zero_grad()
        scritic_loss.mean().backward()
        self.SafeQOptimizer.step()

        actions,_ = self.PolicyNetwork(state)
        critic_value = self.Qnetwork(state,actions)
        actor_loss = - critic_value.mean()
        self.PolicyOptimizer.zero_grad()
        actor_loss.mean().backward()
        self.PolicyOptimizer.step()

        if self.learning_step%self.args.target_update == 0:                
            soft_update(self.TargetPolicyNetwork,self.PolicyNetwork,self.args.tau)
            soft_update(self.TargetQNetwork,self.Qnetwork,self.args.tau)
            soft_update(self.TargetSafeQNetwork,self.SafeQnetwork,self.args.tau)

        if self.learning_step%self.args.aux_step==0:

            actions,safeQTarget = self.PolicyNetwork(state)
            y = self.SafeQnetwork(state,actions)
            aux_loss = torch.mean(torch.square(y - safeQTarget),dim=1)
            self.PolicyOptimizer.zero_grad()
            aux_loss.mean().backward()
            self.PolicyOptimizer.step()

    def add(self,s,action,rwd,constraint,next_state,done):
        self.replay_buffer.store(s,action,rwd,constraint,next_state,done)
    
    def save(self,env):
        print("-------SAVING NETWORK -------")

        os.makedirs("config/saves/training_weights/"+ env + "/idea3_weights", exist_ok=True)
        torch.save(self.PolicyNetwork.state_dict(),"config/saves/training_weights/"+ env + "/idea3_weights/actorWeights.pth")
        torch.save(self.Qnetwork.state_dict(),"config/saves/training_weights/"+ env + "/idea3_weights/QWeights.pth")

    def load(self,env):

        self.PolicyNetwork.load_state_dict(torch.load("config/saves/training_weights/"+ env + "/idea3_weights/actorWeights.pth"))
        self.Qnetwork.load_state_dict(torch.load("config/saves/training_weights/"+ env + "/idea3_weights/QWeights.pth"))
