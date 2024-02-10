import numpy as np
import torch
import torch.nn.functional as F
from trajectory_tracking_rl.pytorch_utils import hard_update,soft_update
import os

class USL:
    '''
    DDPG-USL Algorithm 
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

        self.costQnetwork = critic(args.input_shape,args.n_actions)
        self.costQOptimizer = torch.optim.Adam(self.costQnetwork.parameters(),lr=args.critic_lr)
        self.TargetcostQNetwork = critic(args.input_shape,args.n_actions)

        self.bound_min = torch.tensor(self.args.min_action,dtype=torch.float32)
        self.bound_max = torch.tensor(self.args.max_action,dtype=torch.float32)

        hard_update(self.TargetPolicyNetwork,self.PolicyNetwork)
        hard_update(self.TargetcostQNetwork,self.costQnetwork)
        hard_update(self.TargetQNetwork,self.Qnetwork)

    def choose_action(self,state,stage="training"):
    
        state = torch.Tensor(state)
        mu = self.PolicyNetwork(state)
        action = mu
        if stage == "training":
            action += torch.tensor(self.noiseOBJ())

        action = self.get_safe_action(state,action)[0]

        return action        
    
    def get_safe_action(self,state,action,verbose=False):
        
        state =  torch.reshape(state,shape=(1,state.shape[0]))
        action = torch.reshape(action,shape=(1,action.shape[0]))
        pred = self.TargetcostQNetwork(state,action)
        if pred <= self.args.delta:
            return torch.clamp(action,self.bound_min,self.bound_max).data.numpy()
        else:
            for i in range(self.args.Niter):
                if np.any(np.abs(action.data.numpy()) > self.args.max_action):
                    break
                action.retain_grad()
                self.TargetcostQNetwork.zero_grad()
                pred = self.TargetcostQNetwork(state,action)
                pred.backward(retain_graph=True)
                if verbose and i % 5 == 0:
                    print(f'a{i} = {action.data.numpy()}, C = {pred.item()}')
                if pred <= self.args.delta:
                    break
                Z = np.max(np.abs(action.grad.data.numpy().flatten()))
                action = action - self.args.eta * action.grad / (Z + 1e-8)
            #print(i,pred.item())
            return torch.clamp(action,self.bound_min,self.bound_max).data.numpy()

    def learn(self):
        
        self.learning_step+=1
        if self.learning_step<self.args.batch_size:
            return
        
        state,action,reward,cost,next_state,done = self.replay_buffer.shuffle()
        state = torch.Tensor(state)
        action  = torch.Tensor(action)
        reward = torch.Tensor(reward)
        cost = torch.Tensor(cost)
        next_state = torch.Tensor(next_state)
        done = torch.Tensor(done)

        target_critic_action = self.TargetPolicyNetwork(next_state)
        target_cost = self.TargetcostQNetwork(next_state, target_critic_action)
        target_cost = cost + ((1-done) * self.args.cost_discount * target_cost).detach()

        # Get current C estimate
        cost_critic = self.costQnetwork(state, action)

        # Compute critic loss
        cost_critic_loss = torch.mean(torch.square(cost_critic - target_cost),dim=1)

        # Optimize the critic
        self.costQOptimizer.zero_grad()
        cost_critic_loss.mean().backward()
        self.costQOptimizer.step()

        with torch.no_grad():
            # Select action according to policy and add clipped noise
            noise = torch.Tensor(self.noiseOBJ())
            
            next_action = (self.TargetPolicyNetwork(next_state) + noise).clamp(self.bound_min, self.bound_max)
            # Compute the target Q value
            target_critic = self.TargetQNetwork(next_state, next_action)
            target_critic = reward + (1-done) * self.args.gamma * target_critic

        # Get current Q estimates
        current_critic = self.Qnetwork(state, action)

        # Compute critic loss
        critic_loss = torch.mean(torch.square(current_critic - target_critic),dim=1)
        
        # Optimize the critic
        self.QOptimizer.zero_grad()
        critic_loss.mean().backward()
        self.QOptimizer.step()

        # Compute actor loss
        actions = self.PolicyNetwork(state)
        actor_loss = (
            - self.Qnetwork(state, actions) \
            + self.args.kappa * F.relu(self.costQnetwork(state, actions) - self.args.delta) \
            ).mean()
        
        # Optimize the actor 
        self.PolicyOptimizer.zero_grad()
        actor_loss.backward()
        self.PolicyOptimizer.step()


        if self.learning_step%self.args.target_update == 0:                
            soft_update(self.TargetcostQNetwork,self.costQnetwork,self.args.tau)
            soft_update(self.TargetQNetwork,self.Qnetwork,self.args.tau)

    def add(self,s,action,rwd,constraint,next_state,done):
        self.replay_buffer.store(s,action,rwd,constraint,next_state,done)
    
    def save(self,env):
        print("-------SAVING NETWORK -------")
        os.makedirs("config/saves/training_weights/"+ env + "/ddpg_usl_weights", exist_ok=True)
        torch.save(self.PolicyNetwork.state_dict(),"config/saves/training_weights/"+ env + "/ddpg_usl_weights/actorWeights.pth")
        torch.save(self.Qnetwork.state_dict(),"config/saves/training_weights/"+ env + "/ddpg_usl_weights/QWeights.pth")
        torch.save(self.costQnetwork.state_dict(),"config/saves/training_weights/"+ env + "/ddpg_usl_weights/costQWeights.pth")

    def load(self,env):

        self.PolicyNetwork.load_state_dict(torch.load("config/saves/training_weights/"+ env + "/ddpg_usl_weights/actorWeights.pth"))
        self.Qnetwork.load_state_dict(torch.load("config/saves/training_weights/"+ env + "/ddpg_usl_weights/QWeights.pth"))
        self.costQnetwork.load_state_dict(torch.load("config/saves/training_weights/"+ env + "/ddpg_usl_weights/costQWeights.pth"))
