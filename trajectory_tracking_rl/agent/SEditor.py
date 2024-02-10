import numpy as np
import torch
import os

class SEditor:
    '''
    SEditor Algorithm 
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

        self.DeltaPolicyNetwork = policy(args.input_shape,args.n_actions,args.max_action)
        self.DeltaPolicyOptimizer = torch.optim.Adam(self.DeltaPolicyNetwork.parameters(),lr=args.actor_lr)

        self.Qnetwork = critic(args.input_shape,args.n_actions)
        self.QOptimizer = torch.optim.Adam(self.Qnetwork.parameters(),lr=args.critic_lr)
        
        self.CostQnetwork = critic(args.input_shape,args.n_actions)
        self.CostQOptimizer = torch.optim.Adam(self.CostQnetwork.parameters(),lr=args.critic_lr)

        self.multiplier = torch.ones(1, requires_grad=True)
        self.mult_optimizer = torch.optim.Adam([self.multiplier], lr=args.mult_lr)

    def choose_action(self,state,stage="training"):
        
        state = torch.Tensor(state)
        unsafe_action = self.PolicyNetwork(state) 
        delta_action = self.DeltaPolicyNetwork(state)
        action = unsafe_action +2*delta_action

        if stage == "learning":
            min_bound = torch.tensor(self.args.min_action,dtype=torch.float32)
            max_bound = torch.tensor(self.args.max_action,dtype=torch.float32)
            action = torch.clip(action,min_bound,max_bound)
        else:
            action = action.detach().numpy()
            action = np.clip(action,self.args.min_action,self.args.max_action)

        return action

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

        # Critic function update
        target_next_action = self.choose_action(next_state,"learning")
        target = self.Qnetwork(next_state,target_next_action)
        y = reward + self.args.gamma*target
        critic_value = self.Qnetwork(state,action)
        critic_loss = torch.mean(torch.square(y - critic_value),dim=1)
        self.QOptimizer.zero_grad()
        critic_loss.mean().backward()
        self.QOptimizer.step()

        # Cost critic function update
        target_next_action = self.choose_action(next_state,"learning")
        cost_target = self.CostQnetwork(next_state,target_next_action)
        y = cost + self.args.gamma*cost_target
        critic_value = self.CostQnetwork(state,action)
        cost_critic_loss = torch.mean(torch.square(y - critic_value),dim=1)
        self.CostQOptimizer.zero_grad()
        cost_critic_loss.mean().backward()
        self.CostQOptimizer.step()

        actions = self.choose_action(state,"learning")
        critic_value = self.Qnetwork(state,actions)
        actor_loss = -critic_value.mean()
        self.PolicyOptimizer.zero_grad()
        actor_loss.mean().backward()
        self.PolicyOptimizer.step()

        actions = self.PolicyNetwork(state)
        safe_action = self.choose_action(state,"learning")
        hinge_target = self.Qnetwork(state,actions) - self.Qnetwork(state,safe_action)
        hinge_zero = torch.zeros_like(hinge_target)
        hinge_loss = torch.max(hinge_zero,hinge_target).mean()
        safe_critic_value = self.CostQnetwork(state,safe_action)
        safe_actor_loss = -self.multiplier*safe_critic_value + hinge_loss
        safe_actor_loss = safe_actor_loss.mean()
        self.DeltaPolicyOptimizer.zero_grad()
        safe_actor_loss.mean().backward()
        self.DeltaPolicyOptimizer.step()

        mult_loss = self.multiplier*(torch.sum(cost)/self.args.mem_size + self.args.cost_violation)
        # mult_loss = mult_loss.mean()
        self.mult_optimizer.zero_grad()
        mult_loss.mean().backward()
        self.mult_optimizer.step()

    def add(self,s,action,rwd,constraint,next_state,done):
        self.replay_buffer.store(s,action,rwd,constraint,next_state,done)
            
    def save(self,env):
        print("-------SAVING NETWORK -------")

        os.makedirs("config/saves/training_weights/"+ env + "/s_editor_weights", exist_ok=True)
        torch.save(self.PolicyNetwork.state_dict(),"config/saves/training_weights/"+ env + "/s_editor_weights/actorWeights.pth")
        torch.save(self.Qnetwork.state_dict(),"config/saves/training_weights/"+ env + "/s_editor_weights/QWeights.pth")
        # torch.save(self.TargetPolicyNetwork.state_dict(),"config/saves/training_weights/"+ env + "/ddpg_weights/TargetactorWeights.pth")
        # torch.save(self.TargetQNetwork.state_dict(),"config/saves/training_weights/"+ env + "/ddpg_weights/TargetQWeights.pth")

    def load(self,env):

        self.PolicyNetwork.load_state_dict(torch.load("config/saves/training_weights/"+ env + "/s_editor_weights/actorWeights.pth"))
        self.Qnetwork.load_state_dict(torch.load("config/saves/training_weights/"+ env + "/s_editor_weights/QWeights.pth"))
        # self.TargetPolicyNetwork.load_state_dict(torch.load("config/saves/training_weights/"+ env + "/ddpg_weights/TargetactorWeights.pth"))
        # self.TargetQNetwork.load_state_dict(torch.load("config/saves/training_weights/"+ env + "/ddpg_weights/TargetQWeights.pth"))