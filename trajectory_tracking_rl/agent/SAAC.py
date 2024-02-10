import numpy as np
import torch
from torch.nn.functional import kl_div
from trajectory_tracking_rl.pytorch_utils import hard_update,soft_update
import os


class SAAC:
    '''
    SAAC Algorithm 
    '''
    def __init__(self,args,policy,critic,valueNet, replayBuff,exploration):

        self.args = args # Argument values given by the user
        self.learning_step = 0 # counter to keep track of learning
        # Replay Buffer provided by the user
        self.replay_buffer = replayBuff(input_shape = args.input_shape,mem_size = args.mem_size,n_actions = args.n_actions,batch_size = args.batch_size)
        # Exploration Technique
        self.noiseOBJ = exploration(mean=np.zeros(args.n_actions), std_deviation=float(0.2) * np.ones(args.n_actions))
        
        self.PolicyNetwork = policy(args.input_shape,args.n_actions,args.max_action,args.log_std_min,args.log_std_max)
        self.PolicyOptimizer = torch.optim.Adam(self.PolicyNetwork.parameters(),lr=args.actor_lr)
        
        self.AdversaryPolicyNetwork = policy(args.input_shape,args.n_actions,args.max_action,args.log_std_min,args.log_std_max)
        self.AdversaryPolicyOptimizer = torch.optim.Adam(self.AdversaryPolicyNetwork.parameters(),lr=args.actor_lr)

        self.Qnetwork = critic(args.input_shape,args.n_actions)
        self.QOptimizer = torch.optim.Adam(self.Qnetwork.parameters(),lr=args.critic_lr)
        self.TargetQNetwork = critic(args.input_shape,args.n_actions)

        self.AdversaryQnetwork = critic(args.input_shape,args.n_actions)
        self.AdversaryQOptimizer = torch.optim.Adam(self.AdversaryQnetwork.parameters(),lr=args.critic_lr)
        self.AdversaryTargetQNetwork = critic(args.input_shape,args.n_actions)

        self.VNetwork = valueNet(args.input_shape)
        self.VOptimizer = torch.optim.Adam(self.VNetwork.parameters(),lr=args.critic_lr)
        self.TargetVNetwork = valueNet(args.input_shape)

        # self.alpha = torch.full((1,),0.2, requires_grad=True)
        # self.alpha_optimizer = torch.optim.Adam([self.alpha], lr=args.mult_lr)

        # self.beta = torch.full((1,),0.2, requires_grad=True)
        # self.beta_optimizer = torch.optim.Adam([self.beta], lr=args.mult_lr)

        self.alpha = args.temperature
        self.beta = args.temperature
        hard_update(self.AdversaryTargetQNetwork,self.AdversaryQnetwork)
        hard_update(self.TargetQNetwork,self.Qnetwork)
        hard_update(self.TargetVNetwork,self.VNetwork)

    def choose_action(self,state,stage="training"):
        
        state = torch.tensor(state,dtype=torch.float32)
        obs = self.PolicyNetwork(state) 
        action = obs.pi.detach().numpy()
        if stage == "training":
            action += self.noiseOBJ()
        # action = torch.round(torch.clip(action,self.args.min_action,self.args.max_action),decimals=2).detach().numpy()
        action = np.clip(action,self.args.min_action,self.args.max_action)

        return action

    def learn(self):
        
        self.learning_step+=1
        if self.learning_step<self.args.batch_size:
            return
        state,action,reward,safety_rwd,next_state,done = self.replay_buffer.shuffle()
        state = torch.Tensor(state)
        action  = torch.Tensor(action)
        reward = torch.Tensor(reward)
        safety_rwd = torch.Tensor(safety_rwd)
        next_state = torch.Tensor(next_state)
        done = torch.Tensor(done)

        oldPolicy = self.PolicyNetwork
        AdversaryPolicy = self.AdversaryPolicyNetwork
        
        # Adversary critic function
        criticAdversary = self.AdversaryQnetwork(state,action)
        adversary_action = self.AdversaryPolicyNetwork(state)
        target = self.AdversaryTargetQNetwork(state,adversary_action.pi)
        y = safety_rwd + self.args.gamma*(target - self.alpha*adversary_action.log_prob_pi)
        adversary_critic_loss = torch.mean(torch.square(y - criticAdversary),dim=1)
        self.AdversaryQOptimizer.zero_grad()
        adversary_critic_loss.mean().backward()
        self.AdversaryQOptimizer.step()

        # Adversary policy function
        adversary_action = self.AdversaryPolicyNetwork(state)
        criticAdversary = self.AdversaryQnetwork(state,adversary_action.pi)
        adversary_actor_loss = self.alpha*adversary_action.log_prob_pi - criticAdversary
        self.AdversaryPolicyOptimizer.zero_grad()
        adversary_actor_loss.mean().backward()
        self.AdversaryPolicyOptimizer.zero_grad()

        # # Beta parameter update
        # policy_dist = self.PolicyNetwork(state).pi
        # adv_policy_dist = self.AdversaryPolicyNetwork(state).pi
        # kl_policy = kl_div(policy_dist,adv_policy_dist,reduction="mean")
        # beta_loss = torch.log(self.beta + 1e-8)*(kl_policy - self.args.target_entropy_beta)
        # self.beta_optimizer.zero_grad()
        # beta_loss.mean().backward()
        # self.beta_optimizer.step()

        # value function
        obs = self.PolicyNetwork(state)
        target = self.Qnetwork(state,obs.pi)
        y = target - self.alpha*obs.log_prob_pi
        v_Val = self.VNetwork(state)
        v_loss = torch.mean(torch.square(v_Val-y))
        self.VOptimizer.zero_grad()
        v_loss.mean().backward()
        self.VOptimizer.step()

        # critic function
        target_vVal = self.TargetVNetwork(next_state)
        y = reward + self.args.gamma*target_vVal
        critic_value = self.Qnetwork(state,action)
        critic_loss = torch.mean(torch.square(y - critic_value),dim=1)
        self.QOptimizer.zero_grad()
        critic_loss.mean().backward()
        self.QOptimizer.step()

        # policy function
        action_pred = self.PolicyNetwork(state)
        critic_value = self.Qnetwork(state,action_pred.pi)
        old_action = oldPolicy(state).log_prob_pi
        old_action_adversary = AdversaryPolicy(state).log_prob_pi
        actor_loss = self.alpha*action_pred.log_prob_pi - critic_value - self.beta*(old_action - old_action_adversary)
        self.PolicyOptimizer.zero_grad()
        actor_loss.mean().backward()
        self.PolicyOptimizer.step()

        # # alpha parameter update
        # alpha_loss = torch.log(self.alpha + 1e-8)*(-self.PolicyNetwork(state).log_prob_pi - self.args.target_entropy)
        # self.alpha_optimizer.zero_grad()
        # alpha_loss.mean().backward()
        # self.alpha_optimizer.step()


        if self.learning_step%self.args.target_update == 0:                
            soft_update(self.AdversaryTargetQNetwork,self.AdversaryQnetwork,self.args.tau)
            soft_update(self.TargetQNetwork,self.Qnetwork,self.args.tau)
            soft_update(self.TargetVNetwork,self.VNetwork,self.args.tau)
    
    def add(self,s,action,rwd,constraint,next_state,done):
        self.replay_buffer.store(s,action,rwd,constraint,next_state,done)
        
    def save(self,env):
        print("-------SAVING NETWORK -------")

        os.makedirs("config/saves/training_weights/"+ env + "/saac_weights", exist_ok=True)
        torch.save(self.PolicyNetwork.state_dict(),"config/saves/training_weights/"+ env + "/saac_weights/actorWeights.pth")
        torch.save(self.Qnetwork.state_dict(),"config/saves/training_weights/"+ env + "/saac_weights/QWeights.pth")

    def load(self,env):

        self.PolicyNetwork.load_state_dict(torch.load("config/saves/training_weights/"+ env + "/saac_weights/actorWeights.pth"))
        self.Qnetwork.load_state_dict(torch.load("config/saves/training_weights/"+ env + "/saac_weights/QWeights.pth"))