import random
import torch
import numpy as np

class ReplayBuffer:
    def __init__(self, capacity, device):
        self.capacity = capacity # capacity of the buffer
        self.data = []
        self.index = 0 # index of the next cell to be filled
        self.device = device
    def append(self, app):
        if len(self.data) < self.capacity:
            self.data.append(None)
        self.data[self.index] = app
        self.index = (self.index + 1) % self.capacity
    def sample(self, batch_size):
        batch = random.sample(self.data, batch_size)
        #return list(map(lambda x:torch.Tensor(np.array(x)).to(self.device), list(zip(*batch))))
        return batch
    def __len__(self):
        return len(self.data)
import torch

def greedy_action(network, state):
    device = "cuda" if next(network.parameters()).is_cuda else "cpu"
    with torch.no_grad():
        Q = network(torch.Tensor(state).unsqueeze(0).to(device))
        return torch.argmax(Q).item()
import numpy as np
import torch
import torch.nn as nn
from copy import deepcopy


class dqn_agent:
    def __init__(self, config, model):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.nb_actions = config['nb_actions']
        self.gamma = config['gamma'] if 'gamma' in config.keys() else 0.95
        self.batch_size = config['batch_size'] if 'batch_size' in config.keys() else 100
        buffer_size = config['buffer_size'] if 'buffer_size' in config.keys() else int(1e5)
        self.memory = ReplayBuffer(buffer_size,device)        
        self.epsilon_max = config['epsilon_max'] if 'epsilon_max' in config.keys() else 1.
        self.epsilon_min = config['epsilon_min'] if 'epsilon_min' in config.keys() else 0.01
        self.epsilon_stop = config['epsilon_decay_period'] if 'epsilon_decay_period' in config.keys() else 1000
        self.epsilon_delay = config['epsilon_delay_decay'] if 'epsilon_delay_decay' in config.keys() else 20
        self.epsilon_step = (self.epsilon_max-self.epsilon_min)/self.epsilon_stop
        self.model = model 
        self.target_model = deepcopy(self.model).to(device)
        self.criterion = config['criterion'] if 'criterion' in config.keys() else torch.nn.MSELoss()
        lr = config['learning_rate'] if 'learning_rate' in config.keys() else 0.001
        self.optimizer = config['optimizer'] if 'optimizer' in config.keys() else torch.optim.Adam(self.model.parameters(), lr=lr)
        self.nb_gradient_steps = config['gradient_steps'] if 'gradient_steps' in config.keys() else 1
        self.update_target_strategy = config['update_target_strategy'] if 'update_target_strategy' in config.keys() else 'replace'
        self.update_target_freq = config['update_target_freq'] if 'update_target_freq' in config.keys() else 20
        self.update_target_tau = config['update_target_tau'] if 'update_target_tau' in config.keys() else 0.005
        self.monitoring_nb_trials = config['monitoring_nb_trials'] if 'monitoring_nb_trials' in config.keys() else 0
        self.n_step = config['n_step'] if 'n_step' in config.keys() else 3

    def MC_eval(self, env, nb_trials):   # NEW NEW NEW
        MC_total_reward = []
        MC_discounted_reward = []
        for _ in range(nb_trials):
            x,_ = env.reset()
            done = False
            trunc = False
            total_reward = 0
            discounted_reward = 0
            step = 0
            while not (done or trunc):
                a = greedy_action(self.model, x)
                y,r,done,trunc,_ = env.step(a)
                x = y
                total_reward += r
                discounted_reward += self.gamma**step * r
                step += 1
            MC_total_reward.append(total_reward)
            MC_discounted_reward.append(discounted_reward)
        return np.mean(MC_discounted_reward), np.mean(MC_total_reward)
    
    def V_initial_state(self, env, nb_trials):   # NEW NEW NEW
        with torch.no_grad():
            for _ in range(nb_trials):
                val = []
                x,_ = env.reset()
                val.append(self.model(torch.Tensor(x).unsqueeze(0).to(device)).max().item())
        return np.mean(val)
    
    def gradient_step(self):
        if len(self.memory) > self.batch_size:
            nstep_traj = self.memory.sample(self.batch_size) #bsz nstep 5 
            #X ,A,Y,D = nstep_traj[:,0,0],nstep_traj[:,0,1],nstep_traj[:,0,3],nstep_traj[:,0,4]
            #X, A, R, Y, D = self.memory.sample(self.batch_size)
            #QYmax = self.target_model(Y).max(1)[0].detach()
            #update = torch.addcmul(R, 1-D, QYmax, value=self.gamma)
            Y =torch.Tensor(np.array([x[-1][3] for x in nstep_traj])).to(device)
            QYmax = self.target_model(Y).max(1)[0].detach().to(device)
            update = torch.zeros_like(QYmax).to(device)
            for s in range(self.n_step) :
                D = torch.Tensor([x[s][-1] for x in nstep_traj]).to(device)
                R = torch.Tensor([x[s][2] for x in nstep_traj]).to(device)
                update = torch.addcmul(update, 1-D, R, value=self.gamma**s)
            update = update + QYmax *self.gamma**self.n_step
            X = torch.Tensor(np.array([x[0][0] for x in nstep_traj])).to(device)
            A = torch.Tensor([x[0][1] for x in nstep_traj]).to(device)
            QXA = self.model(X).gather(1, A.to(torch.long).unsqueeze(1))
            loss = self.criterion(QXA, update.unsqueeze(1))
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step() 
    
    def train(self, env, max_episode):
        episode_return = []
        MC_avg_total_reward = []   # NEW NEW NEW
        MC_avg_discounted_reward = []   # NEW NEW NEW
        V_init_state = []   # NEW NEW NEW
        episode = 0
        episode_cum_reward = 0
        state, _ = env.reset()
        epsilon = self.epsilon_max
        step = 0
        nstep_traj = []
        while episode < max_episode:
            # update epsilon
            if step > self.epsilon_delay:
                epsilon = max(self.epsilon_min, epsilon-self.epsilon_step)
            # select epsilon-greedy action
            if np.random.rand() < epsilon:
                action = env.action_space.sample()
            else:
                action = greedy_action(self.model, state)
            # step
            next_state, reward, done, trunc, _ = env.step(action)
            nstep_traj.append([state, action, reward, next_state, done])
            if len(nstep_traj)==self.n_step  :
                self.memory.append(nstep_traj)
                nstep_traj = []
            #self.memory.append(state, action, reward, next_state, done)
            episode_cum_reward += reward
            # train
            for _ in range(self.nb_gradient_steps): 
                self.gradient_step()
            # update target network if needed
            if self.update_target_strategy == 'replace':
                if step % self.update_target_freq == 0: 
                    self.target_model.load_state_dict(self.model.state_dict())
            if self.update_target_strategy == 'ema':
                target_state_dict = self.target_model.state_dict()
                model_state_dict = self.model.state_dict()
                tau = self.update_target_tau
                for key in model_state_dict:
                    target_state_dict[key] = tau*model_state_dict[key] + (1-tau)*target_state_dict[key]
                self.target_model.load_state_dict(target_state_dict)
            # next transition
            step += 1            
            if done or trunc or step%200 ==0:
                episode += 1
                # Monitoring
                if self.monitoring_nb_trials>0:
                    
                    MC_dr, MC_tr = self.MC_eval(env, self.monitoring_nb_trials)    # NEW NEW NEW
                    V0 = self.V_initial_state(env, self.monitoring_nb_trials)   # NEW NEW NEW
                    MC_avg_total_reward.append(MC_tr)   # NEW NEW NEW
                    MC_avg_discounted_reward.append(MC_dr)   # NEW NEW NEW
                    V_init_state.append(V0)   # NEW NEW NEW
                    episode_return.append(episode_cum_reward)   # NEW NEW NEW
                    print("Episode ", '{:2d}'.format(episode), 
                          ", epsilon ", '{:6.2f}'.format(epsilon), 
                          ", batch size ", '{:4d}'.format(len(self.memory)), 
                          ", ep return ", '{:4.1f}'.format(episode_cum_reward), 
                          ", MC tot ", '{:6.2f}'.format(MC_tr),
                          ", MC disc ", '{:6.2f}'.format(MC_dr),
                          ", V0 ", '{:6.2f}'.format(V0),
                          sep='', end = "\r")
                else:
                    episode_return.append(episode_cum_reward)
                    print("Episode ", '{:2d}'.format(episode), 
                          ", epsilon ", '{:6.2f}'.format(epsilon), 
                          ", batch size ", '{:4d}'.format(len(self.memory)), 
                          ", ep return ", '{:e}'.format(episode_cum_reward), 
                          sep='')

                
                state, _ = env.reset()
                episode_cum_reward = 0
            else:
                state = next_state
        return episode_return, MC_avg_discounted_reward, MC_avg_total_reward, V_init_state
import gymnasium as gym
import env_hiv

import torch
import torch.nn as nn
import matplotlib.pyplot as plt
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Declare network
env = env_hiv.HIVPatient(domain_randomization=False)
state_dim = env.observation_space.shape[0]
n_action = env.action_space.n 
nb_neurons=256
DQN = torch.nn.Sequential(nn.Linear(state_dim, nb_neurons),
                          nn.ReLU(),
                          nn.Linear(nb_neurons, nb_neurons),
                          nn.ReLU(), 
                          nn.Linear(nb_neurons, nb_neurons),
                          nn.ReLU(), 
                          nn.Linear(nb_neurons, n_action)).to(device)



from typing import Protocol
import numpy as np
from tqdm import tqdm
import pickle

config = {'nb_actions': env.action_space.n,
          'learning_rate': 0.001,
          'gamma': 0.99,
          'buffer_size': 1000000,
          'epsilon_min': 0.01,
          'epsilon_max': 1.,
          'epsilon_decay_period': 1000,
          'epsilon_delay_decay': 20,
          'batch_size': 1024,
          'gradient_steps': 1,
          'update_target_strategy': 'ema', # or 'ema'
          'update_target_freq': 50,
          'update_target_tau': 0.005,
          'criterion': torch.nn.SmoothL1Loss(),
          'monitoring_nb_trials': 0,
          "domain_randomization":False,
          "epochs":1000}
from torch.distributions.categorical import Categorical
class ProjectAgent:

    def __init__(self) :
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.config = config
        self.env =env_hiv.HIVPatient(domain_randomization=config['domain_randomization'])
        self.agent = dqn_agent(config, DQN)
    # def act(self, observation: np.ndarray, use_random: bool = False) -> int:
    #     return greedy_action(self.agent.model, observation)
    def act(self, observation: np.ndarray, use_random: bool = False) -> int:
        logits = self.agent.model(torch.Tensor(observation).to(self.device))
        probs = Categorical(logits=logits)
        action = probs.sample()
        return action
    def save(self, path=""):
        serialized= {"dqn":self.agent.model.cpu(), "config":self.config}
        with open(path+'saved.pkl', 'wb') as f:  # open a text file
            pickle.dump(serialized, f) # serialize the list
    def load(self):
        
        
        with open('saved.pkl', 'rb') as f:  # open a text file
            saved = pickle.load( f) # serialize the list
        #self.__init__(saved['config'])
        self.agent.model = saved["dqn"].to(self.device)
        try : 
            x,_ = self.env.reset()
            self.act(x)
        except : 
            raise Exception("Actor incompatible with environnement")
    # def load(self):
    #     with open('saved.pkl', 'rb') as f:  # open a text file
    #         saved = pickle.load( f) # serialize the list
    #     #self.__init__(saved['config'])
    #     self.agent.model = saved["dqn"].to(self.device)
    #     try : 
    #         x,_ = self.env.reset()
    #         self.act(x)
    #     except : 
    #         raise E
    
    def train(self):
        return self.agent.train(self.env,self.config['epochs'])
