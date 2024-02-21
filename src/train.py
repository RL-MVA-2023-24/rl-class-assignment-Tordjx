import env_hiv
from pyvirtualdisplay import Display
from IPython import display
import torch
#torch.multiprocessing.set_start_method("spawn")
import torch.nn.functional as F
import torch.multiprocessing as mp
import torch.nn as nn
import cma
import numpy as np
import gymnasium as gym
import matplotlib.pyplot as plt
def fitness(x, ann, env, device):
    ann.set_params(torch.Tensor(x))
    return -evaluate(ann, env, device)
def evaluate(ann, env, device,maxiter = 200):
    #env.seed(0) # deterministic for demonstration
    obs,_ = env.reset()
    total_reward = 0
    niter = 0
    while True:
        niter+=1
        # Output of the neural net
        net_output = ann(torch.from_numpy(obs).to(device))
        # the action is the value clipped returned by the nn
        action = net_output.data.cpu().numpy().argmax()
        obs, reward, done, trunc,infos = env.step(action)
        total_reward += reward
        if done or niter>maxiter:
            break
    return total_reward
hidden_dim = 128
class NeuralNetwork(nn.Module):

    def __init__(self, input_shape, n_actions):
        super(NeuralNetwork, self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.l1 = nn.Linear(input_shape, hidden_dim).to(self.device)
        self.l2 = nn.Linear(hidden_dim, hidden_dim).to(self.device)
        self.lout = nn.Linear(hidden_dim, n_actions).to(self.device)

    def forward(self, x):
        if type(x) != torch.Tensor :
            x = torch.from_numpy(x)
        x = F.relu(self.l1(x.float()))
        x = F.relu(self.l2(x))
        return self.lout(x)

    def get_params(self):
        p = np.empty((0,))
        for n in self.parameters():
            p = np.append(p, n.flatten().cpu().detach().numpy())
        return p

    def set_params(self, x):
        start = 0
        for p in self.parameters():
            e = start + np.prod(p.shape)
            p.data = (x[start:e]).to(torch.float32).reshape(p.shape).to(self.device)
            start = e


from joblib import delayed, Parallel





from typing import Protocol
import numpy as np
from tqdm import tqdm
import pickle
class ProjectAgent():

    def __init__(self,env, niter =50, noffsprings =25) -> None:
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.ann = NeuralNetwork(env.observation_space.shape[0], env.action_space.n).to(self.device)
        self.es = cma.CMAEvolutionStrategy(len(self.ann.get_params()) * [0], 0.1, {'seed': 123})
        self.niter = niter
        self.noffsprings = noffsprings
        self.env = env
    def act(self, observation: np.ndarray, use_random: bool = False) -> int:
        return self.ann(observation)
    def save(self, path=""):
        serialized= {"ann":self.ann.cpu()}
        with open(path+'saved.pkl', 'wb') as f:  # open a text file
            pickle.dump(serialized, f) # serialize the list
    def load(self):
        with open('saved.pkl', 'rb') as f:  # open a text file
            saved = pickle.load( f) # serialize the list
        self.ann = saved['ann']
        try : 
            x,_ = self.env.reset()
            self.act(x)
        except : 
            raise Exception("Actor incompatible with environnement")
    def train(self):
        for i in range(self.niter):
            solutions = torch.from_numpy(np.array(self.es.ask(self.noffsprings))).to(self.device)
            fits=Parallel(n_jobs=17, backend = "loky")(delayed(fitness)(x, self.ann, self.env, self.device) for i, x in enumerate(solutions))    
            self.es.tell(solutions.cpu(), fits)
            self.es.disp()