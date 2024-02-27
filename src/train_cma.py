import torch.nn as nn
import torch
from env_hiv import HIVPatient
import numpy as np
import pickle
class ProjectAgent(nn.Module):
    def __init__(self,state_n=6,action_n=4,hsize = 32) :
        super().__init__()
        self.env =HIVPatient()
        self.fc1= nn.Linear(state_n, hsize)
        self.fc2= nn.Linear(hsize, hsize)
        self.fc3= nn.Linear(hsize, action_n)
    def forward(self,x) : 
        x= (1e-9+torch.Tensor(x)).log()
        if len(x.shape)==1 :
            x= x.unsqueeze(0)
        x = nn.ReLU()(self.fc1(x))
        x = nn.ReLU()(self.fc2(x))
        return nn.ReLU()(self.fc3(x))
    def get_action(self,x) :
        logits = nn.Softmax(dim=-1)(self.forward(x))
        return torch.argmax(logits, dim=-1)#.item()
    def get_params(self):
        p = np.empty((0,))
        for n in self.parameters():
            p = np.append(p, n.flatten().cpu().detach().numpy())
        return p
    def set_params(self, x):
        start = 0
        for p in self.parameters():
            e = start + np.prod(p.shape)
            p.data = torch.FloatTensor(x[start:e]).reshape(p.shape)
            start = e
    def act(self, observation: np.ndarray, use_random: bool = False) -> int:
        return self.get_action(observation)
    def save(self, path=""):
        serialized= {"params" : self.get_params()}
        with open(path+'saved-cma.pkl', 'wb') as f:  # open a text file
            pickle.dump(serialized, f) # serialize the list
    def load(self):
        with open('saved-cma.pkl', 'rb') as f:  # open a text file
            saved = pickle.load( f) # serialize the list
        #self.__init__(saved['config'])
        self.set_params(saved['params'])
        try : 
            x,_ = self.env.reset()
            self.act(x)
        except : 
            raise Exception("Actor incompatible with environnement")