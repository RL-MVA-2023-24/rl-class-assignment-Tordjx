import torch
import env_hiv
import pickle
import numpy as np
class ProjectAgent:
    def __init__(self) :
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.env =env_hiv.HIVPatient(domain_randomization=False)
        self.models = []
    def act(self, observation: np.ndarray, use_random: bool = False) -> int:
        observation = (torch.Tensor(observation).to(torch.float32)+1e-9).log()
        logits = [m(observation) for m in self.models[:3]]
        #logits =[models[0](observation)]+[models[2](observation)]
        logits =torch.softmax(torch.stack(logits,0),1).mean(0)
        return torch.argmax(logits).item()

    def save(self, path=""):
        serialized= {"models" :self.models}
        with open(path+'models.pkl', 'wb') as f:  # open a text file
            pickle.dump(serialized, f) # serialize the list
    def load(self):
        with open('models.pkl', 'rb') as f:  # open a text file
            saved = pickle.load( f) # serialize the list
        self.models= saved["models"]
        try : 
            x,_ = self.env.reset()
            self.act(x)
        except : 
            raise Exception("Actor incompatible with environnement")