import torch
import torch.nn as nn

class Network(nn.Module):
    
    def __init__(self,nb_inputs,nb_actions):
        super(Network,self).__init__()
        self.nb_inputs = nb_inputs
        self.nb_outputs = nb_actions
        self.first_connection = nn.Linear(self.nb_inputs,30)
        self.second_connection = nn.Linear(30,self.nb_outputs)
        
    def forward(self,state):
        fc1_activated = functional.relu(self.first_connection(state))
        q_values = self.second_connection(fc1_activated)
        return q_values