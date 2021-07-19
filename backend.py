import torch
import torch.nn as nn
import torch.nn.functional as functional
import torch.optim as optim
from torch.autograd import Variable
from random import sample

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

class Memory(object):
    def __init__(self,capacity):
        self.capacity = capacity
        self.memory = []
        
    def push(self,state):
        self.memory.append(state)
        if len(self.memory) > self.capacity:
            del self.memory[0]
    
    def pull(self,batch_size):
        rand_sample = zip(*sample(self.memory,batch_size))
        return map(lambda x:Variable(torch.cat(x,0)),rand_sample)

class Brain():
    def __init__(self,input_nodes,nb_actions,gamma):
        self.gamma = gamma
        self.reward_mean = []
        self.memory = Memory(100000)
        self.model = Network(input_nodes,nb_actions)
        self.optimizer = optim.Adam(self.model.parameters(),lr=0.001)
        self.last_state = torch.Tensor(input_nodes).unsqueeze(0)
        self.last_reward = 0
        self.last_action = 0
        
    def select_action(self,state):
        probs = functional.softmax(self.model.forward(Variable(state,volatile=True))*100)
        action = probs.multinomial(1)
        return action.data[0,0]
    
    def learn(self,prev_state,current_state,prev_action,prev_reward):
        outputs = self.model.forward(prev_state).gather(1,prev_action.unsqueeze(1)).squeeze(1)
        max_futures = self.model.forward(current_state).detach().max(1)[0]
        targets = self.gamma*max_futures + prev_reward
        loss = functional.smooth_l1_loss(outputs,targets)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()