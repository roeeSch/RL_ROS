import torch
import torch.utils.data
from typing import List, NamedTuple
from collections import deque
import random
from typing import List
import pickle


class Experience(NamedTuple):
    """
    Represents one experience tuple for the Agent.
    """
    state: torch.FloatTensor
    next_state: torch.FloatTensor
    scores: torch.FloatTensor
    action: int # categorial
    reward: float
    qval: float
    is_done: bool


class TrainBatch(object):
    
    def __init__(
        # all in shape [batch]
        self,
        states: torch.FloatTensor, #[batch,state]
        next_states: torch.FloatTensor, #[batch,state]
        scores: torch.FloatTensor, #[batch,actions]
        actions: torch.LongTensor, #[batch,1]
        rewards: torch.FloatTensor, #[batch,1]
        qvals: torch.FloatTensor, #[batch,1]
        is_dones: torch.BoolTensor, #[batch,1]
        device,
    ):

        states = torch.stack(list(states), dim=0)
        next_states = torch.stack(list(next_states), dim=0)
        scores = torch.stack(list(scores), dim=0)
        actions = torch.LongTensor(list(actions))[...,None]
        rewards = torch.FloatTensor(list(rewards))[...,None]
        qvals = torch.FloatTensor(list(qvals))[...,None]
        is_dones = torch.torch.BoolTensor(list(is_dones))[...,None]
        assert states.shape[0] == actions.shape[0] == rewards.shape[0] == next_states.shape[0] == is_dones.shape[0]

        self.states = states.to(device)
        self.next_states = next_states.to(device)
        self.scores = scores.to(device)
        self.actions = actions.to(device)
        self.rewards = rewards.to(device)
        self.qvals = qvals.to(device)
        self.is_dones = is_dones.to(device)

class ReplayMemory(object):

    def __init__(self, buffer_size, batch_size, gamma=0.99, off_policy=True):
        self.memory = deque([],maxlen=buffer_size) if off_policy else []
        self.batch_size = batch_size
        self.gamma = gamma
        self.off_policy = off_policy
        self.experiences = []
    
    def load_from_data(self, filename):
        with open(filename, 'rb') as file:
            self.memory = pickle.load(file)
    
    def save_data(self, filename):
        if self.off_policy:
            with open(filename, 'wb') as output:
                pickle.dump(self.memory, output, pickle.HIGHEST_PROTOCOL)

    def push_experience(self, exp: Experience):
        self.experiences.append(exp)
    
    def push_episode(self):
        next_qval = 0
        for i in range(len(self.experiences)-1, -1, -1):
            exp = self.experiences[i]
            next_qval = exp.reward + self.gamma * next_qval
            self.experiences[i] = exp._replace(qval = next_qval)
        for exp in self.experiences:
            self.memory.append(exp)
        self.experiences = []

    def get_batch(self, gradient_steps=1) ->List[Experience]:
        size = len(self.memory)
        if self.off_policy:
            if size < self.batch_size:
                raise Exception(f"Memory is too small for sampling ({size})")
            else:
                for _ in range(gradient_steps):
                    yield random.sample(self.memory, self.batch_size)
        elif size>0:
            if self.batch_size==-1:
                yield self.memory
            else:
                for ind in range(size//self.batch_size):
                    yield self.memory[ind*self.batch_size:(ind+1)*self.batch_size]
                if size%self.batch_size > 0:
                    yield self.memory[-(size%self.batch_size):]
            self.memory = []

