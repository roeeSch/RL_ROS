import torch
import torch.nn as nn
from functools import partial
from typing import List, Type, Tuple
import numpy as np
import pickle
f_ut = open('unitTest.csv','a')

def create_mlp(input_dim: int, output_dim: int, net_arch: List[int], activation_fn: Type[nn.Module] = nn.ReLU) -> nn.Module:

    if len(net_arch) > 0:
        modules = [nn.Linear(input_dim, net_arch[0]), activation_fn()]
    else:
        modules = []

    for idx in range(len(net_arch) - 1):
        modules.append(nn.Linear(net_arch[idx], net_arch[idx + 1]))
        modules.append(activation_fn())

    if output_dim > 0:
        last_layer_dim = net_arch[-1] if len(net_arch) > 0 else input_dim
        modules.append(nn.Linear(last_layer_dim, output_dim))
    
    return nn.Sequential(*modules)


class BaseNetwork(nn.Module):
    def forward(self, x):

        action_scores = self.policy_net(x)
        return action_scores
    
    def predict(self, state):

        with torch.no_grad():
            actions_prob = torch.squeeze(self(state))
        
        return actions_prob.cpu()
    
    def set_optimizer(self, optimizer, lr):
        self.optimizer = optimizer(self.parameters(), lr=lr)
    
    @staticmethod
    def soft_update(target_model: nn.Module, from_model: nn.Module, tau):

        for target_param, input_param in zip(target_model.parameters(), from_model.parameters()):
            target_param.data.copy_(tau*input_param.data + (1.0 - tau)*target_param.data)
    
    @staticmethod
    def clip_grads(model: nn.Module, min_grad, max_grad):

        for param in model.parameters():
            param.grad.data.clamp_(-min_grad, max_grad)


class QNetworkPolicy(BaseNetwork):
    def __init__(self, in_features, out_actions, hidden_layers=None, **kw):
        """
        Create a model which represents the agent's policy.
        :param in_features: Number of input features (in one observation).
        :param out_actions: Number of output actions.
        :param kw: Any extra args needed to construct the model.
        """
        super().__init__()

        activation = nn.ReLU

        if hidden_layers is None:
            hidden_layers = [64,64]

        layers = []
        in_dim = in_features
        for h_dim in hidden_layers:
            layers.append(nn.Linear(in_features=in_dim, out_features=h_dim))
            layers.append(activation())
            in_dim = h_dim
        layers.append(nn.Linear(in_features=in_dim, out_features=out_actions))
        self.policy_net = nn.Sequential(*layers)

        self.params = {"class": self.__class__, "in_features": in_features, "out_actions": out_actions, "hidden_layers": hidden_layers}
    
    def predict(self, state):

        actions_prob = super().predict(state)
        selected_action = actions_prob.argmax(dim=0)
        
        return selected_action, actions_prob


class DQNPolicy(BaseNetwork):
    def __init__(self, in_features, out_actions, hidden_layers=None, **kw):
        
        super().__init__()

        activation = nn.ReLU

        if hidden_layers is None:
            hidden_layers = [64,64]

        layers = []
        in_dim = in_features
        for h_dim in hidden_layers:
            layers.append(nn.Linear(in_features=in_dim, out_features=h_dim))
            layers.append(activation())
            in_dim = h_dim
        layers.append(nn.Linear(in_features=in_dim, out_features=out_actions))
        self.policy_net = nn.Sequential(*layers)
        
        layers = []
        in_dim = in_features
        for h_dim in hidden_layers:
            layers.append(nn.Linear(in_features=in_dim, out_features=h_dim))
            layers.append(activation())
            in_dim = h_dim
        layers.append(nn.Linear(in_features=in_dim, out_features=out_actions))
        self.target_net = nn.Sequential(*layers)

        self.soft_update(self.target_net, self.policy_net, tau=1.0)
        self.params = {"class": self.__class__, "in_features": in_features, "out_actions": out_actions, "hidden_layers": hidden_layers}
    
    def predict(self, state):

        actions_prob = super().predict(state)
        selected_action = actions_prob.argmax(dim=0)
        
        return selected_action, actions_prob


class AACPolicy(BaseNetwork):
    def __init__(self, in_features, out_actions, hidden_layers=None, ortho_init=True, **kw):
        
        super().__init__()

        activation = nn.Tanh

        if hidden_layers is None:
            hidden_layers = [64,64]

        value_layers = []
        in_dim = in_features
        for h_dim in hidden_layers:
            value_layers.append(nn.Linear(in_features=in_dim, out_features=h_dim))
            value_layers.append(activation())
            in_dim = h_dim
        value_final = nn.Linear(in_features=in_dim, out_features=1)
        
        policy_layers = []
        in_dim = in_features
        for h_dim in hidden_layers:
            policy_layers.append(nn.Linear(in_features=in_dim, out_features=h_dim))
            policy_layers.append(activation())
            in_dim = h_dim
        policy_final = nn.Linear(in_features=in_dim, out_features=out_actions)

        if ortho_init:
            map(partial(self.init_weights, gain=2**0.5), value_layers)
            self.init_weights(value_final, gain=1)
            map(partial(self.init_weights, gain=2**0.5), policy_layers)
            self.init_weights(policy_final, gain=0.01)

        value_layers.append(value_final)
        self.state_net = nn.Sequential(*value_layers)

        policy_layers.append(policy_final)
        self.policy_net = nn.Sequential(*policy_layers)
        
        self.params = {"class": self.__class__, "in_features": in_features, "out_actions": out_actions, "hidden_layers": hidden_layers}

    @staticmethod
    def init_weights(module: nn.Module, gain: float = 1):

        if isinstance(module, (nn.Linear, nn.Conv2d)):
            nn.init.orthogonal_(module.weight, gain=gain)
            if module.bias is not None:
                module.bias.data.fill_(0.0)
    
    def predict(self, state):

        actions_prob = super().predict(state)
        selected_action = nn.functional.softmax(actions_prob, dim=-1).multinomial(num_samples=1).item()
        
        return selected_action, actions_prob

PPOPolicy = AACPolicy

class Actor(nn.Module):
    def __init__(self, in_features, out_actions, activation, hidden_layers, **kw):
        
        super().__init__()

        self.mu = create_mlp(in_features, out_actions, hidden_layers, activation)

        self.params = {"class": self.__class__, "in_features": in_features, "out_actions": out_actions, "hidden_layers": hidden_layers}
    
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.mu(state)


class ContinuousCritic(nn.Module):
    def __init__(self, in_features, out_actions, activation, hidden_layers, n_critics=2, **kw):
        
        super().__init__()

        self.q_networks = []
        for idx in range(n_critics):
            q_net = create_mlp(in_features, out_actions, hidden_layers, activation)
            self.add_module(f"qf{idx}", q_net)
            self.q_networks.append(q_net)

        self.params = {"class": self.__class__, "in_features": in_features, "out_actions": out_actions, "hidden_layers": hidden_layers}
    
    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        return tuple(q_net(state) for q_net in self.q_networks)

    def q1_forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        Only predict the Q-value using the first network.
        This allows to reduce computation when all the estimates are not needed
        (e.g. when updating the policy in TD3).
        """
        return self.q_networks[0](state)


'''class ContinuousCritic(nn.Module):
    def __init__(self, in_features, out_actions, hidden_layers, n_critics=2, **kw):
        
        super().__init__()

        activation = nn.ReLU

        self.q_networks = []
        for idx in range(n_critics):
            q_net = create_mlp(in_features + out_actions, 1, hidden_layers, activation)
            self.add_module(f"qf{idx}", q_net)
            self.q_networks.append(q_net)

        self.params = {"class": self.__class__, "in_features": in_features, "out_actions": out_actions, "hidden_layers": hidden_layers}
    
    def forward(self, state: torch.Tensor, actions: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        qvalue_input = torch.cat([state, actions], dim=1)
        return tuple(q_net(qvalue_input) for q_net in self.q_networks)

    def q1_forward(self, state: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        """
        Only predict the Q-value using the first network.
        This allows to reduce computation when all the estimates are not needed
        (e.g. when updating the policy in TD3).
        """
        return self.q_networks[0](torch.cat([state, actions], dim=1))'''


class SACPolicy(BaseNetwork):
    def __init__(self, in_features, out_actions, hidden_layers=None, n_critics=2, **kw):
        
        super().__init__()

        activation = nn.ReLU

        if hidden_layers is None:
            hidden_layers = [256, 256]

        self.actor = Actor(in_features, out_actions, activation, hidden_layers)

        self.critic = ContinuousCritic(in_features, out_actions, activation, hidden_layers, n_critics)
        self.critic_target = ContinuousCritic(in_features, out_actions, activation, hidden_layers, n_critics)
        self.critic_target.load_state_dict(self.critic.state_dict())

        self.log_alpha = nn.Parameter(torch.zeros(1), requires_grad=True)

        self.params = {"class": self.__class__, "in_features": in_features, "out_actions": out_actions, "hidden_layers": hidden_layers}
    
    def forward(self, state):

        action_scores = self.actor(state)
        return action_scores
    
    def predict(self, state):

        actions_prob = super().predict(state)
        selected_action = nn.functional.softmax(actions_prob, dim=-1).multinomial(num_samples=1).item()
        
        return selected_action, actions_prob
    
    def set_optimizer(self, optimizer, lr):
        self.actor.optimizer = optimizer(self.actor.parameters(), lr=lr)
        self.critic.optimizer = optimizer(self.critic.parameters(), lr=lr)
        self.alpha_optimizer = optimizer([self.log_alpha], lr=lr)
    
    
'''class TD3Policy(BaseNetwork):
    def __init__(self, in_features, out_actions, hidden_layers=None, n_critics=2, **kw):
        
        super().__init__()

        activation = nn.ReLU

        if hidden_layers is None:
            hidden_layers = [400, 300]

        self.actor = Actor(in_features, out_actions, hidden_layers)
        self.actor_target = Actor(in_features, out_actions, hidden_layers)
        # Initialize the target to have the same weights as the actor
        self.actor_target.load_state_dict(self.actor.state_dict())

        self.critic = ContinuousCritic(in_features, out_actions, hidden_layers, n_critics)
        self.critic_target = ContinuousCritic(in_features, out_actions, hidden_layers, n_critics)
        self.critic_target.load_state_dict(self.critic.state_dict())

        self.params = {"class": self.__class__, "in_features": in_features, "out_actions": out_actions, "hidden_layers": hidden_layers}
    
    def forward(self, state):

        action_scores = self.actor(state)
        return action_scores
    
    def predict(self, state):

        actions_prob = super().predict(state)
        selected_action = actions_prob.argmax(dim=0)
        
        return selected_action, actions_prob
    
    def set_optimizer(self, optimizer, lr):
        self.actor.optimizer = optimizer(self.actor.parameters(), lr=lr)
        self.critic.optimizer = optimizer(self.critic.parameters(), lr=lr)'''

class TD3Policy(BaseNetwork):
    def __init__(self, in_features, out_actions, hidden_layers=None, n_critics=2, **kw):
        
        super().__init__()

        activation = nn.ReLU

        if hidden_layers is None:
            hidden_layers = [400, 300]

        self.actor = Actor(in_features, out_actions, activation, hidden_layers)
        self.actor_target = Actor(in_features, out_actions, activation, hidden_layers)
        # Initialize the target to have the same weights as the actor
        self.actor_target.load_state_dict(self.actor.state_dict())

        self.critic = ContinuousCritic(in_features, out_actions, activation, hidden_layers, n_critics)
        self.critic_target = ContinuousCritic(in_features, out_actions, activation, hidden_layers, n_critics)
        self.critic_target.load_state_dict(self.critic.state_dict())

        self.params = {"class": self.__class__, "in_features": in_features, "out_actions": out_actions, "hidden_layers": hidden_layers}
    
    def forward(self, state):

        action_scores = self.actor(state)
        return action_scores
    
    def predict(self, state):
        actions_prob = super().predict(state)
        selected_action = nn.functional.softmax(actions_prob, dim=-1).multinomial(num_samples=1).item()
        # import pdb;pdb.set_trace()
        # f_ut.write(','.join([str(x.item()) for x in state[0]])+','+','.join([str(x.item()) for x in nn.functional.softmax(actions_prob, dim=-1)])+'\n')
        # f_ut.flush()
        return selected_action, actions_prob
    
    def set_optimizer(self, optimizer, lr):
        self.actor.optimizer = optimizer(self.actor.parameters(), lr=lr)
        self.critic.optimizer = optimizer(self.critic.parameters(), lr=lr)