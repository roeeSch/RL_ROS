import torch
import torch.nn as nn
import torch.optim as optim
from data import Experience, TrainBatch
from typing import List, Type
from abc import ABC, abstractmethod
from torch.distributions import Categorical

entropy_gamma = 0.97

class BaseTrainer(ABC):
    def __init__(
        self,
        policy: nn.Module,
        optimizer: Type[optim.Optimizer],
        lr: float,
        num_actions: int,
        entropy_coef: float = 0.3, #for entropy
        gamma: float = 0.99,
        device = "cpu",
        **kw,
    ):
        self.model = policy
        self.model.set_optimizer(optimizer, lr=lr)
        self.entropy_coef = entropy_coef
        self.gamma = gamma
        if isinstance(device, str):
            self.device = torch.device("cuda" if device=="cuda" and torch.cuda.is_available() else "cpu")
        else:
            self.device = device
        self.device = 'cpu'
        self.current_episode = 0

        self.model.to(self.device)
        self.max_entropy = -torch.log(torch.Tensor([1/num_actions])).to(device=self.device)
    
    def entropy_loss(self, action_scores: torch.Tensor):
        state_entropy = -(action_scores.softmax(dim=1) * action_scores.log_softmax(dim=1)).sum(dim=1).mean(dim=0)
        return -state_entropy/self.max_entropy #min loss_e -> max entropy

    def to(self, device):
        self.model.to(device)
    
    def train(self, batch: List[Experience]):

        self.entropy_coef *= entropy_gamma

        self.current_episode += 1

        batch = TrainBatch(*zip(*batch), device=self.device)

        return self.train_batch(batch)

    @abstractmethod
    def train_batch(self, batch: TrainBatch):

        """
        Train the policy according to batch.
        Return tuple of loss, and a dictionary of all the losses
        """
        pass


class QNetworkTrainer(BaseTrainer):        

    def train_batch(self, batch: TrainBatch):

        next_state_values = self.model(batch.next_states).max(dim=1)[0][..., None].detach() * (~batch.is_dones)
        # next_state_values[batch.is_dones] = torch.zeros(torch.sum(batch.is_dones).item(), device=self.device)

        expected_state_action_values = (next_state_values * self.gamma) + batch.rewards

        action_scores = self.model(batch.states)
        loss_e = self.entropy_loss(action_scores)
        state_action_values = action_scores.gather(1, batch.actions)

        criterion = nn.SmoothL1Loss()
        loss_p = criterion(state_action_values, expected_state_action_values)

        loss = loss_p + self.entropy_coef*loss_e

        # Optimize the model
        self.model.optimizer.zero_grad()
        loss.backward()
        self.model.optimizer.step()
        
        loss = loss.to("cpu").item()
        # print("total_loss = ", loss)
        return loss, dict(
            loss=loss, \
            loss_p=loss_p.to("cpu").item(), \
            loss_e=loss_e.to("cpu").item())


class DQNTrainer(BaseTrainer):
    def __init__(
        self,
        policy: nn.Module,
        optimizer: Type[optim.Optimizer],
        lr: float,
        num_actions: int,
        max_grad_norm: float,
        entropy_coef: float = 0.3, #for entropy
        gamma: float = 0.99,
        tau: float = 1.0,
        target_update: int = 10,
        device = "cpu",
        **kw,
    ):
        super(DQNTrainer, self).__init__(
            policy, 
            optimizer, 
            lr, 
            num_actions, 
            entropy_coef, 
            gamma,  
            device,
            **kw,
        )
        
        self.max_grad_norm = max_grad_norm
        self.tau = tau
        self.target_update = target_update

    def train_batch(self, batch: TrainBatch):

        next_state_values = self.model.target_net(batch.next_states).max(dim=1)[0][..., None].detach() * (~batch.is_dones)
        # next_state_values[batch.is_dones] = torch.zeros(torch.sum(batch.is_dones).item(), device=self.device)

        expected_state_action_values = (next_state_values * self.gamma) + batch.rewards

        action_scores = self.model(batch.states)
        loss_e = self.entropy_loss(action_scores)
        state_action_values = action_scores.gather(1, batch.actions)

        criterion = nn.SmoothL1Loss()
        loss_p = criterion(state_action_values, expected_state_action_values)

        loss = loss_p + self.entropy_coef*loss_e

        # Optimize the model
        self.model.optimizer.zero_grad()
        loss.backward()
        self.model.clip_grads(self.model.policy_net, -self.max_grad_norm, self.max_grad_norm)
        self.model.optimizer.step()
        
        if self.current_episode % self.target_update == 0:
            self.model.soft_update(self.model.target_net, self.model.policy_net, tau=self.tau)
        
        loss = loss.to("cpu").item()
        # print("total_loss = ", loss)
        return loss, dict(
            loss=loss, \
            loss_p=loss_p.to("cpu").item(), \
            loss_e=loss_e.to("cpu").item())


class DQNRainbowTrainer(DQNTrainer):

    def train_batch(self, batch: TrainBatch):

        with torch.no_grad():
            next_action_scores = self.model(batch.next_states)
        next_actions = next_action_scores.max(dim=1)[1][..., None]
        next_state_values = self.model.target_net(batch.next_states).gather(1, next_actions).detach() * (~batch.is_dones)
        # next_state_values[batch.is_dones] = torch.zeros(torch.sum(batch.is_dones).item(), device=self.device)

        expected_state_action_values = (next_state_values * self.gamma) + batch.rewards

        action_scores = self.model(batch.states)
        loss_e = self.entropy_loss(action_scores)
        state_action_values = action_scores.gather(1, batch.actions)

        criterion = nn.SmoothL1Loss()
        loss_p = criterion(state_action_values, expected_state_action_values)

        loss = loss_p + self.entropy_coef*loss_e

        # Optimize the model
        self.model.optimizer.zero_grad()
        loss.backward()
        self.model.clip_grads(self.model.policy_net, -self.max_grad_norm, self.max_grad_norm)
        self.model.optimizer.step()
        
        if self.current_episode % self.target_update == 0:
            self.model.soft_update(self.model.target_net, self.model.policy_net, tau=self.tau)
        
        loss = loss.to("cpu").item()
        # print("total_loss = ", loss)
        return loss, dict(
            loss=loss, \
            loss_p=loss_p.to("cpu").item(), \
            loss_e=loss_e.to("cpu").item())


class AACTrainer(BaseTrainer):
    def __init__(
        self,
        policy: nn.Module,
        optimizer: Type[optim.Optimizer],
        lr: float,
        num_actions: int,
        max_grad_norm: float,
        entropy_coef: float = 0.3, #for entropy
        value_coef: float = 0.5,
        gae_coef: float = 1.0,
        gamma: float = 0.99,
        normalize_advantages: bool = False,
        device = "cpu",
        **kw,
    ):
        super(AACTrainer, self).__init__(
            policy, 
            optimizer, 
            lr, 
            num_actions, 
            entropy_coef, 
            gamma, 
            device,
            **kw,
        )
        
        self.max_grad_norm = max_grad_norm
        self.value_coef = value_coef
        self.gae_coef = gae_coef
        self.normalize_advantages = normalize_advantages

    def train_batch(self, batch: TrainBatch):

        action_scores = self.model(batch.states)
        state_values = self.model.state_net(batch.states)
        
        loss_e = self.entropy_loss(action_scores)

        loss_v = nn.MSELoss()(batch.qvals, state_values)
        
        with torch.no_grad():
            # basically, no need of max since shape is [N,1]
            next_state_values = self.model.state_net(batch.next_states).max(dim=1)[0][..., None] * (~batch.is_dones)
            advantages = next_state_values * self.gamma + batch.rewards - state_values
            next_gae = 0
            for i in range(advantages.shape[0]-1, -1, -1):
                next_gae = advantages[i] + self.gamma * self.gae_coef * next_gae
                advantages[i] = next_gae
            # advantages = batch.qvals - state_values
            if self.normalize_advantages:
                advantages = (advantages - advantages.mean())
                if batch.states.shape[0]>1:
                    advantages /= (advantages.std() + 1e-8)
        
        log_action_proba = nn.functional.log_softmax(action_scores, dim=1)
        selected_action_log_proba = log_action_proba.gather(dim=1, index=batch.actions)
        weighted_avg_experience_rewards = selected_action_log_proba * advantages
        loss_p = -weighted_avg_experience_rewards.mean()

        loss = loss_p + self.entropy_coef*loss_e + self.value_coef*loss_v

        # Optimize the model
        self.model.optimizer.zero_grad()
        loss.backward()
        self.model.clip_grads(self.model.policy_net, -self.max_grad_norm, self.max_grad_norm)
        self.model.clip_grads(self.model.state_net, -self.max_grad_norm, self.max_grad_norm)
        self.model.optimizer.step()
        
        loss = loss.to("cpu").item()
        # print("total_loss = ", loss)
        return loss, dict(
            loss=loss, \
            loss_p=loss_p.to("cpu").item(), \
            loss_e=loss_e.to("cpu").item(), \
            loss_v=loss_v.to("cpu").item())


class PPOTrainer(BaseTrainer):
    def __init__(
        self,
        policy: nn.Module,
        optimizer: Type[optim.Optimizer],
        lr: float,
        num_actions: int,
        max_grad_norm: float,
        entropy_coef: float = 0.3, #for entropy
        value_coef: float = 0.5,
        gae_coef: float = 1.0,
        gamma: float = 0.99,
        normalize_advantages: bool = False,
        n_epochs: int = 10,
        clip_range: float = 0.2,
        device = "cpu",
        **kw,
    ):
        super(PPOTrainer, self).__init__(
            policy, 
            optimizer, 
            lr, 
            num_actions, 
            entropy_coef, 
            gamma, 
            device,
            **kw,
        )
        
        self.max_grad_norm = max_grad_norm
        self.value_coef = value_coef
        self.gae_coef = gae_coef
        self.normalize_advantages = normalize_advantages
        self.n_epochs = n_epochs
        self.clip_range = clip_range

    def train_batch(self, batch: TrainBatch):

        old_action_scores = self.model(batch.states).detach()
        state_values = self.model.state_net(batch.states).detach()
        
        with torch.no_grad():
            # basically, no need of max since shape is [N,1]
            next_state_values = self.model.state_net(batch.next_states).max(dim=1)[0][..., None] * (~batch.is_dones)
            advantages = next_state_values * self.gamma + batch.rewards - state_values
            next_gae = 0
            for i in range(advantages.shape[0]-1, -1, -1):
                next_gae = advantages[i] + self.gamma * self.gae_coef * next_gae
                advantages[i] = next_gae
            # advantages = batch.qvals - state_values
            if self.normalize_advantages:
                advantages = (advantages - advantages.mean())
                if batch.states.shape[0]>1:
                    advantages /= (advantages.std() + 1e-8)
        
            old_log_action_proba = nn.functional.log_softmax(old_action_scores, dim=1)
            old_selected_action_log_proba = old_log_action_proba.gather(dim=1, index=batch.actions)

        losses = []
        losses_p, losses_e, losses_v = [], [], []
        for _ in range(self.n_epochs):

            state_values = self.model.state_net(batch.states)
            loss_v = nn.MSELoss()(batch.qvals, state_values)
            
            action_scores = self.model(batch.states)
            log_action_proba = nn.functional.log_softmax(action_scores, dim=1)
            selected_action_log_proba = log_action_proba.gather(dim=1, index=batch.actions)

            ratio = torch.exp(selected_action_log_proba - old_selected_action_log_proba)

            loss_p1 = advantages * ratio
            loss_p2 = advantages * torch.clamp(ratio, 1 - self.clip_range, 1 + self.clip_range)
            loss_p = -torch.min(loss_p1, loss_p2).mean()
        
            loss_e = self.entropy_loss(action_scores)

            loss = loss_p + self.entropy_coef*loss_e + self.value_coef*loss_v

            # Optimize the model
            self.model.optimizer.zero_grad()
            loss.backward()
            self.model.clip_grads(self.model.policy_net, -self.max_grad_norm, self.max_grad_norm)
            self.model.clip_grads(self.model.state_net, -self.max_grad_norm, self.max_grad_norm)
            self.model.optimizer.step()

            # logging
            losses.append(loss.to("cpu").item())
            losses_p.append(loss_p.to("cpu").item())
            losses_e.append(loss_e.to("cpu").item())
            losses_v.append(loss_v.to("cpu").item())
        
        def mean(l):
            return sum(l)/len(l)

        loss = mean(losses)
        loss_p = mean(losses_p)
        loss_e = mean(losses_e)
        loss_v = mean(losses_v)

        # print("total_loss = ", loss)
        return loss, dict(
            loss=loss, \
            loss_p=loss_p, \
            loss_e=loss_e, \
            loss_v=loss_v)


class SACTrainer(BaseTrainer):
    def __init__(
        self,
        policy: nn.Module,
        optimizer: Type[optim.Optimizer],
        lr: float,
        num_actions: int,
        entropy_coef: float = None, #for entropy
        gamma: float = 0.99,
        tau: float = 0.005,
        device = "cpu",
        **kw,
    ):
        super(SACTrainer, self).__init__(
            policy, 
            optimizer, 
            lr, 
            num_actions, 
            entropy_coef, 
            gamma, 
            device,
            **kw,
        )
        self.tau = tau

    def train_batch(self, batch: TrainBatch):

        alpha = torch.exp(self.model.log_alpha.detach())

        ################# Critic optimizer
        with torch.no_grad():
            action_scores = self.model.actor(batch.next_states)
            action_probabilities = nn.functional.softmax(action_scores, dim=1)
            '''action_distribution = Categorical(action_probabilities)
            next_state_action = action_distribution.sample().cpu()'''
            log_action_probabilities = torch.log(action_probabilities)

            q_next_target = self.model.critic_target(batch.next_states)
            min_q_next_target, _ = torch.min(torch.cat([q[..., None] for q in q_next_target], dim=-1), dim=-1)
            min_q_next_target = action_probabilities * (min_q_next_target - alpha * log_action_probabilities)# 
            min_q_next_target = torch.sum(min_q_next_target, dim=1, keepdim=True)
            next_q_value = batch.rewards + (~batch.is_dones) * self.gamma * min_q_next_target

        # Get current Q-values estimates for each critic network
        all_action_scores = self.model.critic(batch.states)
        current_q_values = tuple(action_scores.gather(1, batch.actions) for action_scores in all_action_scores)

        # Compute critic loss
        critic_loss = sum([nn.functional.mse_loss(current_q, next_q_value) for current_q in current_q_values])
        
        self.model.critic.optimizer.zero_grad()
        critic_loss.backward()
        self.model.critic.optimizer.step()

        loss_v = critic_loss.to("cpu").item()

        ################# Actor optimizer
        action_scores = self.model.actor(batch.states)
        action_probabilities = nn.functional.softmax(action_scores, dim=1)
        log_action_probabilities = torch.log(action_probabilities)
        q_values_pi = self.model.critic(batch.states)
        min_qf_pi, _ = torch.min(torch.cat([q[..., None] for q in q_values_pi], dim=-1), dim=-1)
        inside_term = alpha * log_action_probabilities - min_qf_pi
        actor_loss = (action_probabilities * inside_term).sum(dim=1).mean()# 

        # Optimize the actor
        self.model.actor.optimizer.zero_grad()
        actor_loss.backward()
        self.model.actor.optimizer.step()

        loss_p = actor_loss.to("cpu").item()
        
        ################# Alpha optimizer
        log_pi = torch.sum(log_action_probabilities * action_probabilities, dim=1)# 
        alpha_loss = -(self.model.log_alpha * (log_pi + self.max_entropy).detach()).mean()

        self.model.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.model.alpha_optimizer.step()

        loss_e = alpha_loss.to("cpu").item()

        # Update target networks
        self.model.soft_update(self.model.critic_target, self.model.critic, tau=self.tau)

        loss = loss_v + loss_p + loss_e

        # print("total_loss = ", loss)
        return loss, dict(
            loss=loss, \
            loss_p=loss_p, \
            loss_e=loss_e, \
            loss_v=loss_v)


'''class TD3Trainer(BaseTrainer):
    def __init__(
        self,
        policy: nn.Module,
        optimizer: Type[optim.Optimizer],
        lr: float,
        num_actions: int,
        entropy_coef: float = 0.3, #for entropy
        gamma: float = 0.99,
        tau: float = 0.005,
        target_policy_noise: float = 0.2,
        target_noise_clip: float = 0.5,
        policy_delay: int = 2,
        device = "cpu",
        **kw,
    ):
        super(TD3Trainer, self).__init__(
            policy, 
            optimizer, 
            lr, 
            num_actions, 
            entropy_coef, 
            gamma, 
            device,
            **kw,
        )
        self.tau = tau
        self.target_policy_noise = target_policy_noise
        self.target_noise_clip = target_noise_clip
        self.policy_delay = policy_delay

        self._n_updates = 0

    def train_batch(self, batch: TrainBatch):
        
        self._n_updates += 1
        batch_size = batch.actions.shape[0]

        with torch.no_grad():
            noise = torch.empty(batch_size,1, device=self.device).normal_(0, self.target_policy_noise)
            noise = noise.clamp(-self.target_noise_clip, self.target_noise_clip)
            next_actions = (self.model.actor_target(batch.next_states) + noise).clamp(-1, 1)

            # Compute the next Q-values: min over all critics targets
            next_q_values = torch.cat(self.model.critic_target(batch.next_states, next_actions), dim=1)
            next_q_values, _ = torch.min(next_q_values, dim=1, keepdim=True)
            target_q_values = batch.rewards + (~batch.is_dones) * self.gamma * next_q_values
        
        # Get current Q-values estimates for each critic network
        current_q_values = self.model.critic(batch.states, batch.scores)

        # Compute critic loss
        critic_loss = sum([nn.MSELoss()(current_q, target_q_values) for current_q in current_q_values])
        
        self.model.critic.optimizer.zero_grad()
        critic_loss.backward()
        self.model.critic.optimizer.step()

        loss_v = critic_loss.to("cpu").item()
        loss_e = 0
        loss_p = 0

        # Delayed policy updates
        if self._n_updates % self.policy_delay == 0:
            # Compute actor loss
            action_scores = self.model.actor(batch.states)
            actor_loss = -self.model.critic.q1_forward(batch.states, action_scores).mean()
            # Add entropy
            loss_e = self.entropy_loss(action_scores)
            total_actor_loss = actor_loss + self.entropy_coef*loss_e

            # Optimize the actor
            self.model.actor.optimizer.zero_grad()
            total_actor_loss.backward()
            self.model.actor.optimizer.step()

            self.model.soft_update(self.model.critic_target, self.model.critic, tau=self.tau)
            self.model.soft_update(self.model.actor_target, self.model.actor, tau=self.tau)

            loss_e = loss_e.to("cpu").item()
            loss_p = actor_loss.to("cpu").item()
        loss = loss_v + loss_p + loss_e

        # print("total_loss = ", loss)
        return loss, dict(
            loss=loss, \
            loss_p=loss_p, \
            loss_e=loss_e, \
            loss_v=loss_v)'''

class TD3Trainer(BaseTrainer):
    def __init__(
        self,
        policy: nn.Module,
        optimizer: Type[optim.Optimizer],
        lr: float,
        num_actions: int,
        entropy_coef: float = 0.3, #for entropy
        gamma: float = 0.99,
        tau: float = 0.005,
        target_policy_noise: float = 0.2,
        target_noise_clip: float = 0.5,
        policy_delay: int = 2,
        device = "cpu",
        **kw,
    ):
        super(TD3Trainer, self).__init__(
            policy, 
            optimizer, 
            lr, 
            num_actions, 
            entropy_coef, 
            gamma, 
            device,
            **kw,
        )
        self.tau = tau
        self.target_policy_noise = target_policy_noise
        self.target_noise_clip = target_noise_clip
        self.policy_delay = policy_delay

        self._n_updates = 0

    def train_batch(self, batch: TrainBatch):
        
        self._n_updates += 1
        batch_size = batch.actions.shape[0]

        with torch.no_grad():
            noise = torch.empty(*batch.actions.shape, device=self.device).normal_(0, self.target_policy_noise)
            noise = noise.clamp(-self.target_noise_clip, self.target_noise_clip)
            next_action_scores = self.model.actor_target(batch.next_states) + noise
            next_action_probabilities = nn.functional.softmax(next_action_scores, dim=1)
            # next_action_probabilities = (next_action_probabilities + noise).clamp(-1, 1)

            # Compute the next Q-values: min over all critics targets

            next_q_values = self.model.critic_target(batch.next_states)
            min_next_q_values, _ = torch.min(torch.cat([q[..., None] for q in next_q_values], dim=-1), dim=-1)
            min_next_q_values = torch.sum(next_action_probabilities*min_next_q_values, dim=1, keepdim=True)
            target_q_values = batch.rewards + (~batch.is_dones) * self.gamma * min_next_q_values
        
        all_action_scores = self.model.critic(batch.states)
        current_q_values = tuple(action_scores.gather(1, batch.actions) for action_scores in all_action_scores)

        # Compute critic loss
        critic_loss = sum([nn.functional.mse_loss(current_q, target_q_values) for current_q in current_q_values])
        
        self.model.critic.optimizer.zero_grad()
        critic_loss.backward()
        self.model.critic.optimizer.step()

        loss_v = critic_loss.to("cpu").item()
        loss_e = 0
        loss_p = 0

        # Delayed policy updates
        if self._n_updates % self.policy_delay == 0:
            # Compute actor loss
            action_scores = self.model.actor(batch.states)
            action_probabilities = nn.functional.softmax(action_scores, dim=1)
            critic_scores = torch.sum(action_probabilities*self.model.critic.q1_forward(batch.states), dim=1, keepdim=True)
            actor_loss = -critic_scores.mean()

            # Add entropy
            loss_e = self.entropy_loss(action_scores)
            total_actor_loss = actor_loss + self.entropy_coef*loss_e

            # Optimize the actor
            self.model.actor.optimizer.zero_grad()
            total_actor_loss.backward()
            self.model.actor.optimizer.step()

            self.model.soft_update(self.model.critic_target, self.model.critic, tau=self.tau)
            self.model.soft_update(self.model.actor_target, self.model.actor, tau=self.tau)

            loss_e = loss_e.to("cpu").item()
            loss_p = actor_loss.to("cpu").item()
        loss = loss_v + loss_p + loss_e

        # print("total_loss = ", loss)
        return loss, dict(
            loss=loss, \
            loss_p=loss_p, \
            loss_e=loss_e, \
            loss_v=loss_v)