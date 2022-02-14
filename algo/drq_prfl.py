from typing import Any, Callable, Sequence

import pfrl
from pfrl.replay_buffer import batch_experiences
import torch

from torch.functional import F

from pfrl.agents import SoftActorCritic
from pfrl.utils import clip_l2_grad_norm_

from pfrl.utils.batch_states import _to_recursive
from pfrl.utils.mode_of_distribution import mode_of_distribution

from torch import distributions as dists

from logging import getLogger



def efficient_patch(
    states: Sequence[Any], device: torch.device, phi: Callable[[Any], Any]
) -> Any:
    if isinstance(states, torch.Tensor):
        features = phi(states)
    else:
        features = phi(torch.stack(states))
    # return concat_examples(features, device=device)
    if isinstance(features[0], tuple):
        features = tuple(features)
    return _to_recursive(features, device)

class DrQ(SoftActorCritic):

    saved_attributes = (
        "encoder",
        "policy",
        "q_func1",
        "q_func2",
        "target_q_func1",
        "target_q_func2",
        "encoder_optimizer",
        "policy_optimizer",
        "q_func1_optimizer",
        "q_func2_optimizer",
        "temperature_holder",
        "temperature_optimizer",
    )

    def __init__(
        self,
        policy,
        q_func1,
        q_func2,
        encoder,
        policy_optimizer,
        q_func1_optimizer,
        q_func2_optimizer,
        encoder_optimizer,
        replay_buffer,
        gamma,
        gpu=None,
        replay_start_size=10000,
        minibatch_size=100,
        update_interval=1,
        phi=lambda x: x,
        soft_update_tau=5e-3,
        max_grad_norm=None,
        logger=getLogger(__name__),
        batch_states=efficient_patch,
        burnin_action_func=None,
        initial_temperature=1.0,
        entropy_target=None,
        temperature_optimizer_lr=None,
        act_deterministically=True,
    ):

        super().__init__(policy, q_func1, q_func2, policy_optimizer, q_func1_optimizer, q_func2_optimizer, replay_buffer, gamma, gpu, replay_start_size, minibatch_size, update_interval, phi, soft_update_tau, max_grad_norm, logger, batch_states, burnin_action_func, initial_temperature, entropy_target, temperature_optimizer_lr, act_deterministically)

        self.encoder = encoder
        self.encoder_optimizer = encoder_optimizer
        self.encoder.to(self.device)

    def update_q_func(self, batch):
        """Compute loss for a given Q-function."""

        batch_next_state = batch["next_state"]
        batch_rewards = batch["reward"]
        batch_terminal = batch["is_state_terminal"]
        batch_state = batch["state"]
        batch_actions = batch["action"]
        batch_discount = batch["discount"]

        with torch.no_grad(), pfrl.utils.evaluating(self.policy), pfrl.utils.evaluating(self.target_q_func1), pfrl.utils.evaluating(self.target_q_func2), pfrl.utils.evaluating(self.encoder):
            next_x = self.encoder(batch_next_state)
            next_action_distrib = self.policy(next_x)
            next_actions = next_action_distrib.sample()
            next_log_prob = next_action_distrib.log_prob(next_actions)
            next_q1 = self.target_q_func1((next_x, next_actions))
            next_q2 = self.target_q_func2((next_x, next_actions))
            next_q = torch.min(next_q1, next_q2)
            entropy_term = self.temperature * next_log_prob[..., None]
            assert next_q.shape == entropy_term.shape

            target_q = batch_rewards + batch_discount * (
                1.0 - batch_terminal
            ) * torch.flatten(next_q - entropy_term)
            
        x = self.encoder(batch_state)
        predict_q1 = torch.flatten(self.q_func1((x, batch_actions)))
        predict_q2 = torch.flatten(self.q_func2((x, batch_actions)))

        loss1 = 0.5 * F.mse_loss(target_q, predict_q1)
        loss2 = 0.5 * F.mse_loss(target_q, predict_q2)
        loss = loss1 + loss2

        # Update stats
        self.q1_record.extend(predict_q1.detach().cpu().numpy())
        self.q2_record.extend(predict_q2.detach().cpu().numpy())
        self.q_func1_loss_record.append(float(loss1))
        self.q_func2_loss_record.append(float(loss2))

        self.encoder_optimizer.zero_grad()
        self.q_func1_optimizer.zero_grad()
        self.q_func2_optimizer.zero_grad()

        loss.backward()
        if self.max_grad_norm is not None:
            clip_l2_grad_norm_(self.q_func1.parameters(), self.max_grad_norm)
        self.q_func1_optimizer.step()

        if self.max_grad_norm is not None:
            clip_l2_grad_norm_(self.q_func2.parameters(), self.max_grad_norm)
        self.q_func2_optimizer.step()
        self.encoder_optimizer.step()

    def update_policy_and_temperature(self, batch):
        """Compute loss for actor."""

        batch_state = batch["state"]

        with torch.no_grad(), pfrl.utils.evaluating(self.encoder):
            x = self.encoder(batch_state)

        action_distrib = self.policy(x)
        actions = action_distrib.rsample()
        log_prob = action_distrib.log_prob(actions)
        q1 = self.q_func1((x, actions))
        q2 = self.q_func2((x, actions))
        q = torch.min(q1, q2)

        entropy_term = self.temperature * log_prob[..., None]
        assert q.shape == entropy_term.shape
        loss = torch.mean(entropy_term - q)

        self.policy_optimizer.zero_grad()
        loss.backward()
        if self.max_grad_norm is not None:
            clip_l2_grad_norm_(self.policy.parameters(), self.max_grad_norm)
        self.policy_optimizer.step()

        self.n_policy_updates += 1

        if self.entropy_target is not None:
            self.update_temperature(log_prob.detach())

        # Record entropy
        with torch.no_grad():
            try:
                self.entropy_record.extend(
                    action_distrib.entropy().detach().cpu().numpy()
                )
            except NotImplementedError:
                # Record - log p(x) instead
                self.entropy_record.extend(-log_prob.detach().cpu().numpy())

    def batch_select_greedy_action(self, batch_obs, deterministic=False):
        with torch.no_grad(), pfrl.utils.evaluating(self.policy), pfrl.utils.evaluating(self.encoder):
            batch_xs = self.batch_states(batch_obs, self.device, self.phi)
            policy_out = self.policy(self.encoder(batch_xs))
            if deterministic:
                batch_action = mode_of_distribution(policy_out).cpu().numpy()
            else:
                batch_action = policy_out.sample().cpu().numpy()
        return batch_action

    def update(self, experiences, errors_out=None):
        """Update the model from experiences"""
        batch = batch_experiences(experiences, self.device, self.phi, self.gamma, self.batch_states)
        self.update_q_func(batch)
        self.update_policy_and_temperature(batch)
        self.sync_target_network()







######## Below Invalid Class ########

class DiscreteDrQ(DrQ):
    
    def calc_current_q(self, states, actions):
        x = self.encoder(states)
        curr_q1 = self.q_func1(x)
        curr_q2 = self.q_func2(x)
        act = actions.unsqueeze(-1).long()
        curr_q1 = curr_q1.gather(1, act)
        curr_q2 = curr_q2.gather(1, act)
        return curr_q1, curr_q2
    
    def calc_target_q(self, next_states):
        with torch.no_grad():
            next_x = self.encoder(next_states)
            action_probs = self.policy(next_x)
            next_q1 = self.target_q_func1(next_x)
            next_q2 = self.target_q_func2(next_x)
            next_q = (action_probs * (torch.min(next_q1, next_q2) - self.temperature * torch.log(action_probs))).sum(dim=-1)


    def update_q_func(self, batch):
        """Compute loss for a given Q-function."""

        batch_next_state = batch["next_state"]
        batch_rewards = batch["reward"]
        batch_terminal = batch["is_state_terminal"]
        batch_state = batch["state"]
        batch_actions = batch["action"]
        batch_discount = batch["discount"]

        with torch.no_grad(), pfrl.utils.evaluating(self.policy), pfrl.utils.evaluating(self.target_q_func1), pfrl.utils.evaluating(self.target_q_func2), pfrl.utils.evaluating(self.encoder):
            next_x = self.encoder(batch_next_state)
            next_action_prob = self.policy(next_x)
            next_q1 = self.target_q_func1(next_x)
            next_q2 = self.target_q_func2(next_x)
            next_q = torch.min(next_q1, next_q2)
            entropy_term = self.temperature * torch.log(next_action_prob)
            assert next_q.shape == entropy_term.shape

            v = (next_action_prob * next_q - entropy_term).sum(dim=-1)
            target_q = batch_rewards + batch_discount * (1.0 - batch_terminal) * v
            
        x = self.encoder(batch_state)
        print(f'=====================\nencoder out: {x}')
        actions = batch_actions.unsqueeze(-1).long()
        predict_q1 = self.q_func1((x)).gather(1, actions).flatten()
        predict_q2 = self.q_func2((x)).gather(1, actions).flatten()

        loss1 = 0.5 * torch.functional.F.mse_loss(target_q, predict_q1)
        loss2 = 0.5 * torch.functional.F.mse_loss(target_q, predict_q2)
        loss = loss1 + loss2

        print(f'=====================\nloss: {loss}')

        # Update stats
        self.q1_record.extend(predict_q1.detach().cpu().numpy())
        self.q2_record.extend(predict_q2.detach().cpu().numpy())
        self.q_func1_loss_record.append(float(loss1))
        self.q_func2_loss_record.append(float(loss2))

        self.encoder_optimizer.zero_grad()
        self.q_func1_optimizer.zero_grad()
        self.q_func2_optimizer.zero_grad()

        loss.backward()
        if self.max_grad_norm is not None:
            clip_l2_grad_norm_(self.q_func1.parameters(), self.max_grad_norm)
        self.q_func1_optimizer.step()

        if self.max_grad_norm is not None:
            clip_l2_grad_norm_(self.q_func2.parameters(), self.max_grad_norm)
        self.q_func2_optimizer.step()
        self.encoder_optimizer.step()

    def update_policy_and_temperature(self, batch):
        """Compute loss for actor."""

        batch_state = batch["state"]

        with torch.no_grad(), pfrl.utils.evaluating(self.encoder):
            x = self.encoder(batch_state)

        action_prob = torch.log(self.policy(x))
        
        q1 = self.q_func1(x)
        q2 = self.q_func2(x)
        q = torch.min(q1, q2)

        entropy_term = self.temperature * action_prob
        assert q.shape == entropy_term.shape
        loss = torch.mean(entropy_term - q)

        self.policy_optimizer.zero_grad()
        loss.backward()
        if self.max_grad_norm is not None:
            clip_l2_grad_norm_(self.policy.parameters(), self.max_grad_norm)
        self.policy_optimizer.step()

        self.n_policy_updates += 1

        if self.entropy_target is not None:
            self.update_temperature(torch.log(action_prob).detach())

        # Record entropy
        with torch.no_grad():
            self.entropy_record.extend(-torch.log(action_prob).detach().cpu().numpy())

    def batch_select_greedy_action(self, batch_obs, deterministic=False):
        with torch.no_grad(), pfrl.utils.evaluating(self.policy), pfrl.utils.evaluating(self.encoder):
            batch_xs = self.batch_states(batch_obs, self.device, self.phi)
            policy_out = self.policy(self.encoder(batch_xs))
            print(policy_out)
            policy_out = torch.distributions.Categorical(policy_out)
            if deterministic:
                batch_action = policy_out.probs.argmax(dim=-1).cpu().numpy()
            else:
                batch_action = policy_out.sample().cpu().numpy()
        return batch_action