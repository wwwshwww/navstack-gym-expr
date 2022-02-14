from .drq_prfl import DrQ

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