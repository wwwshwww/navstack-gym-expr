import torch
from torch import nn, distributions
from torch.functional import F
from torchvision import transforms
import pfrl
import numpy as np

from . import utils
from .drq_prfl import DrQ, DiscreteDrQ

class RandomShiftsAug(nn.Module):
    def __init__(self, pad):
        super().__init__()
        self.pad = pad

    def forward(self, x):
        n, c, h, w = x.size()
        assert h == w
        padding = tuple([self.pad] * 4)
        x = F.pad(x, padding, 'replicate')
        eps = 1.0 / (h + 2 * self.pad)
        arange = torch.linspace(-1.0 + eps,
                                1.0 - eps,
                                h + 2 * self.pad,
                                device=x.device,
                                dtype=x.dtype)[:h]
        arange = arange.unsqueeze(0).repeat(h, 1).unsqueeze(2)
        base_grid = torch.cat([arange, arange.transpose(1, 0)], dim=2)
        base_grid = base_grid.unsqueeze(0).repeat(n, 1, 1, 1)

        shift = torch.randint(0,
                              2 * self.pad + 1,
                              size=(n, 1, 1, 2),
                              device=x.device,
                              dtype=x.dtype)
        shift *= 2.0 / (h + 2 * self.pad)

        grid = base_grid + shift
        return F.grid_sample(x,
                             grid,
                             padding_mode='zeros',
                             align_corners=False)

class Encoder(nn.Module):
    def __init__(self, obs_shape, max_val=100):
        super().__init__()

        self.num_layers = 4
        self.num_filters = 32
        self.max_v = max_val

        cw = utils.conv2d_size_out(obs_shape[-1], 3, 2)
        cw = utils.conv2d_size_out(cw, 3, 1)
        cw = utils.conv2d_size_out(cw, 3, 1)
        cw = utils.conv2d_size_out(cw, 3, 1)

        self.repr_dim = self.num_filters * cw * cw

        self.convnet = nn.Sequential(
            nn.Conv2d(obs_shape[0], self.num_filters, 3, stride=2),
            nn.ReLU(inplace=True), 
            nn.Conv2d(self.num_filters, self.num_filters, 3, stride=1),
            nn.ReLU(inplace=True), 
            nn.Conv2d(self.num_filters, self.num_filters, 3, stride=1),
            nn.ReLU(inplace=True), 
            nn.Conv2d(self.num_filters, self.num_filters, 3, stride=1),
            nn.ReLU(inplace=True)
        )

        self.apply(utils.weight_init)

    def forward(self, obs):
        obs = obs / self.max_v # obs value
        h = self.convnet(obs)
        h = h.view(h.shape[0], -1)
        return h

class QFunction(nn.Module):
    """Critic"""
    def __init__(self, repr_dim, action_shape, feature_dim, hidden_dim):
        super().__init__()
        
        self.trunk = nn.Sequential(
            nn.Linear(repr_dim, feature_dim),
            nn.LayerNorm(feature_dim),
            nn.Tanh()
        )
        self.q = utils.make_linear_layer(feature_dim + action_shape[0], hidden_dim, 1)
        self.apply(utils.weight_init)

    def forward(self, obs_and_action):
        obs = obs_and_action[0]
        obs = self.trunk(obs)
        h = torch.cat((obs, obs_and_action[1]), axis=-1)
        return self.q(h)

class PolicyFunction(nn.Module):
    """Actor"""
    def __init__(self, repr_dim, action_shape, feature_dim, hidden_dim):
        super().__init__()

        self.trunk = nn.Sequential(
            nn.Linear(repr_dim, feature_dim),
            nn.LayerNorm(feature_dim),
            nn.Tanh()
        )
        self.policy = utils.make_linear_layer(feature_dim, hidden_dim, action_shape[0]*2)
        self.apply(utils.weight_init)

    def forward(self, obs):
        h = self.trunk(obs)
        mustd = self.policy(h)
        return utils.squashed_diagonal_gaussian_head(mustd)

def make_DrQ_agent(
    experiment_name, 
    obs_space, 
    action_space, 
    feature_dim, 
    hidden_dim, 
    lr, 
    image_pad,
    gamma,
    replay_start_size,
    capacity,
    gpu,
    batch_size,
    update_interval,
    is_persistent_buffer=False):

    utils.create_workspace(experiment_name)
    workspace_path = utils.get_workspace_path(experiment_name)

    encoder = Encoder(obs_space.shape)
    policy = PolicyFunction(encoder.repr_dim, action_space.shape, feature_dim, hidden_dim)
    q_func1 = QFunction(encoder.repr_dim, action_space.shape, feature_dim, hidden_dim)
    q_func2 = QFunction(encoder.repr_dim, action_space.shape, feature_dim, hidden_dim)

    encoder_optimizer = torch.optim.Adam(encoder.parameters(), lr=lr)
    policy_optimizer = torch.optim.Adam(policy.parameters(), lr=lr)
    q_func1_optimizer = torch.optim.Adam(q_func1.parameters(), lr=lr)
    q_func2_optimizer = torch.optim.Adam(q_func2.parameters(), lr=lr)

    if is_persistent_buffer:
        # rbuf = pfrl.replay_buffers.PersistentReplayBuffer(f'{str(workspace_path)}/rbuf', capacity, ancestor=f'{str(workspace_path)}/rbuf')
        rbuf = pfrl.replay_buffers.PersistentEpisodicReplayBuffer(f'{str(workspace_path)}/rbuf', capacity)
    else:
        rbuf = pfrl.replay_buffers.ReplayBuffer(capacity)

    if image_pad <= 0:
        phi = lambda x: x
    else:
        phi = RandomShiftsAug(image_pad)
        
    entropy_target = -action_space.shape[0]
    burnin_action_func = lambda: np.random.uniform(action_space.low, action_space.high).astype(np.float32)

    agent = DrQ(
        policy=policy,
        q_func1=q_func1,
        q_func2=q_func2,
        encoder=encoder,
        policy_optimizer=policy_optimizer,
        q_func1_optimizer=q_func1_optimizer,
        q_func2_optimizer=q_func2_optimizer,
        encoder_optimizer=encoder_optimizer,
        replay_buffer=rbuf,
        gamma=gamma,
        replay_start_size=replay_start_size,
        gpu=gpu,
        minibatch_size=batch_size,
        update_interval=update_interval,
        phi=phi,
        burnin_action_func=burnin_action_func,
        entropy_target=entropy_target,
        temperature_optimizer_lr=lr
    )

    return agent



######## Below Invalid Class ########

# class DiscreteQFunction(nn.Module):
#     """Critic"""
#     def __init__(self, repr_dim, action_n, feature_dim, hidden_dim):
#         super().__init__()
        
#         self.trunk = nn.Sequential(
#             nn.Linear(repr_dim, feature_dim),
#             nn.LayerNorm(feature_dim),
#             nn.Tanh()
#         )
#         self.q = utils.make_linear_layer(feature_dim, hidden_dim, action_n)
#         self.apply(utils.weight_init)

#     def forward(self, obs):
#         obs = self.trunk(obs)
#         return self.q(obs)

# class DiscretePolicyFunction(nn.Module):
#     """Actor"""
#     def __init__(self, repr_dim, action_n, feature_dim, hidden_dim):
#         super().__init__()

#         self.trunk = nn.Sequential(
#             nn.Linear(repr_dim, feature_dim),
#             nn.LayerNorm(feature_dim),
#             nn.Tanh()
#         )
#         self.policy = utils.make_linear_layer(feature_dim, hidden_dim, action_n)
#         self.policy.add_module('softmax', nn.Softmax(dim=1))
#         self.apply(utils.weight_init)

#     def forward(self, obs):
#         h = self.trunk(obs)
#         return self.policy(h)

# def make_DiscreteDrQ_agent(
#     experiment_name, 
#     obs_space, 
#     action_space, 
#     feature_dim, 
#     hidden_dim, 
#     lr, 
#     image_pad,
#     gamma,
#     replay_start_size,
#     capacity,
#     gpu,
#     batch_size,
#     update_interval,
#     is_persistent_buffer=False):

#     utils.create_workspace(experiment_name)
#     workspace_path = utils.get_workspace_path(experiment_name)

#     encoder = Encoder(obs_space.shape, 255)
#     policy = DiscretePolicyFunction(encoder.repr_dim, action_space.n, feature_dim, hidden_dim)
#     q_func1 = DiscreteQFunction(encoder.repr_dim, action_space.n, feature_dim, hidden_dim)
#     q_func2 = DiscreteQFunction(encoder.repr_dim, action_space.n, feature_dim, hidden_dim)

#     encoder_optimizer = torch.optim.Adam(encoder.parameters(), lr=lr)
#     policy_optimizer = torch.optim.Adam(policy.parameters(), lr=lr)
#     q_func1_optimizer = torch.optim.Adam(q_func1.parameters(), lr=lr)
#     q_func2_optimizer = torch.optim.Adam(q_func2.parameters(), lr=lr)

#     if is_persistent_buffer:
#         # rbuf = pfrl.replay_buffers.PersistentReplayBuffer(f'{str(workspace_path)}/rbuf', capacity, ancestor=f'{str(workspace_path)}/rbuf')
#         rbuf = pfrl.replay_buffers.PersistentEpisodicReplayBuffer(f'{str(workspace_path)}/rbuf', capacity)
#     else:
#         rbuf = pfrl.replay_buffers.ReplayBuffer(capacity)

#     if image_pad <= 0:
#         phi = lambda x: x
#     else:
#         phi = RandomShiftsAug(image_pad)
        
#     burnin_action_func = action_space.sample

#     agent = DiscreteDrQ(
#         policy=policy,
#         q_func1=q_func1,
#         q_func2=q_func2,
#         encoder=encoder,
#         policy_optimizer=policy_optimizer,
#         q_func1_optimizer=q_func1_optimizer,
#         q_func2_optimizer=q_func2_optimizer,
#         encoder_optimizer=encoder_optimizer,
#         replay_buffer=rbuf,
#         gamma=gamma,
#         replay_start_size=replay_start_size,
#         gpu=gpu,
#         minibatch_size=batch_size,
#         update_interval=update_interval,
#         phi=phi,
#         burnin_action_func=burnin_action_func,
#     )

#     return agent