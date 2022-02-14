import os
from pathlib import Path
from hydra import compose, initialize_config_dir
from PIL import Image
import torch
from torch import nn, distributions

PARENT_DIR = Path('./work')

def load_param(conf_dir, config_name, overrides=[]):
    path = os.path.join(os.getcwd(), conf_dir)
    with initialize_config_dir(config_dir=path):
        cnf = compose(config_name=config_name, overrides=overrides)
        return cnf

def create_workspace(name) -> None:
    PARENT_DIR.mkdir(exist_ok=True)
    work = PARENT_DIR/Path(f'{name}')
    work.mkdir(exist_ok=True)
    print(f'set workspace: {work}')

def get_workspace_path(name) -> Path:
    return PARENT_DIR/Path(str(name))

def display_frames_as_gif(frames, filename='result.gif'):
    frs = [Image.fromarray(f, mode='RGBA') for f in frames]
    frs[0].save(f'./movie/{filename}', save_all=True, append_images=frs[1:], optimize=False, duration=100, loop=0)

def weight_init(m):
    if isinstance(m, nn.Linear):
        nn.init.orthogonal_(m.weight.data)
        if hasattr(m.bias, 'data'):
            m.bias.data.fill_(0.0)
    elif isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
        gain = nn.init.calculate_gain('relu')
        nn.init.orthogonal_(m.weight.data, gain)
        if hasattr(m.bias, 'data'):
            m.bias.data.fill_(0.0)

def conv2d_size_out(size, kernel_size=5, stride=2):
    return (size - (kernel_size - 1) - 1) // stride + 1

def make_linear_layer(linear_input_size, hidden_dim, out_dim):
    net = nn.Sequential(
        nn.Linear(linear_input_size, hidden_dim),
        nn.ReLU(inplace=True),
        nn.Linear(hidden_dim, hidden_dim),
        nn.ReLU(inplace=True),
        nn.Linear(hidden_dim, out_dim),
    )
    return net

def squashed_diagonal_gaussian_head(x):
    mean, log_scale = torch.chunk(x, 2, dim=1)
    log_scale = torch.clamp(log_scale, -20.0, 2.0)
    var = torch.exp(log_scale * 2)
    base_distribution = distributions.Independent(
        distributions.Normal(loc=mean, scale=torch.sqrt(var)), 1
    )
    # cache_size=1 is required for numerical stability
    return distributions.transformed_distribution.TransformedDistribution(
        base_distribution, [distributions.transforms.TanhTransform(cache_size=1)]
    )