from torch import nn
import gymnasium as gym
import numpy as np
import tianshou as ts
import torch
import custom_env
from gymnasium.wrappers import FlattenObservation

num_training_envs = 10
num_test_envs = 100
env_id = "gym_examples/GridWorld-v0"


def make_env():
    return FlattenObservation(gym.make(env_id))


env = make_env()

train_envs = ts.env.ShmemVectorEnv([make_env for _ in range(num_training_envs)])
test_envs = ts.env.ShmemVectorEnv([make_env for _ in range(num_test_envs)])


class Net(nn.Module):
    def __init__(self, state_shape, action_shape):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(np.prod(state_shape), 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, np.prod(action_shape)),
        )

    def forward(self, obs, state=None, info={}):
        if not isinstance(obs, torch.Tensor):
            obs = torch.tensor(obs, dtype=torch.float)
        batch = obs.shape[0]
        logits = self.model(obs.view(batch, -1))
        return logits, state


state_shape = env.observation_space.shape or env.observation_space.n
action_shape = env.action_space.shape or env.action_space.n
net = Net(state_shape, action_shape)
optim = torch.optim.Adam(net.parameters(), lr=1e-3)

policy = ts.policy.DQNPolicy(
    model=net,
    optim=optim,
    action_space=env.action_space,
    discount_factor=0.9,
    estimation_step=3,
    target_update_freq=320,
)

train_collector = ts.data.Collector(
    policy,
    train_envs,
    ts.data.VectorReplayBuffer(20000, num_training_envs),
    exploration_noise=True,
)
test_collector = ts.data.Collector(policy, test_envs, exploration_noise=True)

result = ts.trainer.OffpolicyTrainer(
    policy=policy,
    train_collector=train_collector,
    test_collector=test_collector,
    max_epoch=10,
    step_per_epoch=1000,
    step_per_collect=10,
    update_per_step=0.1,
    episode_per_test=100,
    batch_size=64,
    train_fn=lambda epoch, env_step: policy.set_eps(0.1),
    test_fn=lambda epoch, env_step: policy.set_eps(0.05),
    stop_fn=lambda mean_rewards: mean_rewards >= env.spec.reward_threshold,
).run()
print(f'Finished training! Use {result["duration"]}')

torch.save(policy.state_dict(), "dqn.pth")
