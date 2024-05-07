from torch import nn
import gymnasium as gym
import numpy as np
import tianshou as ts
import torch
import patrol_scheduling
from gymnasium.wrappers import FlattenObservation

num_training_envs = 10
num_test_envs = 10
env_id = "paws/PatrolScheduling-v0"


def make_env():
    return FlattenObservation(gym.make(env_id))


env = make_env()

train_envs = ts.env.DummyVectorEnv([make_env for _ in range(num_training_envs)])
test_envs = ts.env.DummyVectorEnv([make_env for _ in range(num_test_envs)])
# train_envs = ts.env.ShmemVectorEnv([make_env for _ in range(num_training_envs)])
# test_envs = ts.env.ShmemVectorEnv([make_env for _ in range(num_test_envs)])


class Actor(nn.Module):
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
            nn.Softmax(dim=1)
        )

    def forward(self, obs, state=None, info={}):
        obs = torch.as_tensor(obs, dtype=torch.float)
        batch = obs.shape[0]
        logits = self.model(obs.view(batch, -1))
        return logits, state


class Critic(nn.Module):
    def __init__(self, obs_shape, action_shape):
        super().__init__()
        self.input_shape = np.prod(obs_shape) + np.prod(action_shape)
        self.model = nn.Sequential(
            nn.Linear(self.input_shape, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 1),
        )

    def forward(self, obs, act):
        obs = torch.as_tensor(obs, dtype=torch.float)
        obs = obs.flatten(1)
        act = torch.as_tensor(act, dtype=torch.float)
        act = act.flatten(1)
        obs = torch.cat([obs, act], dim=1)
        batch = obs.shape[0]
        q_value = self.model(obs.view(batch, -1))
        return q_value


state_shape = env.observation_space.shape
action_shape = env.action_space.shape
print(f"{state_shape=}", f"{action_shape=}")
actor = Actor(state_shape, action_shape)
actor_optim = torch.optim.Adam(actor.parameters(), lr=1e-3)
critic = Critic(state_shape, action_shape)
critic_optim = torch.optim.Adam(critic.parameters(), lr=1e-3)

policy = ts.policy.DDPGPolicy(
    actor=actor,
    actor_optim=actor_optim,
    critic=critic,
    critic_optim=critic_optim,
    action_space=env.action_space,
    observation_space=env.observation_space,
    action_scaling=False,
    exploration_noise=False,
)

train_collector = ts.data.Collector(
    policy,
    train_envs,
    ts.data.VectorReplayBuffer(20000, num_training_envs),
    exploration_noise=False,
)
test_collector = ts.data.Collector(policy, test_envs, exploration_noise=False)

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
).run()
print(f'Finished training! Use {result["duration"]}')

torch.save(policy.state_dict(), "dqn.pth")
