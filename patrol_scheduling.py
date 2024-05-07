"""A custom environment for the patrol scheduling problem."""

from dataclasses import dataclass
from typing import Callable, Optional

from gymnasium import spaces
from gymnasium.envs.registration import register
from scipy.ndimage import generic_filter
from scipy.stats import logistic
import gymnasium as gym
import numpy as np


@dataclass
class PatrolProblem:
    """A patrol problem is defined by a grid representing a discretized map, and values
    that specify the wildlife activity at each grid cell, along with parameters related
    to the mathematical model of the problem."""

    # A 2D array of floats representing wildlife activity in the park.
    wildlife: np.ndarray

    # The total budget available to execute patrols.
    total_budget: float

    # learned poacher & environment model parameters

    # A parameter describing the natural growth rate of wildlife, constrained
    # to be greater than 1. Referred to as psi in the paper, see equation (6).
    wildlife_growth_ratio: float

    # A parameter describing the strength of poachers on reducing wildlife,
    # constrained to be greater than 0. Referred to as alpha in the paper, see
    # equation (6).
    poacher_strength: float

    # A parameter describing the effect of patrolling on reducing future
    # poaching. Less than zero if patrolling deters poaching. Referred to as
    # beta in the paper, see equations (3, 5).
    return_on_effort: float

    # A parameter describing the effect of patrolling on poaching in
    # neighboring cells. Greater than zero if patrolling incentivizes poachers
    # to poach elsewhere. Referred to as eta in the paper, see equation (5).
    displacement_effect: float


@dataclass
class PoacherPolicy:
    # A 2D array of floats representing how attractive a grid cell is for
    # poachers to attack. Referred to as z_i, and the entire set as Z. See
    # equation (5).
    #
    # Zero attractiveness corresponds to the middle range: equally likely
    # as not to place a snare there.
    #
    # Negative attractiveness crresponds to disincentive to place a snare
    # there. The values are passed through the logistic function.
    attractiveness: np.ndarray


@dataclass
class DefenderPolicy:
    # A 2D array of values in [0,1] representing how much effort to spend
    # patrolling a given grid cell. Referred to as a_i in the paper, this
    # class represents the policy for a single time step.
    patrol_effort: np.ndarray


def build_attack_model(
    poacher_policy: PoacherPolicy,
    patrol_problem: PatrolProblem,
    defender_policy: DefenderPolicy,
) -> np.ndarray:
    """Computes the probability of poaching activity in each cell."""
    footprint = np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]])
    # computes the sum of values of neighbors of each cell, using zero for
    # values that go outside the boundary of the array.
    neighbor_effort = generic_filter(
        defender_policy.patrol_effort,
        np.sum,
        footprint=footprint,
        mode="constant",
        cval=0,
    )
    logistic_input = (
        poacher_policy.attractiveness
        + patrol_problem.return_on_effort * defender_policy.patrol_effort
        + patrol_problem.displacement_effect * neighbor_effort
    )
    return logistic.cdf(logistic_input)


def draw_poacher_activity(attack_model: np.ndarray) -> np.ndarray:
    return np.random.binomial(1, attack_model)


def update_wildlife(
    wildlife: np.ndarray,
    poacher_activity: np.ndarray,
    defender_policy: DefenderPolicy,
    patrol_problem: PatrolProblem,
) -> np.ndarray:
    """Returns the new wildlife population in each cell."""
    # Note the input is the _current_ patrol effort vs the _past_ poacher
    # activity. Snares were set in the past, but might be detected by current
    # patrol.
    manmade_change = -(
        patrol_problem.poacher_strength
        * poacher_activity
        * (1 - defender_policy.patrol_effort)
    )
    natural_growth = wildlife**patrol_problem.wildlife_growth_ratio
    return np.maximum(0, natural_growth + manmade_change)


class PatrolProblemEnv(gym.Env):
    metadata = {"render_modes": [], "render_fps": 4}

    def __init__(
        self,
        patrol_problem,
        draw_poacher_policy: Callable[[], PoacherPolicy],
        planning_horizon: int = 10,
        render_mode=None,
    ):
        self._patrol_problem = patrol_problem
        self._step = 0
        # this implies there's a fixed poacher strategy for the entire game
        self._draw_poacher_policy = draw_poacher_policy
        self.shape = patrol_problem.wildlife.shape
        num_rows, num_cols = self.shape
        self.observation_space = spaces.Dict(
            {
                # Wildlife population in each cell
                "wildlife": spaces.Box(
                    low=np.zeros(shape=self.shape),
                    high=np.broadcast_to(np.array([np.inf]), self.shape),
                    dtype=np.float64,
                ),
                # Poacher activity in each cell
                "poacher_activity": spaces.MultiBinary(self.shape),
            }
        )

        # How much effort to patrol in each cell. Rescaled to [0, B] where B is
        # the total budget available to the patrollers.
        # Flattened to match output of neural network
        self.action_space = spaces.Box(
            low=0, high=1, shape=(np.prod(self.shape),), dtype=np.float64
        )

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode

    def _get_obs(self):
        return {
            "wildlife": self._wildlife,
            "poacher_activity": self._poacher_activity,
        }

    def _get_info(self):
        return {}

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self._wildlife = self._patrol_problem.wildlife
        # no poacher activity in initial state
        self._poacher_activity = np.zeros(self.shape)

        observation = self._get_obs()
        info = self._get_info()

        return observation, info

    def step(self, action):
        wildlife = self._wildlife
        action = action.reshape(self.shape)
        patrol_effort = action * self._patrol_problem.total_budget
        patrol_effort[np.where(patrol_effort > 1)] = 1
        defender_policy = DefenderPolicy(patrol_effort)

        assert np.sum(patrol_effort) <= self._patrol_problem.total_budget + 1e-05
        # , (
        #     f"Total patrol effort {np.sum(defender_policy.patrol_effort)} exceeds "
        #     f"budget {patrol_problem.total_budget}"
        # )

        poacher_policy = self._draw_poacher_policy()
        p_attack = build_attack_model(
            poacher_policy=poacher_policy,
            patrol_problem=self._patrol_problem,
            defender_policy=defender_policy,
        )
        poacher_activity = draw_poacher_activity(p_attack)
        self._poacher_activity = poacher_activity

        next_wildlife = update_wildlife(
            wildlife,
            poacher_activity,
            defender_policy,
            self._patrol_problem,
        )

        # use the expected wildlife at this step as an intermediate reward
        reward = np.sum(
            update_wildlife(wildlife, p_attack, defender_policy, self._patrol_problem),
        )

        terminated = np.all(np.isclose(self._wildlife, np.zeros(self.shape)))
        observation = self._get_obs()
        info = self._get_info()
        truncated = False

        self._wildlife = next_wildlife
        self._step += 1

        return observation, reward, terminated, truncated, info

    def render(self):
        pass

    def close(self):
        pass


shape = (5, 5)
wildlife = np.zeros(shape)
wildlife[2][2:5] = [7, 7, 7]
wildlife[3][2:5] = [7, 9, 7]
wildlife[4][2:5] = [7, 7, 7]

# exaggerated for test
patrol_problem = PatrolProblem(
    wildlife=wildlife,
    total_budget=1,
    wildlife_growth_ratio=1.0,
    poacher_strength=0.5,
    return_on_effort=-0.5,
    displacement_effect=0.0,
)

# Initial poacher strategies should be "avoid everywhere" except
# in the locations overridden by the test-specific data
strat = -10 * np.ones(shape=shape)
strat[2][2:5] = [10, 10, 10]
strat[3][2:5] = [10, 10, 10]
strat[4][2:5] = [10, 10, 10]
training_rounds = 400
planning_horizon = 100


def draw_poacher_policy() -> PoacherPolicy:
    sampled_poacher_policy = strat
    return PoacherPolicy(sampled_poacher_policy)


register(
    id="paws/PatrolScheduling-v0",
    entry_point="patrol_scheduling:PatrolProblemEnv",
    max_episode_steps=planning_horizon,
    reward_threshold=5000,
    kwargs={
        "patrol_problem": patrol_problem,
        "draw_poacher_policy": draw_poacher_policy,
        "planning_horizon": planning_horizon,
    },
)

env = gym.wrappers.FlattenObservation(gym.make("paws/PatrolScheduling-v0"))
