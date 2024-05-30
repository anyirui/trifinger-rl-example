"""Example policy for Real Robot Challenge 2022"""

import numpy as np
import torch

from trifinger_rl_datasets import PolicyBase, PolicyConfig

from . import policies


class TeleopPolicy(PolicyBase):

    def __init__(
        self,
        action_space,
        observation_space,
        episode_length,
    ):
        self.action_space = action_space
        self.observation_space = observation_space

    @staticmethod
    def get_policy_config():
        return PolicyConfig(
            flatten_obs=True,
            image_obs=False,
        )

    def reset(self):
        pass  # nothing to do here

    # In this function, the policy should receive the current command for the robot joint angles from the user. It will be called every 20ms to update the current robot joints angles.
    def get_action(self, observation):

        self.stream_observation(observation)
        action = np.zeros(9)

        action = np.clip(action, self.action_space.low, self.action_space.high)

        return action

    # This function streams back the current observation to the user
    def stream_observation(self, observation):

        # This might be unnecessary, we can also just access the webcam image directly with the adjusted fish-eye camera script or do the through-vision with the headset as discussed before
        pass
