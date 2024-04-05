"""Example policy for Real Robot Challenge 2022"""

import numpy as np
import torch

from trifinger_rl_datasets import PolicyBase, PolicyConfig

from . import policies


class ForceMapPolicy(PolicyBase):

    _goal_order = ["object_keypoints", "object_position", "object_orientation"]

    def __init__(
        self,
        action_space,
        observation_space,
        episode_length,
    ):
        torch_model_path = "/is/sg2/iandrussow/training_results/2024_03_26_forcemap/crr/working_directories/0/policy.pt"
        self.action_space = action_space
        self.device = "cpu"
        self.dtype = np.float32

        # load torch script
        self.policy = torch.jit.load(
            torch_model_path, map_location=torch.device(self.device)
        )

    @staticmethod
    def get_policy_config():
        return PolicyConfig(
            flatten_obs=True,
            image_obs=False,
        )

    def reset(self):
        pass  # nothing to do here

    def get_action(self, observation, haptic_observation):

        print(observation)
        print(observation[0].shape)
        print(haptic_observation["force_maps"])
        print(np.array(observation[1]["force_maps"]).flatten().shape)
        obs = np.concatenate(
            (observation[0], np.array(observation[1]["force_maps"]).flatten()), axis=0
        )
        obs = torch.tensor(obs, dtype=torch.float, device=self.device)

        action = self.policy(obs.unsqueeze(0))
        action = action.detach().numpy()[0]
        action = np.clip(action, self.action_space.low, self.action_space.high)
        return action


class ForceVecPolicy(PolicyBase):

    _goal_order = ["object_keypoints", "object_position", "object_orientation"]

    def __init__(
        self,
        action_space,
        observation_space,
        episode_length,
    ):
        torch_model_path = policies.get_model_path("lift.pt")
        self.action_space = action_space
        self.device = "cpu"
        self.dtype = np.float32

        # load torch script
        self.policy = torch.jit.load(
            torch_model_path, map_location=torch.device(self.device)
        )

    @staticmethod
    def get_policy_config():
        return PolicyConfig(
            flatten_obs=True,
            image_obs=False,
        )

    def reset(self):
        pass  # nothing to do here

    def get_action(self, observation):
        observation = torch.tensor(observation, dtype=torch.float, device=self.device)
        action = self.policy(observation.unsqueeze(0))
        action = action.detach().numpy()[0]
        action = np.clip(action, self.action_space.low, self.action_space.high)
        return action


class RawImagePolicy(PolicyBase):

    _goal_order = ["object_keypoints", "object_position", "object_orientation"]

    def __init__(
        self,
        action_space,
        observation_space,
        episode_length,
    ):
        torch_model_path = policies.get_model_path("lift.pt")
        self.action_space = action_space
        self.device = "cpu"
        self.dtype = np.float32

        # load torch script
        self.policy = torch.jit.load(
            torch_model_path, map_location=torch.device(self.device)
        )

    @staticmethod
    def get_policy_config():
        return PolicyConfig(
            flatten_obs=True,
            image_obs=False,
        )

    def reset(self):
        pass  # nothing to do here

    def get_action(self, observation):
        observation = torch.tensor(observation, dtype=torch.float, device=self.device)
        action = self.policy(observation.unsqueeze(0))
        action = action.detach().numpy()[0]
        action = np.clip(action, self.action_space.low, self.action_space.high)
        return action
