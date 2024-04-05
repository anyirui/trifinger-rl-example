"""Example policy for Real Robot Challenge 2022"""

import numpy as np
import torch

from trifinger_rl_datasets import PolicyBase, PolicyConfig

from . import policies


class NoHapticsPolicy(PolicyBase):

    _goal_order = ["object_keypoints", "object_position", "object_orientation"]

    def __init__(
        self,
        action_space,
        observation_space,
        episode_length,
    ):

        print("CUDA: ", torch.cuda.is_available())
        self.action_space = action_space
        self.device = "cuda"
        self.dtype = np.float32

        # load torch script
        torch_model_path = policies.get_model_path("lift.pt")
        self.policy = torch.jit.load(
            torch_model_path, map_location=torch.device(self.device)
        )
        self.policy.to(self.device)

    @staticmethod
    def get_policy_config():
        return PolicyConfig(
            flatten_obs=True,
            image_obs=False,
        )

    def reset(self):
        pass  # nothing to do here

    def get_action(self, observation):
        observation = torch.tensor(
            observation["robot_information"], dtype=torch.float, device=self.device
        )
        action = self.policy(torch.unsqueeze(observation, 0))
        action = action.detach().numpy()[0]
        action = np.clip(action, self.action_space.low, self.action_space.high)
        return action


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

    def get_action(self, observation):

        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()

        obs = np.concatenate(
            (
                observation["robot_information"],
                np.array(observation["haptic_information"]["force_maps"]).flatten(),
            ),
            axis=0,
        )
        obs = torch.tensor(obs, dtype=torch.float, device=self.device)

        action = self.policy(obs.unsqueeze(0))
        action = action.detach().numpy()[0]
        action = np.clip(action, self.action_space.low, self.action_space.high)

        end.record()
        torch.cuda.synchronize()
        print(start.elapsed_time(end))

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
