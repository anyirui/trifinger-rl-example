"""Example policy for Real Robot Challenge 2022"""

import numpy as np
import torch

from trifinger_rl_datasets import PolicyBase, PolicyConfig

from . import policies
import logging

# import onnxruntime as ort

logging.basicConfig(level=logging.INFO)


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

        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()

        observation = torch.tensor(
            observation["robot_information"], dtype=torch.float, device=self.device
        )
        action = self.policy(torch.unsqueeze(observation, 0))
        action = action.detach().numpy()[0]
        # action = np.clip(action, self.action_space.low, self.action_space.high)

        end.record()
        torch.cuda.synchronize()

        return action


class ForceMapPolicy(PolicyBase):

    _goal_order = ["object_keypoints", "object_position", "object_orientation"]

    def __init__(
        self,
        action_space,
        observation_space,
        episode_length,
    ):
        print("CUDA: ", torch.cuda.is_available())
        torch_model_path = "/is/sg2/iandrussow/trifinger_robot/trained_models/2024_04_16_forcemap/policy.pt"
        # torch_model_path = "/home/andrussow/cluster/snagi/training_results/2024_03_26_forcemap/crr/working_directories/0/policy.pt"
        self.action_space = action_space
        self.device = "cuda"
        self.dtype = np.float32

        # load torch script
        self.policy = torch.jit.load(
            torch_model_path, map_location=torch.device(self.device)
        )
        self.policy.to(torch.float)

        print("Device: ", self.device)

        # print("ORT device: ", ort.get_device())

        # self.ort_session = ort.InferenceSession(
        #     "/is/sg2/iandrussow/trifinger_robot/trained_models/2024_04_16_forcemap/policy.onnx"
        # )
        self.timings = []

    @staticmethod
    def get_policy_config():
        return PolicyConfig(
            flatten_obs=True,
            image_obs=False,
        )

    def get_timing(self):
        if len(self.timings) > 0:
            msg = f"Mean timing of inference in the last episode: {sum(self.timings) / len(self.timings)}"
            self.timings = []
        else:
            msg = "No timing information available"
        return msg

    def reset(self):
        pass

    def get_action(self, observation):

        obs = torch.concat(
            (
                torch.tensor(observation["robot_information"]),
                torch.flatten(observation["haptic_information"]["force_maps"]),
            ),
            axis=0,
        ).float()

        obs = obs.to(device=self.device)

        # start = torch.cuda.Event(enable_timing=True)
        # end = torch.cuda.Event(enable_timing=True)
        # start.record()

        action = self.policy(torch.unsqueeze(obs, 0))
        action = action.detach().cpu().numpy()[0]
        action = np.clip(action, self.action_space.low, self.action_space.high)

        # end.record()
        # torch.cuda.synchronize()
        # print(start.elapsed_time(end))
        # self.timings.append(start.elapsed_time(end))
        action = [-0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

        # action = self.ort_session.run(None, {"input_0": np.expand_dims(obs, axis=0)})[0]
        return action


class ForceVecPolicy(PolicyBase):

    _goal_order = ["object_keypoints", "object_position", "object_orientation"]

    def __init__(
        self,
        action_space,
        observation_space,
        episode_length,
    ):
        torch_model_path = torch_model_path = (
            "/is/sg2/iandrussow/trifinger_robot/trained_models/2024_03_01_forcevector/policy.pt"
        )
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

        obs = np.concatenate(
            (observation, haptic_observation["force_vecs"].flatten()), axis=1
        )
        obs = torch.tensor(obs, dtype=torch.float, device=self.device)

        action = self.policy(obs.unsqueeze(0))
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
        print("CUDA: ", torch.cuda.is_available())
        torch_model_path = "/is/sg2/iandrussow/trifinger_robot/trained_models/2024_04_16_raw_image/policy.pt"
        self.action_space = action_space
        self.device = "cuda"
        self.dtype = np.float32

        # load torch script
        self.policy = torch.jit.load(
            torch_model_path, map_location=torch.device(self.device)
        )
        self.policy.to(torch.float)
        # print("ORT device: ", ort.get_device())

        # self.ort_session = ort.InferenceSession(
        #     "/is/sg2/iandrussow/trifinger_robot/trained_models/2024_04_16_raw_image/policy.onnx"
        # )

        self.timings = []

    @staticmethod
    def get_policy_config():
        return PolicyConfig(
            flatten_obs=True,
            image_obs=False,
        )

    def get_timing(self):
        if len(self.timings) > 0:
            print(
                "Mean timing of inference in the last episode: ",
                sum(self.timings) / len(self.timings),
            )
            self.timings = []

    def reset(self):
        pass

    def get_action(self, observation):

        obs = np.concatenate(
            (
                observation["robot_information"],
                np.array(observation["haptic_information"]["raw_image"]).flatten(),
            ),
            axis=0,
        )
        obs = torch.tensor(obs, dtype=torch.float, device=self.device)

        # action = self.ort_session.run(None, {"input_0": np.expand_dims(obs, axis=0)})[0]

        action = self.policy(obs.unsqueeze(0))
        action = action.detach().cpu().numpy()[0]
        # action = np.clip(action, self.action_space.low, self.action_space.high)

        return action
