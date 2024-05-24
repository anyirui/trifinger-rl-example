"""Example policy for Real Robot Challenge 2022"""

import numpy as np
import torch

from trifinger_rl_datasets import PolicyBase, PolicyConfig

from . import policies
import logging

import onnxruntime as ort

logging.basicConfig(level=logging.INFO)

# torch.backends.cudnn.benchmark = True


class ExpertPolicy(PolicyBase):

    _goal_order = ["object_keypoints", "object_position", "object_orientation"]

    def __init__(
        self,
        action_space,
        observation_space,
        episode_length,
    ):
        self.action_low_pass = 0.73

        print("CUDA: ", torch.cuda.is_available())
        self.action_space = action_space
        self.device = "cpu"
        self.dtype = np.float32

        # load torch script
        torch_model_path = "/is/sg2/iandrussow/trifinger_robot/trained_models/policy.pt"
        torch_model_path = policies.get_model_path("lift.pt")
        self.policy = torch.jit.load(
            torch_model_path, map_location=torch.device(self.device)
        )
        self.policy.to(self.device)

        self.last_action = None

        # self.ort_session = ort.InferenceSession(
        #     "/is/sg2/iandrussow/trifinger_robot/trained_models/2024_05_07_nohaptic_default/1_48M/policy.onnx"
        # )

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

        # action_target = self.ort_session.run(
        #     None, {"input_0": np.expand_dims(observation, axis=0)}
        # )[0][0]

        action_target = self.policy(observation.unsqueeze(0))

        if self.last_action is None:
            action = action_target
        else:
            action = (
                self.action_low_pass * self.last_action
                + (1.0 - self.action_low_pass) * action_target
            )

        self.last_action = action
        action = np.clip(action, self.action_space.low, self.action_space.high)

        return action


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
        self.device = "cpu"
        self.dtype = np.float32

        # # load torch script
        # torch_model_path = "/is/sg2/iandrussow/trifinger_robot/trained_models/2024_05_07_nohaptic_default/0/policy.pt"
        # torch_model_path = policies.get_model_path("lift.pt")
        # self.policy = torch.jit.load(
        #     torch_model_path, map_location=torch.device(self.device)
        # )
        # self.policy.to(self.device)

        self.ort_session = ort.InferenceSession(
            "/is/sg2/iandrussow/trifinger_robot/trained_models/2024_05_07_nohaptic_default/1_48M/policy.onnx"
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

        # start = torch.cuda.Event(enable_timing=True)
        # end = torch.cuda.Event(enable_timing=True)
        # start.record()

        observation = torch.tensor(
            observation["robot_information"], dtype=torch.float, device=self.device
        )
        # observation = torch.tensor(observation, dtype=torch.float, device=self.device)
        # action = self.policy(observation.unsqueeze(0))
        # action = self.policy(torch.unsqueeze(observation, 0))
        # action = action.detach().numpy()[0]
        # action = np.clip(action, self.action_space.low, self.action_space.high)

        action = self.ort_session.run(
            None, {"input_0": np.expand_dims(observation, axis=0)}
        )[0][0]
        action = np.clip(action, self.action_space.low, self.action_space.high)

        # end.record()
        # torch.cuda.synchronize()

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
        # torch_model_path = "/is/sg2/iandrussow/trifinger_robot/trained_models/2024_04_16_forcemap/policy.pt"
        # torch_model_path = "/home/andrussow/cluster/snagi/training_results/2024_03_26_forcemap/crr/working_directories/0/policy.pt"
        self.action_space = action_space
        self.device = "cpu"
        self.dtype = np.float32

        # load torch script
        # self.policy = torch.jit.load(
        #     torch_model_path, map_location=torch.device(self.device)
        # )
        # self.policy.to(torch.float)
        # print("Device: ", self.device)

        # logging.info("ORT device: ", ort.get_device())

        self.ort_session = ort.InferenceSession(
            "/is/sg2/iandrussow/trifinger_robot/trained_models/2024_05_08_forcemap/1/policy.onnx"
        )
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

        # logging.info(
        #     "Sensor delays:", str(observation["haptic_information"]["capture_delays"])
        # )
        # logging.info("Cube delay:", str(observation["robot_information"][34]))

        obs = torch.concat(
            (
                torch.tensor(observation["robot_information"]),
                torch.flatten(
                    torch.tensor(observation["haptic_information"]["force_maps"])
                ),
            ),
            axis=0,
        ).float()

        # logging.info(np.sum(observation["haptic_information"]["force_maps"]))

        # print(observation["haptic_information"]["force_maps"])
        # obs = obs.to(device=self.device)

        # start = torch.cuda.Event(enable_timing=True)
        # end = torch.cuda.Event(enable_timing=True)
        # start.record()

        # action = self.policy(torch.unsqueeze(obs, 0))
        # action = action.detach().cpu().numpy()[0]
        # action = np.clip(action, self.action_space.low, self.action_space.high)

        # end.record()
        # torch.cuda.synchronize()
        # print(start.elapsed_time(end))
        # self.timings.append(start.elapsed_time(end))
        # action = [-0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

        action = self.ort_session.run(None, {"input_0": np.expand_dims(obs, axis=0)})[
            0
        ][0]
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
        torch_model_path = "/is/sg2/iandrussow/trifinger_robot/trained_models/2024_03_01_forcevector/policy.pt"
        self.action_space = action_space
        self.device = "cpu"
        self.dtype = np.float32

        # # load torch script
        # self.policy = torch.jit.load(
        #     torch_model_path, map_location=torch.device(self.device)
        # )

        print("ORT device: ", ort.get_device())

        self.ort_session = ort.InferenceSession(
            "/is/sg2/iandrussow/trifinger_robot/trained_models/2024_03_01_forcevector/policy.onnx"
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

        obs = torch.concat(
            (
                torch.tensor(observation["robot_information"]),
                torch.flatten(
                    torch.tensor(observation["haptic_information"]["force_vecs"])
                ),
            ),
            axis=0,
        ).float()

        # obs = obs.to(device=self.device)
        action = self.ort_session.run(None, {"input_0": np.expand_dims(obs, axis=0)})[
            0
        ][0]

        # action = self.policy(obs.unsqueeze(0))
        # action = action.detach().cpu().numpy()[0]
        # action = np.clip(action, self.action_space.low, self.action_space.high)
        return action


class BinaryPolicy(PolicyBase):

    _goal_order = ["object_keypoints", "object_position", "object_orientation"]

    def __init__(
        self,
        action_space,
        observation_space,
        episode_length,
    ):
        # torch_model_path = "/is/sg2/iandrussow/trifinger_robot/trained_models/2024_05_07_binary/2/policy.pt"
        self.action_space = action_space
        self.device = "cpu"
        self.dtype = np.float32

        # load torch script
        # self.policy = torch.jit.load(
        #     torch_model_path, map_location=torch.device(self.device)
        # )

        # print("ORT device: ", ort.get_device())
        # print("Running Binary Policy")

        self.ort_session = ort.InferenceSession(
            "/is/sg2/iandrussow/trifinger_robot/trained_models/2024_05_07_binary/2/policy.onnx"
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

        # logging.info(
        #     np.linalg.norm(
        #         observation["haptic_information"]["force_vecs"][:, 3:],
        #         axis=1,
        #     )
        #     > 0.05
        # )

        # logging.info(
        #     np.linalg.norm(
        #         observation["haptic_information"]["force_vecs"][:, 3:],
        #         axis=1,
        #     )
        # )

        obs = torch.concat(
            (
                torch.tensor(observation["robot_information"]),
                torch.flatten(
                    torch.tensor(
                        np.linalg.norm(
                            observation["haptic_information"]["force_vecs"][:, 3:],
                            axis=1,
                        )
                        > 0.05
                    )
                ),
            ),
            axis=0,
        ).float()

        # obs = obs.to(device=self.device)
        action = self.ort_session.run(None, {"input_0": np.expand_dims(obs, axis=0)})[
            0
        ][0]

        # action = self.policy(obs.unsqueeze(0))
        # action = action.detach().cpu().numpy()[0]

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
        torch_model_path = "/is/sg2/iandrussow/trifinger_robot/trained_models/2024_04_29_raw_image_resnet9/policy.pt"
        # torch_model_path = "/is/sg2/iandrussow/trifinger_robot/trained_models/test_models/convmixer/policy.pt"
        self.action_space = action_space
        self.device = "cpu"
        self.dtype = np.float32

        # self.policy = torch.jit.load(
        #     torch_model_path, map_location=torch.device(self.device)
        # )
        # self.policy.to(torch.float)
        # print("ORT device: ", ort.get_device())

        self.ort_session = ort.InferenceSession(
            "/is/sg2/iandrussow/trifinger_robot/trained_models/test_models/small_encoder/policy.onnx"
            # "/is/sg2/iandrussow/trifinger_robot/trained_models/2024_04_29_raw_image_resnet9/policy.onnx"
        )

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

        obs = torch.concat(
            (
                torch.tensor(observation["robot_information"]),
                torch.flatten(
                    torch.tensor(observation["haptic_information"]["raw_images"])
                ),
            ),
            axis=0,
        ).float()

        obs = obs.to(device=self.device)

        action = self.ort_session.run(None, {"input_0": np.expand_dims(obs, axis=0)})[0]

        # action = self.policy(obs.unsqueeze(0))
        # action = action.detach().cpu().numpy()[0]
        # action = np.clip(action, self.action_space.low, self.action_space.high)

        return action
