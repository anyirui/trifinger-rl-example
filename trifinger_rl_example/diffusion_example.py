"""Example policy for Real Robot Challenge 2022"""

import numpy as np
import torch
import collections

from trifinger_rl_datasets import PolicyBase, PolicyConfig
from trifinger_tactile_learning.custom_algorithms import ConditionalUnet1DState
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler

import time

from . import policies


def normalize_data(data, stats):
    # nomalize to [0,1]
    ndata = (data - stats["min"]) / (stats["max"] - stats["min"])
    # normalize to [-1, 1]
    ndata = ndata * 2 - 1
    return ndata


def unnormalize_data(ndata, stats):
    ndata = (ndata + 1) / 2
    data = ndata * (stats["max"] - stats["min"]) + stats["min"]
    return data


def interpolate_data(array):
    upsampled_actions = []
    for i in range(len(array) - 1):
        linfit = interp1d([1, 6], np.vstack([array[i], array[i + 1]]), axis=0)
        linfit_ = linfit([1, 2, 3, 4, 5])
        upsampled_actions.extend(linfit_)
    return upsampled_actions


class DiffusionBasePolicy(PolicyBase):

    _goal_order = ["object_keypoints", "object_position", "object_orientation"]

    def __init__(
        self,
        torch_model_path,
        action_space,
        observation_space,
        episode_length,
        obs_horizon=2,
        action_horizon=4,
        pred_horizon=8,
        action_dim=9,
        obs_dim=138,
        num_diffusion_iters=16,
    ):
        self.action_space = action_space
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print("Using device: ", self.device)
        self.dtype = np.float32
        self.num_diffusion_iters = num_diffusion_iters
        self.pred_horizon = pred_horizon
        self.obs_horizon = obs_horizon
        self.action_horizon = action_horizon
        self.action_dim = action_dim

        model_dict = torch.load(
            torch_model_path, map_location=torch.device(self.device)
        )

        self.stats = model_dict.get("data_stats")
        # Remove column 120 from the observation space
        self.stats["obs"]["max"] = np.delete(self.stats["obs"]["max"], 120)
        self.stats["obs"]["min"] = np.delete(self.stats["obs"]["min"], 120)

        self.action = []
        self.obs_deque = collections.deque(
            [torch.zeros(obs_dim)] * obs_horizon, maxlen=obs_horizon
        )

        self.noise_scheduler = DDPMScheduler(
            num_train_timesteps=num_diffusion_iters,
            # the choise of beta schedule has big impact on performance
            # we found squared cosine works the best
            beta_schedule="squaredcos_cap_v2",
            # clip output to [-1,1] to improve stability
            clip_sample=True,
            # our network predicts noise (instead of denoised action)
            prediction_type="epsilon",
        )

        noise_pred_net = ConditionalUnet1DState(
            input_dim=action_dim, global_cond_dim=obs_dim * obs_horizon
        )

        self.ema_noise_pred_net = noise_pred_net
        self.ema_noise_pred_net.load_state_dict(model_dict.get("model_state_dict"))
        self.ema_noise_pred_net.to(self.device)
        self.ema_noise_pred_net = self.ema_noise_pred_net.eval()

    @staticmethod
    def get_policy_config():
        return PolicyConfig(
            flatten_obs=True,
            image_obs=False,
        )

    def reset(self):
        pass  # nothing to do here

    def get_action(self, observation):

        start = time.time()

        obs_new = observation[:120]
        observation = np.concatenate((obs_new, observation[121:]), axis=0)

        self.obs_deque.append(observation)

        if len(self.action) > 0:
            pass
        else:
            B = 1
            # stack the last obs_horizon (2) number of observations
            obs_seq = np.stack(self.obs_deque)

            # normalize observation
            nobs = normalize_data(obs_seq, stats=self.stats["obs"])
            # device transfer
            nobs = torch.from_numpy(nobs).to(self.device, dtype=torch.float32)

            # infer action
            with torch.no_grad():
                # reshape observation to (B,obs_horizon*obs_dim)
                obs_cond = nobs.unsqueeze(0).flatten(start_dim=1)

                # initialize action from Guassian noise
                noisy_action = torch.randn(
                    (B, self.pred_horizon, self.action_dim), device=self.device
                )
                naction = noisy_action

                # init scheduler
                self.noise_scheduler.set_timesteps(self.num_diffusion_iters)

                for k in self.noise_scheduler.timesteps:
                    # predict noise
                    noise_pred = self.ema_noise_pred_net(
                        sample=naction, timestep=k, global_cond=obs_cond
                    )

                    # inverse diffusion step (remove noise)
                    naction = self.noise_scheduler.step(
                        model_output=noise_pred, timestep=k, sample=naction
                    ).prev_sample

            # unnormalize action
            naction = naction.detach().to("cpu").numpy()
            # (B, pred_horizon, action_dim)
            naction = naction[0]
            action_pred = unnormalize_data(naction, stats=self.stats["action"])

            # only take action_horizon number of actions
            start = self.obs_horizon - 1
            end = start + self.action_horizon
            actions = list(action_pred[start:end, :])

            # interpolate between data points to upsample to 50Hz again
            actions = np.array(actions)
            self.action = interpolate_data(actions)

        action = self.action.pop(0)
        print(action)
        action = np.clip(action, self.action_space.low, self.action_space.high)

        end = time.time()

        print("Time to get action: ", end - start)
        return action


class DiffusionLiftPolicy(DiffusionBasePolicy):
    """Example policy for the lift task, using a torch model to provide actions.

    Expects flattened observations.
    """

    def __init__(
        self, action_space, observation_space, episode_length, model_path=None
    ):
        if model_path is not None:
            model = policies.get_model_path(model_path)
        else:
            model = policies.get_model_path("lift_diffusion_subsampled_cropped_10Hz.pt")
        super().__init__(model, action_space, observation_space, episode_length)
