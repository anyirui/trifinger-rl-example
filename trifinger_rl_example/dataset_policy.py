import h5py

import torch
import numpy as np
from trifinger_rl_datasets import PolicyBase, PolicyConfig

class DatasetBasePolicy(PolicyBase):

    _goal_order = ["object_keypoints", "object_position", "object_orientation"] 

    def __init__(
        self,
        dataset_path,
        observation_space,
        episode_length,
    ):
        hf = h5py.File(dataset_path, "r")

        # first get episode lengths
        n_timesteps = 0

        self.episodes = list(hf.keys())[:50]
        for episode in self.episodes: 
            if n_timesteps == 0:

                act_shape = hf[episode]["actions"].shape[1]
            n_timesteps += hf[episode]["flat_obs"].shape[0]

        self.actions = []

        self.ep = np.random.randint(1, len(self.episodes))
        self.timestep = -1

        
        index = 0
        for episode in self.episodes:

            episode_group = hf[episode]
            self.actions.append(episode_group["flat_obs"][1:, 111:120])



    @staticmethod
    def get_policy_config():
        return PolicyConfig(
            flatten_obs=True,
            image_obs=False,
        )

    def reset(self):
        self.ep = np.random.randint(1, len(self.episodes))
        self.timestep = -1
        
        

    def get_action(self, observation):
        self.timestep +=1
        action = self.actions[self.ep][self.timestep]
        print(action)
        return action


class DatasetLiftPolicy(DatasetBasePolicy):
    """Example policy for the lift task, using a torch model to provide actions.

    Expects flattened observations.
    """

    def __init__(self, action_space, observation_space, episode_length, model_path=None):
        
        model = ("/home/andrussow/cluster/fast/trifinger_data/data/collection1/processed_data/dataset.hdf5")
        # stats = policies.get_model_path("lift_diffusion_stats.npy") #TODO put this into the model file too!
        super().__init__(model, observation_space, episode_length)