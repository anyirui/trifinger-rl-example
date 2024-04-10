import numpy as np
import d3rlpy
import os
from trifinger_tactile_learning import custom_encoders


path = (
    "/home/iandrussow/trained_models/2024_03_01_forcevector/crr/working_directories/2"
)

algo = d3rlpy.load_learnable(os.path.join(path, "algo_state.d3"))
algo.save_model(os.path.join(path, "policy.pt"))
