import numpy as np
import d3rlpy
import os
from trifinger_tactile_learning import custom_encoders


path = "/is/sg2/iandrussow//cluster/fast/experiments/tactile_trifinger/2024_03_26_forcemap/crr/working_directories/0"

algo = d3rlpy.load_learnable(os.path.join(path,"algo_state.d3"))
algo.save_model(os.path.join(path,"policy.pt"))