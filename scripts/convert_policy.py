import numpy as np
import d3rlpy
import os


path = "/is/cluster/fast/iandrussow/experiments/tactile_trifinger/2024_01_27_lift_no_haptic_default/crr_dev/working_directories/4/"

algo = d3rlpy.load_learnable(os.path.join(path,"algo_state.d3"))
algo.save_model(os.path.join(path,"algo_state.pt"))