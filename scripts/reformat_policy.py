import d3rlpy
import argparse
from trifinger_tactile_learning import custom_encoders
import torch

if __name__ == "__main__":

    argparser = argparse.ArgumentParser()
    argparser.add_argument("algo_path", type=str)
    args = argparser.parse_args()

    algo = d3rlpy.load_learnable(args.algo_path + "algo_state.d3", device="cuda")
    print(args.algo_path + "policy.pt")
    algo.save_policy(args.algo_path + "policy.pt")
    #algo.save_policy(args.algo_path + "policy.onnx")
