import d3rlpy
import argparse
from trifinger_tactile_learning import custom_encoders
import torch

if __name__ == "__main__":

    device = "cpu"
    argparser = argparse.ArgumentParser()
    argparser.add_argument("algo_path", type=str)
    args = argparser.parse_args()

    algo = d3rlpy.load_learnable(args.algo_path + "algo_state.d3", device=device)
    # print(args.algo_path + "policy.pt")
    algo.save_policy(args.algo_path + "policy.pt")

    print("Save onnx policy")
    algo.save_policy(args.algo_path + "policy.onnx")
    # dummy_x = torch.rand(1, *algo.observation_shape, device=device)
    # jitted_model = torch.jit.load(args.algo_path + "policy.pt")
    # torch.onnx.export(
    #     jitted_model,
    #     dummy_x,
    #     args.algo_path + "policy.onnx",
    #     opset_version=14,
    # )
