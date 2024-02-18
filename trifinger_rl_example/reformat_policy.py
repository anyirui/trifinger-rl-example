import d3rlpy
import argparse


if __name__ == "__main__":

    argparser = argparse.ArgumentParser()
    argparser.add_argument("algo_path", type=str)
    args = argparser.parse_args()


    algo = d3rlpy.load_learnable(args.algo_path)
    algo.save_policy("policies/lift.pt")