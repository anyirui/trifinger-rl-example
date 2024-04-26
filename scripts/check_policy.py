from trifinger_rl_example.example import (
    ForceMapPolicy,
    RawImagePolicy,
    NoHapticsPolicy,
    ForceVecPolicy,
)
import torch
import numpy as np
import os

# os.nice(-10)


if __name__ == "__main__":

    input = {
        "robot_information": np.random.rand(139),
        "haptic_information": {
            "force_maps": torch.rand(1, 9, 40, 40),
            "raw_images": torch.rand(1, 9, 77, 102),
            "force_vecs": torch.rand(1, 3, 6),
        },
    }
    policy = RawImagePolicy(0, 0, 0)

    for i in range(20):
        policy.get_action(input)

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    timings = []

    for i in range(20):

        start.record()
        action = policy.get_action(input)
        end.record()
        torch.cuda.synchronize()
        timings.append(start.elapsed_time(end))

    mean = sum(timings) / len(timings)
    print("Mean: ", mean)
    print("Std: ", (sum((x - mean) ** 2 for x in timings) / len(timings)) ** 0.5)

    print("Max: ", max(timings))
    print("Min: ", min(timings))
