from trifinger_rl_example.example import ForceMapPolicy, RawImagePolicy, NoHapticsPolicy
import torch


if __name__ == "__main__":

    input = {
        "robot_information": torch.rand(139),
        "haptic_information": {
            "force_maps": torch.rand(1, 9, 40, 40),
            "raw_image": torch.rand(1, 9, 77, 102),
        },
    }
    policy = ForceMapPolicy(0, 0, 0)

    for i in range(200):
        policy.get_action(input)

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    timings = []

    for i in range(1000):

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
