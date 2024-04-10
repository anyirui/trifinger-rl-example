from trifinger_rl_example.example import ForceMapPolicy
import torch


if __name__ == "__main__":

    input = {
        "robot_information": torch.rand(139),
        "haptic_information": {"force_maps": torch.rand(1, 9, 40, 40)},
    }
    policy = ForceMapPolicy(0, 0, 0)

    for i in range(100):
        policy.get_action(input)

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    timings = []
    for i in range(100):

        start.record()
        action = policy.get_action(input)
        end.record()
        torch.cuda.synchronize()
        timings.append(start.elapsed_time(end))

    print("Mean: ", sum(timings) / len(timings))
