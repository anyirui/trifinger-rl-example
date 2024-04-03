import numpy as np
import matplotlib.pyplot as plt
import os

folder = "../data/diffusion_position/"

#load the name of every .npy file in the specified folder

for filename in os.listdir(folder):
    if filename.endswith(".npy"):
        print(filename)
        #make a subplot for every new file
        fig, axs = plt.subplots(9,  1, figsize=(90, 20))
        try:
            action_buffer = np.load(folder + filename)
            for i in range(9):
                axs[i].plot(action_buffer[:, i], label=f"action {i}")
            plt.legend()
            plt.title(f"Episode {os.path.basename(filename).split('.')[0]}")
            plt.savefig(folder + os.path.basename(filename).split('.')[0]+".png")
        except Exception as e:
            print(e)
            pass