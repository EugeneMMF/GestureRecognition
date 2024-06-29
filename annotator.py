import json
import matplotlib.pyplot as plt
import numpy as np

with open("coordinates_data.txt", "r") as f:
    arrs = []
    for line in f:
        arrs.append(line.replace("\n", ''))
        arr = np.array(list(json.loads(line)))
        x = arr[:, 0]
        y = arr[:, 1]
        plt.plot(x[0], y[0], marker='o', color='red')
        plt.plot(x, y)
        plt.plot(x[-1], y[-1], marker='x', color='green')
        plt.ylim([0, 480])
        plt.xlim([0, 640])
        plt.show()
        inp = input('Enter the gesture: ')
        arrs[-1] += f", right-left\n"

with open("coordinates_data.txt", "w") as f:
    for line in arrs:
        f.write(line)
