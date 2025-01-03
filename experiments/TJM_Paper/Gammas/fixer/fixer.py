import pickle
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.legend_handler import HandlerTuple
import os
from matplotlib.gridspec import GridSpec
from matplotlib.colors import LogNorm
from matplotlib.ticker import LogLocator, NullFormatter

def combine_trajectories(dir):
    dir = dir + '/'
    data = []
    for filename in os.listdir(dir):
        filepath = dir + filename
        batch = pickle.load(open(filepath, "rb" ))
        for trajectory in batch['trajectories']:
            data.append(trajectory['expectation_values'])

    # [Trajectory, Expectation Values]
    return np.array(data)

heatmap = pickle.load(open("100L_T10_wall.pickle", "rb"))['heatmap']
heatmap_fixed = pickle.load(open("100L_T10_wall_mini.pickle", "rb"))['heatmap']

transposed_heatmap = heatmap.T
transposed_heatmap_fixed = heatmap_fixed.T

new_heatmap = []
start = 55
for i, _ in enumerate(transposed_heatmap):
    if i in range(start, start+10):
        new_heatmap.append(transposed_heatmap_fixed[i-start])
    else:
        new_heatmap.append(transposed_heatmap[i])


new_heatmap = np.array(new_heatmap)
new_heatmap = new_heatmap.T
im = plt.imshow(new_heatmap, aspect='auto', extent=(0, 100, 100, 0), vmin=-1, vmax=1)
filename = f"100L_T10_wall_fixed.pickle"
# with open(filename, 'wb') as f:
#     pickle.dump({
#         'heatmap': new_heatmap,
#     }, f)

plt.show()

