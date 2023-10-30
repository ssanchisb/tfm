import networkx as nx
import pandas as pd

am_df = pd.read_csv('/home/vant/code/clonemmri/data/FA/0001.csv', header=None)

G = nx.Graph()

num_nodes = am_df.shape[0]

for i in range(num_nodes):
    for j in range(i + 1, num_nodes):
        weight = am_df.iloc[i, j]
        if weight != 0:
            G.add_edge(i, j, weight=weight)



print(G)

import matplotlib.pyplot as plt

nx.draw(G, with_labels=True)
plt.show()
