import os.path
import networkx as nx
import glob
import pandas as pd
import matplotlib.pyplot as plt


FA_matrices = [pd.read_csv(file, header=None) for file in
               glob.glob(os.path.join('/home/vant/code/clonemmri/data/FA', "*.csv"))]

# print(len(FA_matrices))
# print(avg_all_m)

# extract labels of MS vs. HV:
#MS_labels = pd.read_csv('/home/vant/code/clonemmri/data/demographics.csv')
labels = MS_labels['mstype'].tolist()

print(len(MS_labels))
print(len(labels))
print(labels.count(0))

# use labels to separate into 2 groups:
MS_FA = [FA_matrices[i] for i, value in enumerate(labels) if value == 0]
HV_FA = [FA_matrices[i] for i, value in enumerate(labels) if value == -1]
print(len(MS_FA))
print(len(HV_FA))

# Create average matrices for each of the 2 groups:
avg_fa_ms = pd.concat(MS_FA).groupby(level=0).mean()
avg_fa_hv = pd.concat(HV_FA).groupby(level=0).mean()

# extract a matrix of a single ms patient:
ms1 = pd.read_csv('/home/vant/code/clonemmri/data/FA/0001.csv', header=None)

# print(avg_fa_ms)

# create graphs:
G = nx.Graph()
H = nx.Graph()
P1 = nx.Graph()

num_nodes = avg_fa_ms.shape[0]

# populate graphs with weights:
for i in range(num_nodes):
    for j in range(i + 1, num_nodes):
        weight = avg_fa_ms.iloc[i, j]
        if weight != 0:
            G.add_edge(i, j, weight=weight)

for i in range(num_nodes):
    for j in range(i + 1, num_nodes):
        weight = avg_fa_hv.iloc[i, j]
        if weight != 0:
            H.add_edge(i, j, weight=weight)

for i in range(num_nodes):
    for j in range(i + 1, num_nodes):
        weight = ms1.iloc[i, j]
        if weight != 0:
            P1.add_edge(i, j, weight=weight)

print(H)
print(P1)

# define missing edges (difference between healthy and (avg)patient):
missing_edges = [(u, v) for u, v in H.edges() if not P1.has_edge(u, v)]

# create missing edges graph:
missing_edges_graph = nx.Graph()
missing_edges_graph.add_edges_from(missing_edges)

# plotting results:


# extract weights to create color maps:
ms_weights = [d["weight"] for u, v, d in P1.edges(data=True)]
max_ms_weight = max(ms_weights)
hv_weights = [d["weight"] for u, v, d in H.edges(data=True)]
max_hv_weight = max(hv_weights)
P1_weights = [d["weight"] for u, v, d in P1.edges(data=True)]
max_P1_weight = max(P1_weights)

# Normalize edge weights for colormap
norm = plt.Normalize(0, max(max_hv_weight, max_ms_weight))
# Choose a colormap (e.g., viridis) to map weights to colors
cmap = plt.get_cmap("cividis")

# Map edge weights to colors
edge_colors1 = [cmap(norm(weight)) for weight in ms_weights]
edge_colors2 = [cmap(norm(weight)) for weight in hv_weights]

# commented lines represent weight with width instead of color (not so perceivable):
# plot_options1 = {"node_size": 10, "with_labels": True, "width": 0.30, "edge_color":'brown'}
# plot_options2 = {"node_size": 10, "with_labels": True, "width": 0.30, "edge_color":'green'}
pos = nx.spring_layout(P1, iterations=15, seed=1721)
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 9))
ax1.axis("off")
ax2.axis("off")
ax3.axis("off")
# nx.draw_networkx_edges(P1, pos, ax=ax1, width=[w / max_P1_weight * 0.3 for w in P1_weights])
nx.draw_networkx_edges(G, pos, ax=ax1, edge_color=edge_colors1)
nx.draw_networkx_nodes(G, pos, ax=ax1, node_size=10)
nx.draw_networkx_labels(G, pos, ax=ax1)
# nx.draw_networkx_edges(H, pos2, ax=ax2, width=[w / max_hv_weight * 0.3 for w in hv_weights])
nx.draw_networkx_edges(H, pos, ax=ax2, edge_color=edge_colors2)
nx.draw_networkx_nodes(H, pos, ax=ax2, node_size=10)
nx.draw_networkx_labels(H, pos, ax=ax2)

nx.draw_networkx_edges(missing_edges_graph, pos, ax=ax3, width=0.3)
nx.draw_networkx_nodes(missing_edges_graph, pos, ax=ax3, node_size=10)
nx.draw_networkx_labels(missing_edges_graph, pos, ax=ax3)

# nx.draw(G, with_labels=True)
plt.show()
