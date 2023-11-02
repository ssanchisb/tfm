import os.path
import networkx as nx
import glob
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from sklearn.preprocessing import MinMaxScaler


FA_matrices = [pd.read_csv(file, header=None) for file in
               glob.glob(os.path.join('/home/vant/code/tfm1/data/subject_networks_FA_v1', "*.csv"))]

# print(len(FA_matrices))
# print(avg_all_m)

# extract labels of MS vs. HV:
demographics_df = pd.read_csv('/home/vant/code/tfm1/data/clinic.csv')
labels = demographics_df['controls_ms'].tolist()

#print(len(labels))
#print(labels.count(0))

# use labels to separate into 2 groups:
MS_FA = [FA_matrices[i] for i, value in enumerate(labels) if value == 1]
HV_FA = [FA_matrices[i] for i, value in enumerate(labels) if value == 0]
print(len(MS_FA))
print(len(HV_FA))

# Create average matrices for each of the 2 groups:
avg_fa_ms = pd.concat(MS_FA).groupby(level=0).mean()
avg_fa_hv = pd.concat(HV_FA).groupby(level=0).mean()

# extract a matrix of a single ms patient:
ms1 = pd.read_csv('/home/vant/code/tfm1/data/subject_networks_FA_v1/003MSVIS_FA_factor.csv', header=None)


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

print(P1)
print(H)

# define missing edges (difference between healthy and (avg)patient):
#missing_edges = [(u, v) for u, v in H.edges() if not P1.has_edge(u, v)]
missing_edges = [(u, v) for u, v in G.edges() if not H.has_edge(u, v)]


# create missing edges graph:
missing_edges_graph = nx.difference(H, P1)
extra_edges_graph = nx.difference(P1, H)
#missing_edges_graph = nx.Graph()
#missing_edges_graph.add_edges_from(missing_edges)
print(missing_edges_graph)

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

sm_H = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=0, vmax=max_hv_weight))
sm_H.set_array([])


# commented lines represent weight with width instead of color (not so perceivable):
# plot_options1 = {"node_size": 10, "with_labels": True, "width": 0.30, "edge_color":'brown'}
# plot_options2 = {"node_size": 10, "with_labels": True, "width": 0.30, "edge_color":'green'}
pos = nx.spring_layout(P1, iterations=15, seed=1721)
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 9))
ax1.axis("off")
ax2.axis("off")
ax3.axis("off")
# nx.draw_networkx_edges(P1, pos, ax=ax1, width=[w / max_P1_weight * 0.3 for w in P1_weights])
nx.draw_networkx_edges(P1, pos, ax=ax1, edge_color=edge_colors2)
nx.draw_networkx_nodes(P1, pos, ax=ax1, node_size=10)
nx.draw_networkx_labels(P1, pos, ax=ax1)
# nx.draw_networkx_edges(H, pos2, ax=ax2, width=[w / max_hv_weight * 0.3 for w in hv_weights])
nx.draw_networkx_edges(H, pos, ax=ax2, edge_color=edge_colors2)
nx.draw_networkx_nodes(H, pos, ax=ax2, node_size=10)
nx.draw_networkx_labels(H, pos, ax=ax2)

nx.draw_networkx_edges(missing_edges_graph, pos, ax=ax3, width=0.3)
nx.draw_networkx_nodes(missing_edges_graph, pos, ax=ax3, node_size=10)
nx.draw_networkx_labels(missing_edges_graph, pos, ax=ax3)


# Add the colorbar
cbar_ax1 = fig.add_axes([0.05, 0.1, 0.02, 0.8])
cbar1 = plt.colorbar(sm_H, cax=cbar_ax1)
cbar1.set_label("Edge Weight")


#ax1.set_title("MS Avg")
ax1.set_title("MS Single Patient")
ax2.set_title("HV Avg")
ax3.set_title("Missing connections")

average_weight_P1 = sum(P1_weights) / len(P1_weights)
average_weight_H = sum(hv_weights) / len(hv_weights)
average_weight_ms = sum(ms_weights) / len(ms_weights)

ax1.text(0.5, 0, f"Average weight: {average_weight_P1:.2f}\n  {P1.number_of_edges()} edges", size=12, ha='center', transform=ax1.transAxes)
ax2.text(0.5, 0, f"Average weight: {average_weight_H:.2f}\n  {H.number_of_edges()} edges", size=12, ha='center', transform=ax2.transAxes)
ax3.text(0.5, 0, f"{missing_edges_graph.number_of_edges()} edges in avg. HV not in patient\n ({extra_edges_graph.number_of_edges()} "
                 f"edges in patient not in avg. HV)", size=12, ha='center', transform=ax3.transAxes)

print(average_weight_P1)
print(average_weight_H)
# nx.draw(G, with_labels=True)
plt.show()