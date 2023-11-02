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
ms1 = pd.read_csv('/home/vant/code/tfm1/data/subject_networks_FA_v1/090MSVIS_FA_factor.csv', header=None)


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

# create missing edges graph:
missing_edges_graph = nx.difference(H, P1)
print(missing_edges_graph)

edge_hv = H.get_edge_data(48, 58)['weight']
edge_P1 = P1.get_edge_data(48, 58)['weight']
print(edge_hv)
print(edge_P1)



# plotting results:


# extract weights to create color maps:
ms_weights = [d["weight"] for u, v, d in G.edges(data=True)]
max_ms_weight = max(ms_weights)
hv_weights = [d["weight"] for u, v, d in H.edges(data=True)]
max_hv_weight = max(hv_weights)
P1_weights = [d["weight"] for u, v, d in P1.edges(data=True)]
max_P1_weight = max(P1_weights)


# Calculate the normalization factor based on max_hv_weight
normalization_factor = max_hv_weight

# Normalize ms_weights
normalized_ms_weights = [weight / normalization_factor for weight in ms_weights]

# Normalize hv_weights
normalized_hv_weights = [weight / normalization_factor for weight in hv_weights]

# Normalize P1_weights
normalized_P1_weights = [weight / normalization_factor for weight in P1_weights]

# Print the normalized values
print(max(normalized_ms_weights))
print(max(normalized_hv_weights))
print(max(normalized_P1_weights))
print(normalization_factor)


cmap = plt.get_cmap("cividis")

edge_colors_G = cmap(ms_weights)
edge_colors_H = cmap(hv_weights)
edge_colors_P1 = cmap(P1_weights)


#sm_H =plt.cm.ScalarMappable(cmap=cmap)
sm_H = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=0, vmax=max_hv_weight))
sm_H.set_array([])  # This sets the data range for the color bar


#weight_threshold = 0.3

#filtered_edges_H = [(u, v, d) for u, v, d in H.edges(data=True) if d['weight'] > weight_threshold]
#filtered_edges_P1 = [(u, v, d) for u, v, d in P1.edges(data=True) if d['weight'] > weight_threshold]


pos = nx.spring_layout(P1, iterations=15, seed=1721)
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 9))
ax1.axis("off")
ax2.axis("off")
ax3.axis("off")

#nx.draw_networkx_edges(G, pos, ax=ax1, edge_color=edge_colors_avg)
#nx.draw_networkx_nodes(G, pos, ax=ax1, node_size=10)
#nx.draw_networkx_labels(G, pos, ax=ax1)

nx.draw_networkx_edges(P1, pos, ax=ax1, edge_color=edge_colors_P1)
nx.draw_networkx_nodes(P1, pos, ax=ax1, node_size=10)
nx.draw_networkx_labels(P1, pos, ax=ax1)

#remember to change edge_color in H according to comparison type:
# edge_colors_avg or egde_colors_one

nx.draw_networkx_edges(H, pos, ax=ax2, edge_color=edge_colors_H)
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

# nx.draw(G, with_labels=True)
plt.show()


