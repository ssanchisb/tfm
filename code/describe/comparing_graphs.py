import os.path
import networkx as nx
import glob
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from sklearn.preprocessing import MinMaxScaler


def create_graphs(type='structural', graph1='avgMS'):
    if type == 'structural':
        FA_matrices = [pd.read_csv(file, header=None) for file in
                       glob.glob(os.path.join('/home/vant/code/tfm1/data/subject_networks_FA_v1', "*.csv"))]
    else:
        FA_matrices = [pd.read_csv(file, header=None) for file in
                       glob.glob(os.path.join('/home/vant/code/tfm1/data/subject_networks_rfMRI_v1', "*.csv"))]

    # extract labels of MS vs. HV:
    demographics_df = pd.read_csv('/home/vant/code/tfm1/data/clinic.csv')
    labels = demographics_df['controls_ms'].tolist()
    edss = demographics_df['edss'].tolist()

    #print(len(labels))
    #print(labels.count(0))

    # use labels to separate into 2 groups:
    MS_FA = [FA_matrices[i] for i, value in enumerate(labels) if value == 1]
    HV_FA = [FA_matrices[i] for i, value in enumerate(labels) if value == 0]
    MS_FA_EDSS5 = [FA_matrices[i] for i, value in enumerate(edss) if value >= 5.0]
    print(len(MS_FA))
    print(len(HV_FA))
    print(len(MS_FA_EDSS5))

    # Create average matrices for each of the 2 groups:
    avg_fa_ms = pd.concat(MS_FA).groupby(level=0).mean()
    avg_fa_hv = pd.concat(HV_FA).groupby(level=0).mean()
    avg_fa_ms_edss = pd.concat(MS_FA_EDSS5).groupby(level=0).mean()


    if graph1 == 'avgMS':
        patients = avg_fa_ms
    else:
        patients = pd.read_csv('/home/vant/code/tfm1/data/subject_networks_FA_v1/{}_FA_factor.csv'.format(graph1), header=None)


    # create graphs:
    G = nx.Graph()
    H = nx.Graph()

    num_nodes = avg_fa_hv.shape[0]

    # populate graphs with weights:
    for i in range(num_nodes):
        for j in range(i + 1, num_nodes):
            weight = patients.iloc[i, j]
            if weight != 0:
                G.add_edge(i, j, weight=weight)

    for i in range(num_nodes):
        for j in range(i + 1, num_nodes):
            weight = avg_fa_hv.iloc[i, j]
            if weight != 0:
                H.add_edge(i, j, weight=weight)

    print(G)
    print(H)


    # create missing edges graph:
    missing_edges_graph = nx.difference(H, G)
    extra_edges_graph = nx.difference(G, H)
    print(missing_edges_graph)

    # plotting results:

    # extract weights to create color maps:
    G_weights = [d["weight"] for u, v, d in G.edges(data=True)]
    max_ms_weight = max(G_weights)
    H_weights = [d["weight"] for u, v, d in H.edges(data=True)]
    max_hv_weight = max(H_weights)

    # Normalize edge weights for colormap
    norm = plt.Normalize(0, max(max_hv_weight, max_ms_weight))
    # Choose a colormap
    cmap = plt.get_cmap("cividis")

    # Map edge weights to colors
    edge_colors2 = [cmap(norm(weight)) for weight in H_weights]

    sm_H = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=0, vmax=max_hv_weight))
    sm_H.set_array([])

    pos = nx.spring_layout(H, iterations=15, seed=1721)
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 9))
    ax1.axis("off")
    ax2.axis("off")
    ax3.axis("off")

    nx.draw_networkx_edges(G, pos, ax=ax1, edge_color=edge_colors2)
    nx.draw_networkx_nodes(G, pos, ax=ax1, node_size=10)
    nx.draw_networkx_labels(G, pos, ax=ax1)


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

    ax1.set_title(graph1)
    ax2.set_title("HV Avg")
    ax3.set_title("Missing connections")

    average_weight_G = sum(G_weights) / len(G_weights)
    average_weight_H = sum(H_weights) / len(H_weights)


    ax1.text(0.5, 0, f"Average weight: {average_weight_G:.2f}\n  {G.number_of_edges()} edges", size=12, ha='center', transform=ax1.transAxes)
    ax2.text(0.5, 0, f"Average weight: {average_weight_H:.2f}\n  {H.number_of_edges()} edges", size=12, ha='center', transform=ax2.transAxes)
    ax3.text(0.5, 0, f"{missing_edges_graph.number_of_edges()} edges in avg. HV not in patient(s)\n ({extra_edges_graph.number_of_edges()} "
                     f"edges in patient(s) not in avg. HV)", size=12, ha='center', transform=ax3.transAxes)

    print(average_weight_G)
    print(average_weight_H)
    # nx.draw(G, with_labels=True)
    plt.show()
