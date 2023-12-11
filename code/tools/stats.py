import numpy as np
import pandas as pd
import glob
import networkx as nx
from scipy import stats
from statistics import mean, median

# This script contains functions to calculate graph measures and perform statistical tests.


def calculate_graph_measures(data, measure='degree', weight='weight', network='global', layer='single'):
    """
    Calculates graph measures from a list of adjacency matrices (data).
    The matrices can be either single-layer or multilayer.
    :param data: list of adjacency matrices
    :param measure: graph measure to calculate
    :param weight: weight of the edges
    :param network: global or local network measure
    :param layer: single-layer or multilayer
    :return: list of graph measures
    """
    measures_list = []

    for m in data:
        if layer == 'single':
            G = nx.from_pandas_adjacency(m)
        elif layer == 'multilayer':
            G = nx.from_numpy_array(m)
        if measure == 'degree':
            measure_result = G.degree(weight=weight)
            measures = {node: val for (node, val) in measure_result}
        elif measure == 'node_degree':
            measures = sum(dict(G.degree()).values())
        elif measure == 'degree_centrality':
            if network == 'global':
                measures = list(nx.degree_centrality(G).values())
            else:
                measures = nx.degree_centrality(G)
        elif measure == 'closeness_centrality':
            if network == 'global':
                measures = list(nx.closeness_centrality(G).values())
            else:
                measures = nx.closeness_centrality(G)
        elif measure == 'average_clustering':
            measures = nx.average_clustering(G)
        elif measure == 'clustering':
            measures = nx.clustering(G)
        elif measure == 'local_efficiency':
            measures = nx.local_efficiency(G)
        elif measure == 'global_efficiency':
            measures = nx.global_efficiency(G)
        elif measure == 'betweenness_centrality':
            if network == 'global':
                measures = list(nx.betweenness_centrality(G).values())
            else:
                measures = nx.betweenness_centrality(G)
        elif measure == 'eigenvector_centrality':
            if network == 'global':
                measures = list(nx.eigenvector_centrality(G).values())
            else:
                measures = nx.eigenvector_centrality(G)
        elif measure == 'assortativity':
            measures = nx.degree_assortativity_coefficient(G)
        elif measure == 'rich_club':
            measures = nx.rich_club_coefficient(G, normalized=False, seed=42)
        else:
            raise ValueError(f"Unsupported graph measure: {measure}")

        measures_list.append(measures)
    return measures_list


def stats_test(ms='ms_st', hv='hv_st', measure='degree', weight='weight', centrality='mean', network='global',
               layer='single'):
    """
    Performs a Shapiro-Wilk test to check if the data is normally distributed.
    If the data is normally distributed, a t-test is performed.
    If the data is not normally distributed, a Mann-Whitney test is performed.

    :param ms: list of adjacency matrices for MS
    :param hv: list of adjacency matrices for HV
    :param measure: graph measure to calculate
    :param weight: weight of the edges
    :param centrality: mean or median
    :param network: global or local network measure
    :param layer: single-layer or multilayer
    :return: None
    """
    ms_list = calculate_graph_measures(ms, measure=measure, weight=weight, layer=layer)
    hv_list = calculate_graph_measures(hv, measure=measure, weight=weight, layer=layer)

    if measure == 'degree' or network != 'global':
        if centrality == 'mean':
            global_ms = [np.mean(list(s.values())) for s in ms_list]
            global_hv = [np.mean(list(s.values())) for s in hv_list]
        elif centrality == 'median':
            global_ms = [np.median(list(s.values())) for s in ms_list]
            global_hv = [np.median(list(s.values())) for s in hv_list]
        else:
            raise ValueError(f"Unsupported centrality measure: {centrality}")
    elif (measure == 'node_degree' or
          measure == 'average_clustering' or
          measure == 'local_efficiency' or
          measure == 'global_efficiency' or
          measure == 'assortativity' or
          measure == 'clustering_coefficient'):
        global_ms = [s for s in ms_list]
        global_hv = [s for s in hv_list]
    else:
        if centrality == 'mean':
            global_ms = [mean(s) for s in ms_list]
            global_hv = [mean(s) for s in hv_list]
        elif centrality == 'median':
            global_ms = [median(s) for s in ms_list]
            global_hv = [median(s) for s in hv_list]
        else:
            raise ValueError(f"Unsupported centrality measure: {centrality}")

    _, p_value_shapiro_ms = stats.shapiro(global_ms)
    _, p_value_shapiro_hv = stats.shapiro(global_hv)

    if p_value_shapiro_ms > 0.05 and p_value_shapiro_hv > 0.05:
        test = 'ttest'
        stat, p_value = stats.ttest_ind(global_ms, global_hv)
    else:
        test = 'mann-whitney'
        stat, p_value = stats.mannwhitneyu(global_ms, global_hv)

    if test == 'ttest':
        if p_value < 0.001:
            print(f"t-statistic: {stat}, p < 0.001")
        else:
            print(f"t-statistic: {stat}, p = {p_value:.3f}")

    elif test == 'mann-whitney':
        if p_value < 0.001:
            print(f"mann-whitney-statistic: {stat}, p < 0.001")
        else:
            print(f"mann-whitney-statistic: {stat}, p = {p_value:.3f}")


def return_global(data, measure='degree', weight='weight', centrality='mean', network='global',
                  layer='single'):
    """
    Returns a list of global graph measures.
    :param data: list of adjacency matrices
    :param measure: graph measure to calculate
    :param weight: weight of the edges
    :param centrality: mean or median
    :param network: global or local network measure
    :param layer: single-layer or multilayer
    :return: list of global graph measures
    """
    data_list = calculate_graph_measures(data, measure=measure, weight=weight, layer=layer)

    if measure == 'degree' or network != 'global':
        if centrality == 'mean':
            global_list = [np.mean(list(s.values())) for s in data_list]
        elif centrality == 'median':
            global_list = [np.median(list(s.values())) for s in data_list]
        else:
            raise ValueError(f"Unsupported centrality measure: {centrality}")
    elif (measure == 'node_degree' or
          measure == 'average_clustering' or
          measure == 'local_efficiency' or
          measure == 'global_efficiency' or
          measure == 'assortativity' or
          measure == 'clustering_coefficient'):
        global_list = [s for s in data_list]
    else:
        if centrality == 'mean':
            global_list = [mean(s) for s in data_list]
        elif centrality == 'median':
            global_list = [median(s) for s in data_list]
        else:
            raise ValueError(f"Unsupported centrality measure: {centrality}")

    return global_list


def stats_nodal_test(ms='ms_st', hv='hv_st', measure='degree', weight='weight', verbose=True, network='nodal',
                     sn=False):
    """
    Performs a Shapiro-Wilk test to check if the data is normally distributed.
    If the data is normally distributed, a t-test is performed.
    If the data is not normally distributed, a Mann-Whitney test is performed.

    :param ms: list of adjacency matrices for MS
    :param hv: list of adjacency matrices for HV
    :param measure: graph measure to calculate
    :param weight: weight of the edges
    :param verbose: shows detailed results
    :param network: global or local network measure
    :param layer: single-layer or multilayer
    :param sn: returns significant_nodes
    :return: significant_nodes of ms and hv
    """
    ms_list = calculate_graph_measures(ms, measure=measure, weight=weight, network=network)
    hv_list = calculate_graph_measures(hv, measure=measure, weight=weight, network=network)

    nodes_ms = [[d[key] for d in ms_list] for key in ms_list[0]]
    nodes_hv = [[d[key] for d in hv_list] for key in hv_list[0]]

    significant_nodes = []
    significant_nodes_measures_ms = []
    significant_nodes_measures_hv = []

    for i in range(len(nodes_ms)):
        _, p_value_shapiro_ms = stats.shapiro(nodes_ms)
        _, p_value_shapiro_hv = stats.shapiro(nodes_hv)

        if p_value_shapiro_ms > 0.05 and p_value_shapiro_hv > 0.05:
            test = 'ttest'
        else:
            test = 'mann-whitney'

        if test == 'ttest':
            stat, p_value = stats.ttest_ind(nodes_ms[i], nodes_hv[i])
        elif test == 'mann-whitney':
            stat, p_value = stats.mannwhitneyu(nodes_ms[i], nodes_hv[i])

        if verbose:
            if p_value < 0.001:
                print(f"statistic: {stat}, p < 0.001")
                print("  Significant")
            elif p_value < 0.05:
                print(f"statistic: {stat}, p = {p_value:.3f}")
                print("  Significant")
            else:
                print(f"statistic: {stat}, p = {p_value:.3f}")

        # Check if the p-value is less than 0.01
        if p_value < 0.001:
            significant_nodes.append(i + 1)
            significant_nodes_measures_ms.append(nodes_ms[i])
            significant_nodes_measures_hv.append(nodes_hv[i])

    print(f"Amount of significant nodes: {len(significant_nodes)}/76")
    print(f"List of significant nodes: {significant_nodes}")
    print(f"Test used: {test}")

    significant_nodes_measures_ms = np.array(significant_nodes_measures_ms)
    significant_nodes_measures_hv = np.array(significant_nodes_measures_hv)

    if sn:
        return significant_nodes_measures_ms, significant_nodes_measures_hv, significant_nodes
    else:
        return significant_nodes_measures_ms, significant_nodes_measures_hv
