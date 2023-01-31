from typing import List, Set, Tuple

import numpy as np
from skimage.future.graph import RAG

from polarityjam.model.properties import NeighborhoodProps
from polarityjam.polarityjam_logging import get_logger


def shared_edges(rag: RAG, node: int) -> List[Set[int]]:
    """Return all first nearest neighbor nodes of a given node in a rag

    Args:
        rag:
            The RegionAdjacencyGraph.
        node:
            The node of interest.

    Returns:

    """
    first_nearest = list(set(rag.neighbors(node)))
    return first_nearest


def n_neighbors(rag: RAG, node: int) -> Tuple[List[Set[int]], List[int]]:
    """Get first and second nearest neighbors of a node in a rag

    Args:
        rag:
            The RegionAdjacencyGraph.
        node:
            The node of interest.

    Returns:
        Tuple of the first and second nearest neighbors of a node in a rag.

    """
    first_nearest_list = shared_edges(rag, node)

    # get list of second nearest neighbors
    k_nearest_list_unfiltered = []
    for first_nearest in first_nearest_list:
        k_nearest = list(rag.neighbors(first_nearest))
        k_nearest_list_unfiltered += k_nearest

    first_nearest_set = set(first_nearest_list)
    k_nearest_set = set(k_nearest_list_unfiltered)

    # get the difference of both sets and remove the node of interest from the final set
    second_nearest_list = list(k_nearest_set.difference(first_nearest_set))
    second_nearest_list = list(filter(lambda n: n != node, second_nearest_list))

    return first_nearest_list, second_nearest_list


def k_neighbor_dif(graph: RAG, foi: str) -> List[NeighborhoodProps]:
    """Extracts neighborhood statics from graph."

    Args:
        graph:
            The RegionAdjacencyGraph.
        foi:
            The feature of interest.

    Returns:
        List of NeighborhoodProps objects.

    """
    get_logger().info("Calculating first and second nearest neighbor statistic...")
    neighborhood_props_list = []
    for node in graph.nodes():
        neighborhood_props = NeighborhoodProps()
        neighbours = len(list(graph.neighbors(int(node))))

        neighborhood_props.num_neighbours = neighbours

        # feature of interest, first and second_nearest_neighbors
        node_foi = graph.nodes[node][foi]
        first_nearest_neighbors, second_nearest_neighbors = n_neighbors(graph, node)

        # feature of interest in comparison with first_nearest neighbor nodes
        foi_first_nearest_diff = [graph.nodes[neighbor][foi] - node_foi for neighbor in first_nearest_neighbors]

        # feature of interest in comparison to second nearest neighbor nodes
        foi_second_nearest_diff = [graph.nodes[neighbor][foi] - node_foi for neighbor in second_nearest_neighbors]

        # append first nearest neighbors statistics
        if len(foi_first_nearest_diff) > 0:
            neighborhood_props.mean_dif_first_neighbors = np.mean(foi_first_nearest_diff)
            neighborhood_props.median_dif_first_neighbors = np.median(foi_first_nearest_diff)
            neighborhood_props.var_dif_first_neighbors = np.std(foi_first_nearest_diff)
            neighborhood_props.range_dif_first_neighbors = abs(
                np.min(foi_first_nearest_diff) - np.max(foi_first_nearest_diff))

        # append second nearest neighbors statistics
        if len(foi_second_nearest_diff) > 0:
            neighborhood_props.mean_dif_second_neighbors = np.mean(foi_second_nearest_diff)
            neighborhood_props.median_dif_second_neighbors = np.median(foi_second_nearest_diff)
            neighborhood_props.var_dif_second_neighbors = np.std(foi_second_nearest_diff)
            neighborhood_props.range_dif_second_neighbors = abs(
                np.min(foi_second_nearest_diff) - np.max(foi_second_nearest_diff))

        neighborhood_props_list.append(neighborhood_props)

    return neighborhood_props_list
