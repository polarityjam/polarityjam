from typing import List

import numpy as np
from skimage.future.graph import RAG


def orientation_graph_nf(img: np.ndarray) -> RAG:
    """Gets the RegionAdjacencyGraph for an instance segmentation image.

    Args:
        img:
            The instance segmentation image.

    Returns:
        The RegionAdjacencyGraph.

    """
    rag = RAG(img.astype("int"))
    rag.remove_node(0)
    return rag


def remove_islands(frame_graph: RAG, list_of_islands: List[int]) -> RAG:
    """Remove unconnected cells (Cells without neighbours).

    Args:
        frame_graph:
            The RegionAdjacencyGraph.
        list_of_islands:
            List of unconnected cells.

    Returns:
        The RegionAdjacencyGraph without unconnected cells.

    """

    # remove islands from image and graph
    for elem in np.unique(list_of_islands):
        frame_graph.remove_node(elem)

    return frame_graph
