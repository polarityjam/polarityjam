import numpy as np
from skimage.future.graph import RAG


def orientation_graph_nf(img):
    """Gets the RegionAdjacencyGraph for an instance segmentation """
    rag = RAG(img.astype("int"))
    rag.remove_node(0)
    return rag


def remove_islands(frame_graph, list_of_islands):
    """Remove unconnected cells (Cells without neighbours)."""

    # remove islands from image and graph
    for elem in np.unique(list_of_islands):
        frame_graph.remove_node(elem)

    return frame_graph
