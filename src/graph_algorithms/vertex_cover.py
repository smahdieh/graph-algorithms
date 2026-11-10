"""Approximation algorithms for minimum vertex cover.

A vertex cover of G = (V, E) is a set C of vertices such that every edge
has at least one endpoint in C. Finding the smallest one is NP-hard, so
both functions below return a cover that is small but not always minimum.
"""

from __future__ import annotations

import random

from graph_algorithms.graph import Graph


def basic_vertex_cover(graph: Graph) -> set[int]:
    """Edge-picking 2-approximation for vertex cover.

    While an uncovered edge (u, v) remains, add BOTH u and v to the cover;
    every edge touching u or v is then covered.

    Input:  graph -- an undirected Graph.
    Output: a set of vertices that covers every edge; its size is at most
            twice the size of a minimum vertex cover.
    """
    # TODO (step 3): implement.
    # 1. Start with an empty set `cover`.
    # 2. Loop over every edge (u, v) in graph.edges().
    # 3. If neither u nor v is in `cover`, the edge is still uncovered:
    #    add both u and v.
    #    (Skipping covered edges is the same as "deleting all edges
    #    incident to u and v" in the pseudocode.)
    # 4. Return `cover`.
    raise NotImplementedError


def randomized_vertex_cover(graph: Graph, seed: int | None = None) -> set[int]:
    """Randomized edge-picking algorithm for vertex cover.

    While an uncovered edge (u, v) remains, add ONE of u or v, chosen
    uniformly at random; every edge touching that vertex is then covered.

    Input:  graph -- an undirected Graph;
            seed  -- optional random seed, so runs can be repeated.
    Output: a set of vertices that covers every edge; its expected size
            is at most twice the size of a minimum vertex cover.
    """
    rng = random.Random(seed)
    # TODO (step 3): implement.
    # Same loop as basic_vertex_cover, but for an uncovered edge add only
    # one endpoint: rng.choice((u, v)).
    raise NotImplementedError


def is_vertex_cover(graph: Graph, cover: set[int]) -> bool:
    """Check that a set of vertices covers every edge.

    Input:  graph -- an undirected Graph; cover -- a set of vertices.
    Output: True if every edge has at least one endpoint in cover.
    """
    return all(u in cover or v in cover for u, v in graph.edges())
