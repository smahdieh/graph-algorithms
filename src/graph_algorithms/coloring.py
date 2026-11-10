"""Greedy graph coloring with and without a chosen vertex ordering.

A proper coloring gives every vertex a color (an integer 0, 1, 2, ...) so
that no edge joins two vertices of the same color. Using the fewest colors
is NP-hard. Greedy coloring visits the vertices in some order and gives
each one the lowest color not used by its already-colored neighbors; the
order decides how many colors it ends up using.
"""

from __future__ import annotations

from typing import Sequence

from graph_algorithms.graph import Graph


def greedy_coloring(graph: Graph, order: Sequence[int] | None = None) -> list[int]:
    """Greedy coloring: give each vertex the lowest legal color, in order.

    Input:  graph -- an undirected Graph;
            order -- the order to visit the vertices (a permutation of
                     0 .. n-1). None means the arbitrary order 0, 1, ..., n-1,
                     which is the basic greedy algorithm.
    Output: colors, a list where colors[v] is the color of vertex v.
            The coloring is proper and uses at most max_degree + 1 colors.
    """
    n = graph.num_vertices
    if order is None:
        order = range(n)
    colors = [-1] * n  # -1 means "not colored yet"
    # TODO (step 4): implement.
    # For each vertex v in `order`:
    #   1. Collect the colors already used by v's neighbors
    #      (graph.neighbors(v); skip neighbors whose color is -1).
    #   2. Find the smallest color 0, 1, 2, ... not in that collection.
    #   3. Set colors[v] to it.
    # Return `colors`.
    # To get O(V + E) time, avoid building a new set per vertex: keep one
    # list `marked_by` of size n + 1 and set marked_by[c] = v for each
    # neighbor color c; then color c is taken for v iff marked_by[c] == v.
    raise NotImplementedError


def smallest_last_order(graph: Graph) -> list[int]:
    """Smallest-last vertex ordering (Matula and Beck, 1983).

    Repeatedly remove a vertex of minimum degree in the remaining graph;
    the ordering is the removal order reversed, so the vertex removed last
    is colored first. Greedy coloring in this order uses at most d + 1
    colors, where d is the degeneracy of the graph.

    Input:  graph -- an undirected Graph.
    Output: a list containing every vertex exactly once.
    """
    n = graph.num_vertices
    degree = [graph.degree(v) for v in range(n)]
    # TODO (step 4): implement with a bucket queue, O(V + E) time.
    # 1. buckets[d] = set of vertices whose current degree is d
    #    (d goes from 0 to graph.max_degree()).
    # 2. Repeat n times:
    #    a. Find the smallest d with a non-empty bucket (keep a pointer
    #       `min_deg`; it only needs to move back by 1 after a removal).
    #    b. Pop a vertex v from buckets[d], mark it removed, append it to
    #       `removal_order`.
    #    c. For each neighbor u of v that is not removed yet: move u from
    #       buckets[degree[u]] to buckets[degree[u] - 1] and decrease
    #       degree[u].
    # 3. Return removal_order reversed.
    raise NotImplementedError


def smallest_last_coloring(graph: Graph) -> list[int]:
    """Greedy coloring in smallest-last order.

    Input:  graph -- an undirected Graph.
    Output: colors, where colors[v] is the color of vertex v.
    """
    return greedy_coloring(graph, smallest_last_order(graph))


def num_colors(colors: Sequence[int]) -> int:
    """Return how many distinct colors a coloring uses."""
    return len(set(colors))


def is_proper_coloring(graph: Graph, colors: Sequence[int]) -> bool:
    """Check that a coloring is complete and proper.

    Input:  graph -- an undirected Graph; colors -- colors[v] for every v.
    Output: True if every vertex has a color >= 0 and no edge joins two
            vertices of the same color.
    """
    if len(colors) != graph.num_vertices or any(c < 0 for c in colors):
        return False
    return all(colors[u] != colors[v] for u, v in graph.edges())
