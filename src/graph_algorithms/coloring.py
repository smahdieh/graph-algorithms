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
    # marked_by[c] == v means color c is used by a neighbor of v. Storing v
    # instead of True means the list never has to be cleared between
    # vertices. A vertex has at most n - 1 neighbors, so colors 0 .. n are
    # enough. Time O(V + E): vertex v costs O(deg(v)). Space O(V).
    marked_by = [-1] * (n + 1)
    for v in order:
        for u in graph.neighbors(v):
            if colors[u] != -1:
                marked_by[colors[u]] = v
        color = 0
        while marked_by[color] == v:
            color += 1
        colors[v] = color
    return colors


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
    # Bucket queue: buckets[d] holds the remaining vertices whose current
    # degree is d. Removing a vertex lowers each remaining neighbor's
    # degree by one, so the minimum degree drops by at most one per
    # removal; the pointer min_deg therefore moves O(V + E) steps in total.
    # Time O(V + E), space O(V).
    buckets: list[set[int]] = [set() for _ in range(graph.max_degree() + 1)]
    for v in range(n):
        buckets[degree[v]].add(v)
    removed = [False] * n
    removal_order: list[int] = []
    min_deg = 0
    for _ in range(n):
        while not buckets[min_deg]:
            min_deg += 1
        v = buckets[min_deg].pop()
        removed[v] = True
        removal_order.append(v)
        for u in graph.neighbors(v):
            if not removed[u]:
                buckets[degree[u]].remove(u)
                degree[u] -= 1
                buckets[degree[u]].add(u)
        min_deg = max(min_deg - 1, 0)
    removal_order.reverse()
    return removal_order


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
