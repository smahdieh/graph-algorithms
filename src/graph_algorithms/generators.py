"""Graph generators used by the tests, the demo and the benchmark.

Small graphs with known answers (star, path, cycle, complete, crown, tree)
and two random graph models. Every random generator takes a seed, so the
same call always returns the same graph.
"""

from __future__ import annotations

import itertools
import random

from graph_algorithms.graph import Graph


def star(num_leaves: int) -> Graph:
    """Star: center 0 joined to leaves 1 .. num_leaves.

    Input:  num_leaves -- number of leaves (>= 0).
    Output: graph with num_leaves + 1 vertices; minimum cover {0}, 2 colors.
    """
    return Graph.from_edges(num_leaves + 1, [(0, i) for i in range(1, num_leaves + 1)])


def path(n: int) -> Graph:
    """Path 0 - 1 - ... - (n-1).

    Input:  n -- number of vertices.
    Output: graph with n - 1 edges; minimum cover has n // 2 vertices.
    """
    return Graph.from_edges(n, [(i, i + 1) for i in range(n - 1)])


def cycle(n: int) -> Graph:
    """Cycle 0 - 1 - ... - (n-1) - 0.

    Input:  n -- number of vertices (>= 3).
    Output: graph with n edges; needs 3 colors if n is odd, 2 if even.
    """
    return Graph.from_edges(n, [(i, (i + 1) % n) for i in range(n)])


def complete(n: int) -> Graph:
    """Complete graph K_n: every pair of vertices is joined.

    Input:  n -- number of vertices.
    Output: graph that needs n colors; minimum cover has n - 1 vertices.
    """
    return Graph.from_edges(n, itertools.combinations(range(n), 2))


def crown(k: int) -> Graph:
    """Crown graph: K_{k,k} minus a perfect matching.

    Vertices are interleaved a0, b0, a1, b1, ... (even = side a, odd = side
    b), so the natural order 0, 1, 2, ... is the worst case for greedy
    coloring and forces k colors, although the graph is bipartite.

    Input:  k -- vertices per side.
    Output: graph with 2k vertices; 2 colors, minimum cover k (for k >= 2).
    """
    edges = [(2 * i, 2 * j + 1) for i in range(k) for j in range(k) if i != j]
    return Graph.from_edges(2 * k, edges)


def random_tree(n: int, seed: int) -> Graph:
    """Random tree: each vertex i > 0 joins a random earlier vertex.

    Input:  n -- number of vertices; seed -- random seed.
    Output: a tree (connected, n - 1 edges); needs 2 colors if n >= 2.
    """
    rng = random.Random(seed)
    return Graph.from_edges(n, [(i, rng.randrange(i)) for i in range(1, n)])


def random_graph(n: int, p: float, seed: int) -> Graph:
    """Erdos-Renyi G(n, p): each pair is joined independently with probability p.

    Input:  n -- number of vertices; p -- edge probability; seed -- random seed.
    Output: the random graph. Takes O(n^2) time, so use it for small n.
    """
    rng = random.Random(seed)
    edges = [(u, v) for u, v in itertools.combinations(range(n), 2) if rng.random() < p]
    return Graph.from_edges(n, edges)


def random_sparse_graph(n: int, avg_degree: float, seed: int) -> Graph:
    """Random G(n, m) graph with m = n * avg_degree / 2 distinct edges.

    Draws random pairs and skips self-loops and repeats until m edges exist,
    which is fast for large sparse graphs.

    Input:  n -- number of vertices; avg_degree -- target average degree;
            seed -- random seed.
    Output: the random graph.
    """
    rng = random.Random(seed)
    graph = Graph(n)
    target = int(n * avg_degree / 2)
    while graph.num_edges < target:
        u, v = rng.randrange(n), rng.randrange(n)
        if u != v:
            graph.add_edge(u, v)
    return graph
