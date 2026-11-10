"""Unit tests for greedy coloring and the smallest-last ordering."""

import itertools
import random

import pytest

from graph_algorithms.coloring import (
    greedy_coloring,
    is_proper_coloring,
    num_colors,
    smallest_last_coloring,
    smallest_last_order,
)
from graph_algorithms.graph import Graph


def random_graph(n: int, p: float, seed: int) -> Graph:
    """Erdos-Renyi G(n, p) graph, used only by the tests."""
    rng = random.Random(seed)
    edges = [(u, v) for u, v in itertools.combinations(range(n), 2) if rng.random() < p]
    return Graph.from_edges(n, edges)


def complete(n: int) -> Graph:
    return Graph.from_edges(n, itertools.combinations(range(n), 2))


def cycle(n: int) -> Graph:
    return Graph.from_edges(n, [(i, (i + 1) % n) for i in range(n)])


def crown(k: int) -> Graph:
    """Crown graph: K_{k,k} minus a perfect matching (bipartite).

    Vertices are interleaved a0, b0, a1, b1, ... so that the natural order
    0, 1, 2, ... is the worst case for greedy and forces k colors.
    """
    edges = [(2 * i, 2 * j + 1) for i in range(k) for j in range(k) if i != j]
    return Graph.from_edges(2 * k, edges)


def tree(n: int, seed: int) -> Graph:
    """Random tree: each vertex i > 0 joins a random earlier vertex."""
    rng = random.Random(seed)
    return Graph.from_edges(n, [(i, rng.randrange(i)) for i in range(1, n)])


COLORINGS = {"greedy": greedy_coloring, "smallest_last": smallest_last_coloring}


@pytest.fixture(params=list(COLORINGS), ids=list(COLORINGS))
def color_fn(request):
    return COLORINGS[request.param]


# --- helpers ---------------------------------------------------------------

def test_is_proper_coloring_detects_conflict():
    graph = Graph.from_edges(2, [(0, 1)])
    assert is_proper_coloring(graph, [0, 1])
    assert not is_proper_coloring(graph, [0, 0])
    assert not is_proper_coloring(graph, [0, -1])  # uncolored vertex
    assert not is_proper_coloring(graph, [0])      # wrong length


def test_num_colors():
    assert num_colors([0, 1, 0, 2]) == 3
    assert num_colors([]) == 0


# --- both colorings ---------------------------------------------------------

def test_empty_graph(color_fn):
    assert color_fn(Graph(0)) == []


def test_no_edges_uses_one_color(color_fn):
    assert color_fn(Graph(5)) == [0] * 5


@pytest.mark.parametrize("n", [1, 2, 5, 8])
def test_complete_graph_needs_n_colors(color_fn, n):
    colors = color_fn(complete(n))
    assert is_proper_coloring(complete(n), colors)
    assert num_colors(colors) == n


@pytest.mark.parametrize(
    "graph",
    [cycle(7), crown(5), tree(30, seed=1), random_graph(60, 0.15, seed=2)],
    ids=["odd_cycle", "crown", "tree", "random"],
)
def test_coloring_is_proper(color_fn, graph):
    assert is_proper_coloring(graph, color_fn(graph))


@pytest.mark.parametrize("seed", range(10))
def test_at_most_max_degree_plus_one_colors(color_fn, seed):
    graph = random_graph(50, 0.2, seed=seed)
    assert num_colors(color_fn(graph)) <= graph.max_degree() + 1


def test_odd_cycle_needs_three_colors(color_fn):
    assert num_colors(color_fn(cycle(9))) == 3


# --- greedy with a given order ----------------------------------------------

def test_greedy_follows_given_order():
    # Path 0-1-2: coloring the middle first gives it color 0.
    graph = Graph.from_edges(3, [(0, 1), (1, 2)])
    assert greedy_coloring(graph, [1, 0, 2]) == [1, 0, 1]
    assert greedy_coloring(graph) == [0, 1, 0]


def test_basic_greedy_worst_case_on_crown():
    # Natural order is bad on the crown graph: k colors for a bipartite graph.
    assert num_colors(greedy_coloring(crown(6))) == 6


# --- smallest-last ordering ---------------------------------------------------

@pytest.mark.parametrize("seed", range(5))
def test_order_is_permutation(seed):
    graph = random_graph(40, 0.2, seed=seed)
    assert sorted(smallest_last_order(graph)) == list(graph.vertices())


def test_order_of_empty_graph():
    assert smallest_last_order(Graph(0)) == []


def test_smallest_last_fixes_crown():
    # Crown graphs are bipartite; smallest-last colors them with 2 colors.
    assert num_colors(smallest_last_coloring(crown(6))) == 2


@pytest.mark.parametrize("seed", range(5))
def test_smallest_last_colors_trees_with_two(seed):
    # Trees have degeneracy 1, so smallest-last uses at most 2 colors.
    assert num_colors(smallest_last_coloring(tree(40, seed=seed))) == 2


def test_order_puts_low_degree_vertex_last():
    # Star: the leaves (degree 1) are removed first, so they come last.
    star = Graph.from_edges(6, [(0, i) for i in range(1, 6)])
    order = smallest_last_order(star)
    assert set(order[-4:]) <= {1, 2, 3, 4, 5}
    assert num_colors(smallest_last_coloring(star)) == 2
