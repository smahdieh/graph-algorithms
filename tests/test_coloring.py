"""Unit tests for greedy coloring and the smallest-last ordering."""

import pytest

from graph_algorithms.coloring import (
    greedy_coloring,
    is_proper_coloring,
    num_colors,
    smallest_last_coloring,
    smallest_last_order,
)
from graph_algorithms.generators import complete, crown, cycle, random_graph, random_tree, star
from graph_algorithms.graph import Graph

# Shared tests run on both colorings.
COLORINGS = {"greedy": greedy_coloring, "smallest_last": smallest_last_coloring}


@pytest.fixture(params=list(COLORINGS), ids=list(COLORINGS))
def color_fn(request):
    """Run a test once with each coloring algorithm.

    Output: a function graph -> list of colors.
    """
    return COLORINGS[request.param]


# --- helpers ---------------------------------------------------------------

def test_is_proper_coloring_detects_conflict():
    """The checker rejects equal neighbours, uncolored vertices and wrong lengths."""
    graph = Graph.from_edges(2, [(0, 1)])
    assert is_proper_coloring(graph, [0, 1])
    assert not is_proper_coloring(graph, [0, 0])
    assert not is_proper_coloring(graph, [0, -1])  # uncolored vertex
    assert not is_proper_coloring(graph, [0])      # wrong length


def test_num_colors():
    """num_colors counts distinct colors."""
    assert num_colors([0, 1, 0, 2]) == 3
    assert num_colors([]) == 0


# --- both colorings ---------------------------------------------------------

def test_empty_graph(color_fn):
    """A graph with no vertices gets an empty coloring."""
    assert color_fn(Graph(0)) == []


def test_no_edges_uses_one_color(color_fn):
    """Without edges every vertex can take color 0."""
    assert color_fn(Graph(5)) == [0] * 5


@pytest.mark.parametrize("n", [1, 2, 5, 8])
def test_complete_graph_needs_n_colors(color_fn, n):
    """In K_n every vertex sees all others, so n colors are needed."""
    colors = color_fn(complete(n))
    assert is_proper_coloring(complete(n), colors)
    assert num_colors(colors) == n


@pytest.mark.parametrize(
    "graph",
    [cycle(7), crown(5), random_tree(30, seed=1), random_graph(60, 0.15, seed=2)],
    ids=["odd_cycle", "crown", "tree", "random"],
)
def test_coloring_is_proper(color_fn, graph):
    """No edge joins two vertices of the same color."""
    assert is_proper_coloring(graph, color_fn(graph))


@pytest.mark.parametrize("seed", range(10))
def test_at_most_max_degree_plus_one_colors(color_fn, seed):
    """Greedy coloring in any order uses at most max_degree + 1 colors."""
    graph = random_graph(50, 0.2, seed=seed)
    assert num_colors(color_fn(graph)) <= graph.max_degree() + 1


def test_odd_cycle_needs_three_colors(color_fn):
    """An odd cycle is not bipartite, so it needs exactly 3 colors."""
    assert num_colors(color_fn(cycle(9))) == 3


# --- greedy with a given order ----------------------------------------------

def test_greedy_follows_given_order():
    """On path 0-1-2, coloring the middle vertex first gives it color 0."""
    graph = Graph.from_edges(3, [(0, 1), (1, 2)])
    assert greedy_coloring(graph, [1, 0, 2]) == [1, 0, 1]
    assert greedy_coloring(graph) == [0, 1, 0]


def test_basic_greedy_worst_case_on_crown():
    """The natural order is the worst case on a crown graph: k colors, not 2."""
    assert num_colors(greedy_coloring(crown(6))) == 6


# --- smallest-last ordering ---------------------------------------------------

@pytest.mark.parametrize("seed", range(5))
def test_order_is_permutation(seed):
    """The ordering contains every vertex exactly once."""
    graph = random_graph(40, 0.2, seed=seed)
    assert sorted(smallest_last_order(graph)) == list(graph.vertices())


def test_order_of_empty_graph():
    """A graph with no vertices has an empty ordering."""
    assert smallest_last_order(Graph(0)) == []


def test_smallest_last_fixes_crown():
    """Crown graphs are bipartite, and smallest-last colors them with 2 colors."""
    assert num_colors(smallest_last_coloring(crown(6))) == 2


@pytest.mark.parametrize("seed", range(5))
def test_smallest_last_colors_trees_with_two(seed):
    """Trees have degeneracy 1, so smallest-last uses at most 2 colors."""
    assert num_colors(smallest_last_coloring(random_tree(40, seed=seed))) == 2


def test_order_puts_low_degree_vertex_last():
    """On a star the leaves (degree 1) are removed first, so they come last."""
    graph = star(5)
    order = smallest_last_order(graph)
    assert set(order[-4:]) <= {1, 2, 3, 4, 5}
    assert num_colors(smallest_last_coloring(graph)) == 2
