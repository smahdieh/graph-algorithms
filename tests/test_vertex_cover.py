"""Unit tests for the vertex cover algorithms."""

import itertools
import random

import pytest

from graph_algorithms.graph import Graph
from graph_algorithms.vertex_cover import (
    basic_vertex_cover,
    is_vertex_cover,
    randomized_vertex_cover,
)


def random_graph(n: int, p: float, seed: int) -> Graph:
    """Erdos-Renyi G(n, p) graph, used only by the tests."""
    rng = random.Random(seed)
    edges = [(u, v) for u, v in itertools.combinations(range(n), 2) if rng.random() < p]
    return Graph.from_edges(n, edges)


def min_cover_size(graph: Graph) -> int:
    """Exact minimum vertex cover size by brute force (small graphs only)."""
    for k in range(graph.num_vertices + 1):
        for subset in itertools.combinations(graph.vertices(), k):
            if is_vertex_cover(graph, set(subset)):
                return k
    return graph.num_vertices


def star(n_leaves: int) -> Graph:
    return Graph.from_edges(n_leaves + 1, [(0, i) for i in range(1, n_leaves + 1)])


def complete(n: int) -> Graph:
    return Graph.from_edges(n, itertools.combinations(range(n), 2))


def path(n: int) -> Graph:
    return Graph.from_edges(n, [(i, i + 1) for i in range(n - 1)])


# Run every shared test on both algorithms (randomized with a fixed seed).
ALGORITHMS = {
    "basic": basic_vertex_cover,
    "randomized": lambda g: randomized_vertex_cover(g, seed=0),
}


@pytest.fixture(params=list(ALGORITHMS), ids=list(ALGORITHMS))
def cover_fn(request):
    return ALGORITHMS[request.param]


# --- is_vertex_cover helper -------------------------------------------------

def test_is_vertex_cover_detects_uncovered_edge():
    graph = path(3)  # edges 0-1, 1-2
    assert is_vertex_cover(graph, {1})
    assert not is_vertex_cover(graph, {0})


# --- both algorithms ----------------------------------------------------------

def test_graph_without_edges_gives_empty_cover(cover_fn):
    assert cover_fn(Graph(0)) == set()
    assert cover_fn(Graph(5)) == set()


def test_single_edge(cover_fn):
    cover = cover_fn(Graph.from_edges(2, [(0, 1)]))
    assert is_vertex_cover(Graph.from_edges(2, [(0, 1)]), cover)
    assert cover <= {0, 1}


@pytest.mark.parametrize(
    "graph",
    [star(6), complete(6), path(7), random_graph(40, 0.2, seed=1)],
    ids=["star", "complete", "path", "random"],
)
def test_output_is_valid_cover(cover_fn, graph):
    cover = cover_fn(graph)
    assert is_vertex_cover(graph, cover)
    assert cover <= set(graph.vertices())


def test_input_graph_not_modified(cover_fn):
    graph = random_graph(20, 0.3, seed=2)
    edges_before = sorted(graph.edges())
    cover_fn(graph)
    assert sorted(graph.edges()) == edges_before


# --- basic algorithm ----------------------------------------------------------

@pytest.mark.parametrize("seed", range(20))
def test_basic_is_2_approximation(seed):
    graph = random_graph(10, 0.35, seed=seed)
    assert len(basic_vertex_cover(graph)) <= 2 * min_cover_size(graph)


def test_basic_cover_has_even_size():
    # It always adds both endpoints of an edge, so the size is even.
    for seed in range(10):
        assert len(basic_vertex_cover(random_graph(30, 0.2, seed))) % 2 == 0


def test_basic_on_star_takes_one_edge():
    assert len(basic_vertex_cover(star(10))) == 2


# --- randomized algorithm -----------------------------------------------------

def test_randomized_same_seed_same_result():
    graph = random_graph(50, 0.1, seed=3)
    assert randomized_vertex_cover(graph, seed=7) == randomized_vertex_cover(graph, seed=7)


def test_randomized_valid_for_many_seeds():
    graph = random_graph(30, 0.2, seed=4)
    for seed in range(50):
        assert is_vertex_cover(graph, randomized_vertex_cover(graph, seed=seed))


def test_randomized_average_within_factor_2():
    # The guarantee is on the expected size, so check the mean over many seeds.
    graph = random_graph(10, 0.4, seed=5)
    sizes = [len(randomized_vertex_cover(graph, seed=s)) for s in range(200)]
    assert sum(sizes) / len(sizes) <= 2 * min_cover_size(graph)
