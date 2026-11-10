"""Undirected simple graph and a loader for DIMACS edge files.

Vertices are the integers 0 .. n-1. DIMACS files number vertices from 1,
so the loader subtracts 1 from every vertex it reads.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Iterator


class Graph:
    """Undirected graph with no self-loops or parallel edges.

    Stored as an adjacency list: ``self._adj[v]`` is the set of neighbors of
    v. This uses O(V + E) space, and adding an edge, checking an edge, or
    reading a degree each take O(1) time on average.
    """

    def __init__(self, num_vertices: int) -> None:
        """Create a graph with no edges.

        Input:  num_vertices -- number of vertices n (must be >= 0).
        Output: a graph with vertices 0 .. n-1 and no edges.
        """
        if num_vertices < 0:
            raise ValueError("num_vertices must be non-negative")
        self._adj: list[set[int]] = [set() for _ in range(num_vertices)]
        self._num_edges = 0

    @classmethod
    def from_edges(cls, num_vertices: int, edges: Iterable[tuple[int, int]]) -> Graph:
        """Build a graph from a list of edges.

        Input:  num_vertices -- number of vertices n;
                edges        -- pairs (u, v) with 0 <= u, v < n.
        Output: the graph containing those edges (duplicates are ignored).
        """
        graph = cls(num_vertices)
        for u, v in edges:
            graph.add_edge(u, v)
        return graph

    @property
    def num_vertices(self) -> int:
        """Return the number of vertices n."""
        return len(self._adj)

    @property
    def num_edges(self) -> int:
        """Return the number of edges m."""
        return self._num_edges

    def add_edge(self, u: int, v: int) -> None:
        """Add the undirected edge {u, v}.

        Input:  u, v -- distinct vertices in 0 .. n-1.
        Output: none. Adding an edge that already exists does nothing.
        Raises ValueError for a self-loop or an out-of-range vertex.
        """
        self._check_vertex(u)
        self._check_vertex(v)
        if u == v:
            raise ValueError(f"self-loop on vertex {u} is not allowed")
        if v not in self._adj[u]:
            self._adj[u].add(v)
            self._adj[v].add(u)
            self._num_edges += 1

    def has_edge(self, u: int, v: int) -> bool:
        """Return True if the edge {u, v} is in the graph."""
        return 0 <= u < len(self._adj) and v in self._adj[u]

    def neighbors(self, v: int) -> set[int]:
        """Return the set of neighbors of v (do not modify it)."""
        return self._adj[v]

    def degree(self, v: int) -> int:
        """Return the number of neighbors of v."""
        return len(self._adj[v])

    def vertices(self) -> range:
        """Return the vertices 0 .. n-1."""
        return range(len(self._adj))

    def edges(self) -> Iterator[tuple[int, int]]:
        """Yield every edge once, as (u, v) with u < v."""
        for u, nbrs in enumerate(self._adj):
            for v in nbrs:
                if u < v:
                    yield (u, v)

    def max_degree(self) -> int:
        """Return the largest degree, or 0 for a graph with no vertices."""
        return max((len(nbrs) for nbrs in self._adj), default=0)

    def _check_vertex(self, v: int) -> None:
        """Raise ValueError if v is not in 0 .. n-1."""
        if not 0 <= v < len(self._adj):
            raise ValueError(f"vertex {v} is out of range 0..{len(self._adj) - 1}")

    def __repr__(self) -> str:
        return f"Graph(n={self.num_vertices}, m={self.num_edges})"


def parse_dimacs(lines: Iterable[str]) -> Graph:
    """Build a graph from the lines of a DIMACS edge file.

    Format: comment lines start with 'c', one header line 'p edge N M'
    (some files write 'p col N M'), then one line 'e U V' per edge, with
    vertices numbered 1 .. N.

    Input:  lines -- the file's lines.
    Output: the graph, with vertices renumbered to 0 .. N-1.
    Raises ValueError if the header is missing or a line is malformed.
    """
    graph: Graph | None = None
    for line_no, raw in enumerate(lines, start=1):
        parts = raw.split()
        if not parts or parts[0] == "c":
            continue
        if parts[0] == "p":
            if len(parts) < 4:
                raise ValueError(f"line {line_no}: bad header {raw.strip()!r}")
            graph = Graph(int(parts[2]))
        elif parts[0] == "e":
            if graph is None:
                raise ValueError(f"line {line_no}: edge before the 'p' header")
            if len(parts) < 3:
                raise ValueError(f"line {line_no}: bad edge {raw.strip()!r}")
            graph.add_edge(int(parts[1]) - 1, int(parts[2]) - 1)
        else:
            raise ValueError(f"line {line_no}: unknown line type {parts[0]!r}")
    if graph is None:
        raise ValueError("no 'p' header found")
    return graph


def load_dimacs(path: str | Path) -> Graph:
    """Read a DIMACS edge file (e.g. a BHOSLIB/FRB instance).

    Input:  path -- path to the file.
    Output: the graph it describes (vertices 0 .. N-1).
    """
    with open(path, encoding="utf-8") as f:
        return parse_dimacs(f)
