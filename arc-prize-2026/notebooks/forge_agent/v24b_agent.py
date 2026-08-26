# =====================================================================
# v24 — Pure Graph-Explorer port (arXiv 2512.24156, MIT licensed)
# Source: github.com/dolphin-in-a-coma/arc-agi-3-just-explore (3rd private LB)
# Adapted to Kaggle MyAgent shell + arcengine imports.
#
# Approach: NO BFS, NO CNN. Frame-by-frame state-graph exploration.
# - Connected-component segmentation (4-connected by color)
# - Status-bar masking (rule: edge + ratio + twins)
# - 5 priority tiers for click candidates
# - Level Graph Explorer: state-hash nodes, frontier-aware action choice
# =====================================================================
import logging
import time
import hashlib
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Any, Dict, Hashable, List, Optional, Set, Tuple
import random

import numpy as np

from agents.agent import Agent
from arcengine import FrameData, GameAction, GameState

logger = logging.getLogger(__name__)


INFINITY = np.iinfo(np.int32).max


# NOTE: all data formats here chosen crudely, to be optimized later
edge_dtype = np.dtype([
    ("group", "i4"), # 0-indexed group id
    ("result", "i4"), # 1 if success, -1 if failed, 0 if not tested yet
    ("target", "U32"), # target node hash-name, "" if not tested or failed
    ("distance", "i4"), # distance to the frontier node, 0 means next node is the frontier
    ("errors", "i4"), # number of errors so far
])

def format_struct_table(arr):
    names = ("idx",) + arr.dtype.names
    cols = []
    for name in names:
        if name == "idx":
            cols.append([str(i) for i in range(len(arr))])
        else:
            cols.append([str(r[name]) for r in arr])
    widths = [max(len(n), *(len(v) for v in col)) for n, col in zip(names, cols)]
    header = " | ".join(n.ljust(w) for n, w in zip(names, widths))
    sep = "-+-".join("-"*w for w in widths)
    lines = []
    for i in range(len(arr)):
        line = " | ".join(cols[j][i].ljust(widths[j]) for j in range(len(names)))
        lines.append(line)
    return "\n".join([header, sep, *lines])

@dataclass
class NodeInfo:
    name: Hashable

    total_candidates: int # how many exist
    num_groups: int = 1 # FIXME: is never used
    active_group: int = 0

    group2remaining_candidate_ids: List[Set[int]] = field(default_factory=list)

    edge_data: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=edge_dtype))

    error_threshold: int = 3
    closed: bool = False # flips when last probe done
    distance: float | None = 0 # TODO: how is it initialized?

    def __post_init__(self):

        assert self.name is not None, "Node name must be provided"

        if self.num_groups > 1 and self.group2remaining_candidate_ids is None:
            raise ValueError("group2remaining_candidate_ids must be provided if num_groups > 1")

        if self.num_groups == 1 and self.group2remaining_candidate_ids is None:
            self.group2remaining_candidate_ids = [set(range(self.total_candidates))]

        self.group2remaining_candidate_ids = [set(r_c_ids) for r_c_ids in self.group2remaining_candidate_ids] # ensure it's a list of sets

        self.edge_data = np.zeros(self.total_candidates, dtype=edge_dtype)
            
        for group_id, remaining_candidate_ids in enumerate(self.group2remaining_candidate_ids):
            self.edge_data["group"][list(remaining_candidate_ids)] = group_id

    @property
    def has_open(self) -> bool:
        """Still hiding ≥1 untested edge?"""
        return len(self.tested) < self.total_candidates

    def record_test(self, edge_idx: int, success: int, target_node: Hashable | None = None) -> bool:

        edge_group_id = self.edge_data[edge_idx]["group"]

        assert self.edge_data["result"][edge_idx] == 0 and \
            self.edge_data["target"][edge_idx] == "" and \
            self.edge_data["distance"][edge_idx] == 0, \
            "Edge result must be untested before recording a test"

        if success == -1:
            self.edge_data["errors"][edge_idx] += 1
            if self.edge_data["errors"][edge_idx] >= self.error_threshold:
                self.edge_data["errors"][edge_idx] = 0
                new_group_id = edge_group_id + 1
                if new_group_id > self.num_groups - 1:
                    # count it as failed and move on
                    self.group2remaining_candidate_ids[edge_group_id].discard(edge_idx)
                    self.edge_data["result"][edge_idx] = -1
                    self.edge_data["distance"][edge_idx] = INFINITY
                    return True
                else:
                    self.edge_data["group"][edge_idx] = new_group_id
                    self.group2remaining_candidate_ids[new_group_id].add(edge_idx)
                    self.group2remaining_candidate_ids[edge_group_id].discard(edge_idx)
            return False

        self.group2remaining_candidate_ids[edge_group_id].discard(edge_idx)

        if success == 1:
            self.edge_data["target"][edge_idx] = str(target_node)
            self.edge_data["distance"][edge_idx] = -1 # NOTE: distance is maintained by the GraphExplorer class
            self.edge_data["result"][edge_idx] = 1
        elif success == 0:
            self.edge_data["distance"][edge_idx] = INFINITY
            self.edge_data["result"][edge_idx] = -1

        return True

    def has_open_group(self, group_id: int) -> bool:
        """Return True if this node has at least one untested edge belonging to *group_id* or below."""
        for i in range(group_id+1):
            if len(self.group2remaining_candidate_ids[i]) > 0:
                return True
        return False
    
    def __repr__(self) -> str:
        edge_data_repr = format_struct_table(self.edge_data)

        return f"""NodeInfo:
name={self.name},
total_candidates={self.total_candidates},
num_groups={self.num_groups},
distance={self.distance},
closed={self.closed},
{edge_data_repr}
"""


class GraphExplorer:

    def __init__(
        self,
        start_node: Hashable | None = None, 
        num_candidates: int | None = None, 
        group2remaining_candidate_ids: List[Set[int]] | None = None,
        n_groups: int = 1,
        verbose_level: int = 0,
        ) -> None:

        self._verbose_level = verbose_level
        self._n_groups = max(1, n_groups)

        self.reset()

    def reset(self) -> None:
        self._nodes: Dict[Hashable, NodeInfo] = {}
        self._G: Dict[Hashable, Set[Tuple[int, Hashable]]] = defaultdict(set) # (edge_idx, target_node)
        self._G_rev: Dict[Hashable, Set[Tuple[int, Hashable]]] = defaultdict(set) # (edge_idx, source_node)
        self._frontier: Set[Hashable] = set()
        self._dist: Dict[Hashable, int] = {}
        self._next: Dict[Hashable, Tuple[int, Hashable]] = {} # (edge_idx, target_node)
        self._active_group: int = 0  # current priority group

        self.suspicious_transitions: Dict[Tuple[Hashable, int, Hashable], int] = {} # (source_node, edge_idx, target_node) -> count
        self.suspicious_transitions_threshold: int = 3

        self._empty = True
    
    def initialize(self, start_node: Hashable | None = None, num_candidates: int | None = None, group2remaining_candidate_ids: List[Set[int]] | None = None) -> None:


        if start_node is not None:
            self._add_new_node(start_node, num_candidates, group2remaining_candidate_ids=group2remaining_candidate_ids)

        if self._verbose_level >= 1:
            print(f"\nGraph is initialized with node: {self._nodes[start_node]}")
            self.dump()

    def record_test(
        self,
        node: Hashable,
        edge_idx: Hashable,
        success: bool,
        target_node: Optional[Hashable] = None,
        target_num_candidates: Optional[int] = None,
        group2remaining_candidate_ids: Optional[List[Set[int]]] = None,
        suspicious_transition: bool = False,
    ) -> None:

        if node not in self._nodes:
            raise KeyError(f"unknown node {node!r}") # TODO: alternatively, add it to the graph
        node_info = self._nodes[node]

        if node_info.closed:
            if target_node == self._nodes[node].edge_data["target"][edge_idx]:
                if self._verbose_level >= 1:
                    print(f"Node {node!r} is closed, skipping test {edge_idx!r}")
                return
            else:
                if self._verbose_level >= 1:
                    print(f"Node {node!r} is closed, we perform the test only if the target node is closer to frontier than the original target node. It will allow to fix the broken transition.")
                dist_to_frontier = self._dist.get(target_node, 0) # 0 if it wasn't previously recorded (so it's in the frontier)
                prev_target_node = self._nodes[node].edge_data["target"][edge_idx]
                prev_dist_to_frontier = self._dist.get(prev_target_node, INFINITY)

                if dist_to_frontier < prev_dist_to_frontier:
                    if self._verbose_level >= 1:
                        print(f"Target node {target_node!r} is closer to frontier than the original target node {prev_target_node!r}, we perform the test")
                else:
                    if self._verbose_level >= 1:
                        print(f"Target node {target_node!r} is further from frontier than the original target node {prev_target_node!r}, we skip the test")
                    return

        # store metadata immediately
        if self._verbose_level >= 1:
            print(f"Recording action {edge_idx} from {node} to {target_node} with success {success}")

        if suspicious_transition:
            self.suspicious_transitions[(node, edge_idx, target_node)] = self.suspicious_transitions.get((node, edge_idx, target_node), 0) + 1
            print(f"Suspicious transition detected: {node, edge_idx, target_node}, count: {self.suspicious_transitions[(node, edge_idx, target_node)]}")

            if self.suspicious_transitions[(node, edge_idx, target_node)] < self.suspicious_transitions_threshold:
                print(f"It will be ignored for now, but will be allowed after {self.suspicious_transitions_threshold} attempts")
                return
            else:
                print(f"Transition is recorded as permanent")
        
        node_info.record_test(edge_idx, success, target_node)
        
        # successful hop ⇒ register edge and maybe discover a brand-new node
        if success == 1:
            if target_node is None:
                raise ValueError("target_node required when success=True")

            if target_node not in self._nodes:
                new_node = True
                if target_num_candidates is None:
                    raise ValueError(
                        "target_num_candidates required for a new node"
                    )
                self._add_new_node(target_node, target_num_candidates, group2remaining_candidate_ids=group2remaining_candidate_ids)
            else:
                new_node = False


            self._G[node].add((edge_idx, target_node))
            self._G_rev[target_node].add((edge_idx, node))

            if not self._nodes[node].has_open_group(self.active_group):
                self._close_node(node)

            if self._nodes[target_node].has_open_group(self.active_group):
                # self._tighten_from_new_source(target_node)
                self._rebuild_distances()
            else:
                self._close_node(target_node)
                self._maybe_advance_group(target_node)

        else:
            if not self._nodes[node].has_open_group(self.active_group):
                self._close_node(node)
                self._maybe_advance_group(node)

        if self._verbose_level >= 1:
            if success == 1:
                success_str = "succeeded"
            elif success == -1:
                success_str = "threw an error"
            else:
                success_str = "failed"

            print(f"\n\nNode {node!r} candidate {edge_idx!r} {success_str}:")
            print(f"Source node:\n{self._nodes[node]}")
            if success == 1:
                print(f"{'NEW' if new_node else 'Existing'} target node:\n{self._nodes[target_node]}")
        self.dump()

    def get_distance(self, node: Hashable) -> Optional[int]:
        d = self._dist.get(node)
        return None if d is None or d == float("inf") else d

    def get_next_hop(self, node: Hashable) -> Optional[Hashable]:
        # NOTE: DEPRECATED
        # Return the node itself only if it truly has open edges in the active group
        if node in self._frontier: # and self._nodes[node].has_open_group(self.active_group):
            return node
        nxt = self._next.get(node)
        if nxt is None:
            return None
        # _next may store (edge_idx, next_node); return the node only
        if isinstance(nxt, tuple) and len(nxt) == 2:
            return nxt[1]
        return nxt

    def edge_info(self, node: Hashable, edge_idx: Hashable) -> np.ndarray:
        return self._nodes[node].edge_data[edge_idx]

    def is_finished(self) -> bool:
        return not self._frontier

    @property
    def active_group(self) -> int:
        return self._active_group
    
    @property
    def empty(self) -> bool:
        return self._empty

    def _add_new_node(self, node: Hashable, 
        n_candidates: int, 
        group2remaining_candidate_ids: Optional[List[Set[int]]] = None
        ) -> None:

        if n_candidates < 1:
            raise ValueError("num_candidates must be positive")

        self._nodes[node] = NodeInfo(node, n_candidates, self._n_groups, group2remaining_candidate_ids=group2remaining_candidate_ids)
        self._G[node] = set()
        self._G_rev[node] = set()

        if self._empty:
            self._empty = False

        if self._nodes[node].has_open_group(self.active_group):
            self._frontier.add(node)
        else:
            self._close_node(node)
            self._maybe_advance_group(node)


    def _close_node(self, node: Hashable) -> None:
        node_info = self._nodes[node]
        if node_info.closed:
            return
        node_info.closed = True
        self._frontier.discard(node)
        self._rebuild_distances() # removal from frontier may increase some distances in the graph

    def _tighten_from_new_source(self, src: Hashable) -> None:
        # NOTE: is not used anymore
        dq = deque([src])
        self._dist[src] = 0
        self._nodes[src].distance = 0
        while dq:
            v = dq.popleft()
            v_dist = self._dist.get(v, INFINITY)
            for edge_idx, u in self._G_rev.get(v, ()):  # (edge_idx, source_node)
                initial_u_dist = self._dist.get(u, INFINITY)
                u_edge_data = self._nodes[u].edge_data
                u_edge_data["distance"][edge_idx] = self._nodes[v].distance + 1
                updated_u_dist = u_edge_data["distance"][u_edge_data["group"] <= self.active_group].min()
                self._nodes[u].distance = updated_u_dist
                self._dist[u] = updated_u_dist
                if updated_u_dist > initial_u_dist:
                    dq.append(u)

    def _rebuild_distances(self) -> None:
        """
        Rebuild the distances from the frontier nodes in the graph.
        """
        self._dist.clear()
        self._next.clear()
        dq = deque(self._frontier)
        for node, node_info in self._nodes.items():
            node_info.distance = INFINITY
            self._dist[node] = INFINITY
        for src in self._frontier:
            self._nodes[src].distance = 0
            self._dist[src] = 0
        while dq:
            v = dq.popleft()
            v_dist = self._dist.get(v, INFINITY)
            for edge_idx, u in self._G_rev.get(v, ()):  # (edge_idx, source_node)
                u_info = self._nodes[u]
                u_dist = self._dist.get(u, INFINITY)
                u_info.edge_data["distance"][edge_idx] = v_dist + 1
                if u_dist > u_info.edge_data["distance"][edge_idx]:
                    u_info.distance = u_info.edge_data["distance"][edge_idx]
                    self._dist[u] = u_info.edge_data["distance"][edge_idx]
                    self._next[u] = (edge_idx, v)
                    dq.append(u)

    def _maybe_advance_group(self, current_node: Hashable) -> None:
        """
        If it's not possible to reach any frontier node from the current node,
        given the current active group, advance to the next higher group id and rebuild distances.
        """

        distance = self._nodes[current_node].distance
        while distance == INFINITY and self.active_group < self._n_groups - 1:
            print(f"Node {current_node!r} is not reachable from any frontier node under {self.active_group}, advancing to the next group")

            self._active_group += 1
            self._dist.clear()
            self._next.clear()
            self._frontier.clear()

            for node, node_info in self._nodes.items():
                node_info.active_group = self.active_group
                if node_info.has_open_group(self.active_group):
                    self._frontier.add(node)
                    node_info.closed = False

            self._rebuild_distances()
            distance = self._dist.get(current_node)
        
    def dump(self) -> None:
        if self._verbose_level >= 1:
            print("=== explorer state ===")
            print("frontier :", self._frontier)
            print("N nodes  :", len(self._nodes))
            print("N edged candidates  :", sum(len(node_info.edge_data) for node_info in self._nodes.values()))
            if self._verbose_level >= 2:
                print("Graph    :", self._G)
                print("dist     :", self._dist)
                print("next hop :", self._next)
            print("======================")

    def print_all_nodes(self) -> None:
        for node_info in self._nodes.values():
            print(node_info)

    def choose_edge(self, node: Hashable, return_reasoning: bool = False) -> Hashable:
        # TODO: make it possible to choose completely random edge
        node_info = self._nodes[node]
        if node_info.has_open_group(self.active_group):
            untested_edges = []
            for group_id in range(self.active_group + 1):
                untested_edges.extend(node_info.group2remaining_candidate_ids[group_id])
            if not untested_edges:
                raise ValueError("No untested edges in the current group while the group is open")

            edge_idx = random.choice(untested_edges)
            reasoning = f"Randomly chose untested edge {edge_idx} from group {self.active_group} with {node_info.group2remaining_candidate_ids} group2candidates\n"
        else:
            lowest_dist = node_info.distance
            print(f"Lowest dist: {lowest_dist}")
            # print(f"Node info: {node_info}")
            edges_with_lowest_dist = [edge_idx for edge_idx, edge_data in enumerate(node_info.edge_data) if edge_data["distance"] <= lowest_dist and edge_data["result"] == 1 and edge_data["group"] <= self.active_group]
            edge_idx = random.choice(edges_with_lowest_dist)
            reasoning = f"Chose edge {edge_idx} with lowest dist {lowest_dist}\n"

        reasoning += f"Node info: {node_info}\n"
        

        if return_reasoning:
            return edge_idx, reasoning
        else:
            return edge_idx



def _generate_random_grid(rows: int, cols: int, density: float = 0.7, seed: int | None = None) -> np.ndarray:
    """
    Return a boolean numpy array of shape *(rows, cols)* where **True** denotes
    a traversable cell (graph node) and **False** denotes an empty/wall cell.
    The *density* parameter controls the probability of a cell being present.
    """

    rng = np.random.default_rng(seed)
    grid = rng.random((rows, cols)) < density

    # Safety: ensure at least one node exists so that we have a valid start.
    if not grid.any():
        # Force the central cell to be traversable.
        grid[rows // 2, cols // 2] = True

    return grid


# Direction vectors indexed 0-3  (U, R, D, L)
_DIRS = {
    0: (-1, 0),  # up
    1: (0, 1),   # right
    2: (1, 0),   # down
    3: (0, -1),  # left
}


def _visualize_grid(grid: np.ndarray, explorer: "GraphExplorer", start_node: tuple[int, int]) -> None:
    """
    Pretty-print the current knowledge stored inside *explorer* on top of the
    underlying *grid*.

    Legend:
        "#"  wall / empty cell
        "?"  traversable cell but undiscovered yet
        "o"  discovered & closed node (all edges tested)
        "F"  frontier node (still holds untested candidates)
        "S"  the start node
    """

    rows, cols = grid.shape
    lines: list[str] = []
    for r in range(rows):
        row_chars: list[str] = []
        for c in range(cols):
            cell = (r, c)
            if not grid[r, c]:
                row_chars.append("#")
                continue

            if cell == start_node:
                row_chars.append("S")
            elif cell in explorer._frontier:
                row_chars.append("F")
            elif cell in explorer._nodes:
                row_chars.append("o")
            else:
                row_chars.append("?")
        lines.append(" ".join(row_chars))

    print("\nCurrent explorer view:")
    print("\n".join(lines))
    print()

def _plot_grid(
    grid: np.ndarray,
    explorer: "GraphExplorer",
    start_node: tuple[int, int],
    last_node: tuple[int, int] | None = None,
    last_edge: tuple[tuple[int, int], int] | None = None,  # (node_coords, edge_idx)
    log_text: str | None = None,
    *,
    figsize: tuple[int, int] | None = None,
    frames: list[np.ndarray] | None = None,
    group_colors: dict[int, str] | None = None,
    n_groups: int = 1,
) -> None:
    """
    Render *grid* with matplotlib showing explorer's knowledge so far.

    - Walls - nothing drawn (white)
    - Undiscovered traversable cells - light grey dots
    - Discovered nodes - blue dots, frontier in orange, start in gold
    - Arrows:
        - Success (edge exists)  - green
        - Failed probe           - red
        - Untested candidate     - grey (thin)
    """

    if n_groups > 1 and group_colors is None:
        default_palette = plt.get_cmap("tab10")
        group_colors = {grp: default_palette(grp % 10) for grp in range(n_groups)}

    rows, cols = grid.shape

    if figsize is None:
        figsize = (max(4, cols), max(4, rows))

    plt.clf()
    fig = plt.gcf()
    fig.set_size_inches(*figsize)
    ax = fig.gca()

    ax.set_aspect("equal")
    # Grid lines
    ax.set_xticks(np.arange(-0.5, cols, 1))
    ax.set_yticks(np.arange(-0.5, rows, 1))
    ax.grid(True, which="both", color="lightgrey", linewidth=0.5)
    ax.set_xticklabels([])
    ax.set_yticklabels([])

    for r in range(rows):
        for c in range(cols):
            # rectangle lower-left corner at (c-0.5, r-0.5)
            facecolor = "black"  # default for walls
            if grid[r, c]:
                cell = (r, c)
                if cell == last_node:
                    facecolor = "blue"
                elif cell in explorer._frontier:
                    facecolor = "green"
                elif cell in explorer._nodes:
                    facecolor = "white"
                else:
                    facecolor = "grey"

            rect = Rectangle(
                (c - 0.5, r - 0.5),
                1,
                1,
                facecolor=facecolor,
                edgecolor="lightgrey",
                linewidth=0.5,
                alpha=0.6,
                zorder=0,
            )
            ax.add_patch(rect)

    # Overlay start marker
    ax.plot(start_node[1], start_node[0], marker="*", color="gold", markersize=12, zorder=4)

    # Draw arrows for each explored node
    for (r, c), info in explorer._nodes.items():
        for edge_idx in range(4):
            dr, dc = _DIRS[edge_idx]

            # Convert to plotting vector (remember inverted y later). Use dy = dr to correct flipped arrow issue.
            dx, dy = dc, dr

            # Decide arrow color & style with fixed length
            length_scale = 0.4  # stays inside cell borders
            succ_flag = False  # will stay False for untested or failed edges

            res = info.edge_data["result"][edge_idx] if edge_idx < len(info.edge_data) else 0
            if res != 0:
                succ_flag = (res == 1)

                # Highlight the very last tested edge in black
                if last_edge is not None and last_edge == ((r, c), edge_idx):
                    color = "black"
                    alpha = 1.0
                    lw = 2.5
                else:
                    color = "green" if succ_flag else "red"  # success green, failed red
                    alpha = 0.9
                    lw = 1.8
            else:
                group_id = int(info.edge_data["group"][edge_idx]) if edge_idx < len(info.edge_data) else 0
                color = group_colors.get(group_id, "grey") if group_colors else "grey"
                alpha = 0.8
                lw = 1.2

            arr = ax.arrow(
                c,
                r,
                dx * length_scale,
                dy * length_scale,
                head_width=0.15,
                head_length=0.15,
                fc=color,
                ec=color,
                alpha=alpha,
                linewidth=lw,
                length_includes_head=True,
                zorder=1,
            )

            # Annotate distance to frontier for successful edges
            if succ_flag:
                # Look up target from explorer graph if exists; otherwise skip distance annotation
                target = None
                for e_idx, tgt in explorer._G.get((r, c), set()):
                    if e_idx == edge_idx:
                        target = tgt
                        break
                if target is not None:
                    dist_val = explorer.get_distance(target)
                    dist_val_txt = "∞" if dist_val is None else str(dist_val)

                    text_x = c + dx * length_scale * 0.5
                    text_y = r + dy * length_scale * 0.5
                    ax.text(text_x, text_y, dist_val_txt, color="black", fontsize=8, ha="center", va="center", zorder=4)

    ax.set_xlim(-0.5, cols - 0.5)
    ax.set_ylim(-0.5, rows - 0.5)
    ax.invert_yaxis()
    plt.tight_layout()

    # Add log text overlay
    if log_text is not None:
        fig.text(0.02, 0.98, log_text, fontsize=9, va='top', ha='left', bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))

    legend_elements = [
        Patch(facecolor="black", edgecolor="lightgrey", label="Wall"),
        Patch(facecolor="grey", edgecolor="lightgrey", label="Unknown node"),
        Patch(facecolor="white", edgecolor="lightgrey", label="Discovered node"),
        Patch(facecolor="green", edgecolor="lightgrey", label="Frontier node"),
        Patch(facecolor="blue", edgecolor="lightgrey", label="Current node"),
        Patch(facecolor="gold", edgecolor="lightgrey", label="Start node"),
        Line2D([0], [0], color="black", lw=2, label="Last tested edge"),
        Line2D([0], [0], color="green", lw=2, label="Successful edge"),
        Line2D([0], [0], color="red", lw=2, label="Failed edge"),
        Line2D([0], [0], color="grey", lw=2, label="Untested candidate"),
    ]

    # Add candidate group colors to legend
    if n_groups > 1:
        for gid in range(n_groups):
            col = group_colors.get(gid, plt.get_cmap("tab10")(gid % 10)) if group_colors else plt.get_cmap("tab10")(gid % 10)
            legend_elements.append(Line2D([0], [0], color=col, lw=2, label=f"Candidate group {gid}"))

    # Reserve more space on the right for legend
    plt.subplots_adjust(right=0.65)

    # Place legend outside, based on figure coords for consistent layout
    legend_obj = fig.legend(
        handles=legend_elements,
        loc="center left",
        bbox_to_anchor=(0.68, 0.5),
        bbox_transform=fig.transFigure,
        fontsize=7,
        framealpha=0.9,
    )

    # Optionally increase z-order so legend overlays anything else
    legend_obj.set_zorder(10)

    plt.draw()
    plt.pause(0.001)

    # Capture frame for gif if requested
    if frames is not None:
        canvas = fig.canvas
        canvas.draw()
        w, h = canvas.get_width_height()
        if hasattr(canvas, "tostring_rgb"):
            buf = canvas.tostring_rgb()
            channels = 3
        elif hasattr(canvas, "tostring_argb"):
            buf = canvas.tostring_argb()
            channels = 4
        else:
            raise RuntimeError("Canvas does not support RGB extraction")

        # Account for HiDPI / retina scaling: actual buffer may be larger than (w*h*channels)
        total_px = len(buf) // channels
        scale = int(round((total_px / (w * h)) ** 0.5))
        w_scaled, h_scaled = w * scale, h * scale

        img = np.frombuffer(buf, dtype=np.uint8).reshape(h_scaled, w_scaled, channels)
        if channels == 4:
            # ARGB -> RGB
            img = img[:, :, [1, 2, 3]]
        frames.append(img.copy())


def run_grid_demo(
    rows: int = 6,
    cols: int = 6,
    density: float = 0.7,
    seed: int | None = None,
    step_sleep: float | None = None,
    n_groups: int = 1,
    group_colors: dict[int, str] | None = None,
    plot: bool = True,
    save_gif: bool = True,
    gif_path: str = "exploration.gif",
    error_chance: float = 0.3,
) -> None:
    """
    Drive *GraphExplorer* over a random grid-world and visualize every step.

    - *rows*, *cols*         - grid dimensions
    - *density*              - probability that a cell contains a node
    - *seed*                 - RNG seed for reproducibility (``None`` ⇒ random)
    - *step_sleep*           - optional ``time.sleep`` delay after each step
    """

    import time

    grid = _generate_random_grid(rows, cols, density, seed)

    # Pick a random starting node
    node_coords = list(zip(*np.where(grid)))
    start_node = random.choice(node_coords)

    candidate2group = {i: random.randint(0, n_groups-1) for i in range(4)}

    print(f"Starting exploration at {start_node} on a {rows}x{cols} grid (density={density:.2f})\n")

    gx = GraphExplorer(n_groups=n_groups, verbose_level=2)
    print(f"candidate2group: {candidate2group}\n")
    gx.initialize(start_node=start_node, num_candidates=4, group2remaining_candidate_ids=[{i for i, g in candidate2group.items() if g == gid} for gid in range(n_groups)])

    frames: list[np.ndarray] = [] if plot and save_gif else []

    if plot:
        plt.ion()

    step_counter = 0
    _visualize_grid(grid, gx, start_node)
    if plot:
        _plot_grid(grid, gx, start_node, last_node=start_node, last_edge=None, log_text=f"Group NA | Moved to {start_node}", frames=frames if save_gif else None, n_groups=n_groups, group_colors=group_colors)

        gx.dump()

    current_node = start_node
    while not gx.is_finished():
        node_info = gx._nodes[current_node]

        # If current node is exhausted, travel along the shortest path to the frontier.
        if not node_info.has_open_group(gx.active_group):
            next_hop = gx.get_next_hop(current_node)
            if next_hop is None:
                print(f"Node {current_node} is exhausted and no path to frontier. Finishing.")
                break

            # Guard against degenerate self-looping next-hop
            if next_hop == current_node:
                gx._close_node(current_node)
                gx._maybe_advance_group(current_node)
                next_hop = gx.get_next_hop(current_node)
                if next_hop is None or next_hop == current_node:
                    print(f"Node {current_node} is exhausted and stuck. Finishing.")
                    break

            print(f"Node {current_node} exhausted. Traveling to {next_hop} towards nearest frontier.")
            step_counter += 1
            current_node = next_hop

            # If we arrived at a node that is not open (due to group constraints), try advancing group
            if not gx._nodes[current_node].has_open_group(gx.active_group):
                gx._maybe_advance_group(current_node)

            _visualize_grid(grid, gx, start_node)
            if plot:
                _plot_grid(
                    grid, gx, start_node,
                    last_node=current_node,
                    last_edge=None,
                    log_text=f"Group {gx.active_group} | travel",  
                    frames=frames if save_gif else None,
                    n_groups=n_groups, group_colors=group_colors,
                )

                gx.dump()
                gx.print_all_nodes()
            if step_sleep is not None:
                time.sleep(step_sleep)
                continue

        # We are at a node with open edges. Try them until success.
        group_id = gx.active_group
        prioritized_edges = []
        for gid in range(0, group_id + 1):
            prioritized_edges.extend(list(node_info.group2remaining_candidate_ids[gid]))

        moved = False
        for edge_idx in prioritized_edges:
            step_counter += 1

            dr, dc = _DIRS[edge_idx]
            neigh = (current_node[0] + dr, current_node[1] + dc)

            is_success = 0 <= neigh[0] < rows and 0 <= neigh[1] < cols and grid[neigh]

            if error_chance > random.random():
                result_code = -1
            else:
                result_code = 1 if is_success else 0

            # Record test result
            outcome_str = "fail"
            if result_code == 1:
                outcome_str = "success"
                target_group2remaining_candidate_ids = [set() for _ in range(n_groups)]
                for i in range(4):
                    gid = random.randint(0, n_groups - 1)
                    target_group2remaining_candidate_ids[gid].add(i)
                gx.record_test(current_node, edge_idx, 1, neigh, 4, group2remaining_candidate_ids=target_group2remaining_candidate_ids)
            elif result_code == 0:
                gx.record_test(current_node, edge_idx, 0)
            else:  # result_code == -1
                outcome_str = "error"
                gx.record_test(current_node, edge_idx, -1)

            print(f"Step {step_counter}: at {current_node} tested edge {edge_idx} → {outcome_str}")

            edge_group_id = int(node_info.edge_data["group"][edge_idx]) if edge_idx < len(node_info.edge_data) else 0
            cur_dist = gx.get_distance(current_node)
            dist_txt = "∞" if cur_dist is None else str(cur_dist)
            log_line = (
                f"group={gx.active_group} node={current_node} (dist {dist_txt}) | "
                f"edge {edge_idx} (grp {edge_group_id}) → {outcome_str}"
            )
            _visualize_grid(grid, gx, start_node)
            if plot:
                _plot_grid(
                    grid, gx, start_node,
                    last_node=current_node,
                    last_edge=((current_node), edge_idx),
                    log_text=log_line,
                    frames=frames if save_gif else None,
                    n_groups=n_groups, group_colors=group_colors,
                )

                gx.dump()
                gx.print_all_nodes()
            if step_sleep is not None:
                time.sleep(step_sleep)

            # Update agent position based on outcome
            if result_code == 1:
                current_node = neigh
                moved = True
                break
            elif result_code == -1:
                print(f"Probe error at {current_node}! Returning to start node {start_node}.")
                current_node = start_node
                moved = True
                break

        if not moved:
            # All available edges were tried and failed/errored.
            # Next loop iteration will trigger the travel-to-frontier logic.
            pass

    print("Exploration finished – every node is closed and no frontier remains.")
    if plot:
        # Final frame with no current node highlight
        _plot_grid(
            grid,
            gx,
            start_node,
            last_node=None,
            frames=frames if save_gif else None,
            n_groups=n_groups,
            group_colors=group_colors,
        )

        # Keep the final plot open for the user until they close the figure.
        plt.ioff()

        if save_gif and frames:
            print(f"Saving cropped GIF with {len(frames)} frames to {gif_path} …")

            from PIL import Image, ImageChops

            pil_frames = [Image.fromarray(frame) for frame in frames]

            # Compute union bounding box of non-white areas across frames
            bbox_union = None
            white_bg = Image.new("RGB", pil_frames[0].size, (255, 255, 255))
            for im in pil_frames:
                diff = ImageChops.difference(im, white_bg)
                bbox = diff.getbbox()
                if bbox is None:
                    continue
                if bbox_union is None:
                    bbox_union = bbox
                else:
                    l1, t1, r1, b1 = bbox_union
                    l2, t2, r2, b2 = bbox
                    bbox_union = (min(l1, l2), min(t1, t2), max(r1, r2), max(b1, b2))

            # Fallback to full image if bbox detection failed
            if bbox_union is None:
                bbox_union = (0, 0) + pil_frames[0].size

            cropped_frames = [im.crop(bbox_union) for im in pil_frames]

            # Save using Pillow directly
            cropped_frames[0].save(
                gif_path,
                save_all=True,
                append_images=cropped_frames[1:],
                duration=500,
                loop=0,
            )

        plt.show()


if __name__ == "__main__":

    print("\n========== SIMPLE TEST ==========")
    gx = GraphExplorer(verbose_level=2)
        
    gx.initialize("A", 2) # node A has 2 candidates

    gx.record_test("A", 0, -1) # simulate error
    gx.record_test("A", 0, -1) # simulate error
    gx.record_test("A", 0, -1) # simulate error

    # gx.record_test("A", 0, True,  "B", 3)   # throws an error

    gx.record_test("A", 1, True, "B", 3) # now A is closed automatically


    gx.record_test("B", 0, False)
    gx.record_test("B", 1, True,  "C", 1) # discovers C 
    gx.record_test("B", 2, False) # B becomes closed

    gx.record_test("C", 0, True, "D", 4)

    gx.print_all_nodes()


    print("\n========== TEST WITH GROUPS ==========")
    gx = GraphExplorer(n_groups=3, verbose_level=2)
    gx.initialize("A", 4, group2remaining_candidate_ids=[[0, 1], [2], [3]])

    gx.record_test("A", 0, False)
    gx.record_test("A", 1, False)

    gx.record_test("A", 2, True, "B", 3, group2remaining_candidate_ids=[[0], [2], [1]])

    gx.record_test("B", 0, True, "A")

    gx.record_test("A", 2, True, "B")

    gx.record_test("B", 2, False)

    gx.print_all_nodes()


    print("\n========== GRID WORLD DEMO ==========")
    group_colors = {0: "purple", 1: "orange", 2: "grey"}
    run_grid_demo(rows=6, cols=6, density=0.7, seed=12345, step_sleep=None, plot=True, save_gif=True, gif_path="grid_exploration.gif", n_groups=3, group_colors=group_colors, error_chance=0.1)


class FrameProcessor:
    OFFSETS4: tuple[tuple[int, int], ...] = ((-1, 0), (1, 0), (0, -1), (0, 1))
    OFFSETS8: tuple[tuple[int, int], ...] = ((-1, -1), (-1, 1), (1, -1), (1, 1), (-1, 0), (1, 0), (0, -1), (0, 1))

    def __init__(self):
        self.connectivity_rank = 4
        self.status_bar_mode = "rule"
        self.status_bar_distance_threshold = 3
        self.status_bar_ratio_threshold = 5
        self.status_bar_twins_threshold = 3
        self.frame_shape = (64, 64)

        self.status_bar_color = 16
        self.minimal_width = 2
        self.maximal_width = 32
        self.non_salient_color = set([0,1,2,3,4,5])
        self.salient_color = set([6,7,8,9,10,11,12,13,14,15])

    def process_frame(self, frame: np.ndarray) -> np.ndarray:
        pass

    def segment_frame(self, frame: np.ndarray) -> tuple[np.ndarray, list[dict]]:
        """
        Segment `frame` into {self.connectivity_rank}-connected components (same color).

        NOTE: the twins identification increases complexity of the algorithm to O(n^2)

        Returns
        -------
        list[dict]
            One dict per component with keys
            - bounding_box : (x1, y1, x2, y2)   # inclusive pixel coords
            - color        : int                # original greyscale value
            - area         : int                # pixel count
            - is_rectangle : bool               # fully fills its bounding box
            - number_of_twins : int             # number of other components considered twins
            - twin_ids     : list[int]          # ids (1-based) of those twins
                NOTE: here we don't check shapes of the twins thoroughly

        """

        h, w = frame.shape
        label_map = np.zeros((h, w), dtype=int) - 1 # -1 = unvisited
        components: list[dict] = []
        cid = -1                                          # component id counter

        offsets = self.OFFSETS4 if self.connectivity_rank == 4 else self.OFFSETS8

        # --- first pass: flood-fill each blob ---------------------------------
        for y in range(h):
            for x in range(w):
                if label_map[y, x] != -1:                      # already labelled
                    continue
                cid += 1
                color = int(frame[y, x])
                q = deque([(y, x)])
                label_map[y, x] = cid

                min_x = max_x = x
                min_y = max_y = y
                area = 0

                while q:                                 # BFS
                    cy, cx = q.popleft()
                    area += 1
                    min_x, max_x = min(min_x, cx), max(max_x, cx)
                    min_y, max_y = min(min_y, cy), max(max_y, cy)

                    for dy, dx in offsets:
                        ny, nx = cy + dy, cx + dx
                        if (
                            0 <= ny < h and 0 <= nx < w
                            and label_map[ny, nx] == -1 # not visited
                            and frame[ny, nx] == color
                        ):
                            label_map[ny, nx] = cid
                            q.append((ny, nx))

                # rectangle test
                rect_area = (max_x - min_x + 1) * (max_y - min_y + 1)
                is_rect = area == rect_area

                components.append(
                    dict(
                        bounding_box=(min_x, min_y, max_x, max_y),
                        color=color,
                        area=area,
                        is_rectangle=is_rect,
                    )
                )

        # --- second pass: identify twins --------------------------------------
        # here: simple rule → same area, same rectangle status, and same color
        for i, comp in enumerate(components):
            twins = [
                j
                for j, other in enumerate(components)
                if i != j # skip self
                and other["area"] == comp["area"]
                and other["is_rectangle"] == comp["is_rectangle"]
                and other["color"] == comp["color"]
            ]
            comp["number_of_twins"] = len(twins)
            comp["twin_ids"] = twins

        return label_map, components

    def identify_status_bars(self, segmented_frame: np.ndarray, frame_segments: list[dict]) -> tuple[list[list[dict]] | None, np.ndarray]:
        """
        Identify the status bars from the frame segments
        Return a list of dictionaries and a frame mask.
        The list of dictionaries is the same as the input list of dictionaries in frame_segments, but with "id" key added.
        The frame mask is a binary mask where the status bars are 1 and the rest are 0.
        """
        if self.status_bar_mode == "crude":
            status_bar_mask = self.identify_status_bars_crude()
            status_bar_segments_list = None
        elif self.status_bar_mode == "rule" or self.status_bar_mode == "move":
            status_bar_segments_list, status_bar_mask = self.identify_status_bars_with_rule(segmented_frame, frame_segments)
            if self.status_bar_mode == "move":
                raise NotImplementedError("'move' mode is not implemented yet")
        else:
            raise ValueError(f"Invalid status bar mode: {self.status_bar_mode}")
        return status_bar_segments_list, status_bar_mask

    def identify_status_bars_crude(self) -> np.ndarray:
        status_bar_mask = np.zeros(self.frame_shape)
        status_bar_mask[:self.status_bar_distance_threshold, :] = 1
        status_bar_mask[-self.status_bar_distance_threshold:, :] = 1
        status_bar_mask[:, :self.status_bar_distance_threshold] = 1
        status_bar_mask[:, -self.status_bar_distance_threshold:] = 1
        return status_bar_mask
       
    def identify_status_bars_with_rule(self, segmented_frame: np.ndarray, frame_segments: list[dict]) -> tuple[list[list[dict]], np.ndarray]:
        """
        Identify the status bars from the frame segments
        Return a list of dictionaries and a frame mask.
        The list of dictionaries is the same as the input list of dictionaries in frame_segments, but with "id" key added.
        The frame mask is a binary mask where the status bars are 1 and the rest are 0.
        """

        # modes:
            # crude: remove all screen edges 
            # rule: rule-based
            # move: rule-based + movement after the first action 


        # the rules are:
            # the status bars are close to the edges of the screen
            # they can be in any orientation
            # the can be duplicated from both sides of the screen
            # there are 2 types of status bars:
                # 1. the line 
                # 2. the dots, for the dots there should be at least 3 twins


        checked_segment_ids = set()
        status_bar_segment_ids_list = [] # list[list[int]]
        for i, segment in enumerate(frame_segments):

            status_bar_segment_ids = [i]

            if i in checked_segment_ids:
                continue
            checked_segment_ids.add(i)
            on_edge_list = self.check_segment_fully_on_edge(segment, edges=['any'])
            if len(on_edge_list) == 0:
                continue
            directions = []
            if 'left' in on_edge_list or 'right' in on_edge_list:
                directions.append('vertical')
            if 'top' in on_edge_list or 'bottom' in on_edge_list:
                directions.append('horizontal')
            if len(directions) == 2:
                direction = 'any'
            else:
                direction = directions[0]
            is_long_ratio = self.check_segment_ratio(segment, direction=direction)  

            if not is_long_ratio:
                twin_ids_on_edge_list = self.segment_twins_on_edge(segment, frame_segments)
                for twin_id in twin_ids_on_edge_list:
                    checked_segment_ids.add(twin_id)
                if len(twin_ids_on_edge_list) + 1 < self.status_bar_twins_threshold:
                    continue
                status_bar_segment_ids.extend(twin_ids_on_edge_list)

            status_bar_segment_ids_list.append(status_bar_segment_ids)

        status_bar_segments_list = []
        status_bar_mask = np.zeros(segmented_frame.shape, dtype=bool)

        for i, status_bar_segment_ids in enumerate(status_bar_segment_ids_list):
            status_bar_segments = []
            for status_bar_segment_id in status_bar_segment_ids:
                status_bar_mask[segmented_frame == status_bar_segment_id] = 1

                status_bar_segments.append(frame_segments[status_bar_segment_id])
            status_bar_segments_list.append(status_bar_segments)

        return status_bar_segments_list, status_bar_mask

    def check_segment_fully_on_edge(self, segment: dict, edges: list[str] | None = None) -> list[str]:
        """
        Check if the segment is fully on the edge of the screen
        """
        x1, y1, x2, y2 = segment["bounding_box"]
        if edges is None:
            edges = ['any']
        for edge in edges:
            assert edge in ['any', 'left', 'right', 'top', 'bottom']

        result = []

        if 'left' in edges or 'any' in edges:
            max_x = max(x1, x2)
            if max_x < self.status_bar_distance_threshold:
                result.append('left')
        if 'right' in edges or 'any' in edges:
            min_x = min(x1, x2)
            if min_x > self.frame_shape[1] - self.status_bar_distance_threshold:
                result.append('right')
        if 'top' in edges or 'any' in edges:
            max_y = max(y1, y2)
            if max_y < self.status_bar_distance_threshold:
                result.append('top')
        if 'bottom' in edges or 'any' in edges:
            min_y = min(y1, y2)
            if min_y > self.frame_shape[0] - self.status_bar_distance_threshold:
                result.append('bottom')
        # NOTE: there can be some mess with the y-axis direction (should it start from the top or the bottom), need to double check
        return result

    def check_segment_ratio(self, segment: dict, direction: str | None = None) -> bool:
        """
        Check if the segment is a status bar
        """
        if direction is None:
            direction = 'any'
        assert direction in ['any', 'horizontal', 'vertical']

        x_length, y_length = segment["bounding_box"][2] - segment["bounding_box"][0] + 1, segment["bounding_box"][3] - segment["bounding_box"][1] + 1
        x_to_y_ratio = x_length / y_length
        if x_to_y_ratio >= self.status_bar_ratio_threshold and direction in ('any', 'horizontal'):
            return True
        if x_to_y_ratio <= 1 / self.status_bar_ratio_threshold and direction in ('any', 'vertical'):
            return True
        return False

    def segment_twins_on_edge(self, segment: dict, frame_segments: list[dict], edges: list[str] | None = None) -> list[int]:
        """
        Check if the segment has twins on the same edge
        """

        if edges is None:
            edges = self.check_segment_fully_on_edge(segment, edges=['any'])
            if len(edges) == 0:
                return []

        twins = []
        for twin_id in segment["twin_ids"]:
            twin = frame_segments[twin_id]
            twin_edges = self.check_segment_fully_on_edge(twin, edges=edges)
            if len(twin_edges) > 0:
                twins.append(twin_id)
        
        return twins
        
    def visualize_components(self, frame: np.ndarray, components: list[dict], *, cmap: str = "nipy_spectral",
                             save_path: str = "components.png", click_points: list[tuple[int, int]] | None = None
    ) -> None:
        """
        Show the frame with every connected component marked and
        print a short description for each one.

        Parameters
        ----------
        frame : np.ndarray
            The original HxW greyscale (label-value) image.
        components : list[dict]
            Output of `segment_frame()`.
        cmap : str, optional
            Matplotlib colour map for the background image.  Default is *nipy_spectral*.
        """
        if frame.ndim != 2:
            raise ValueError("`frame` must be a 2-D array")

        fig, ax = plt.subplots(figsize=(6, 6))
        ax.imshow(frame, cmap=cmap, interpolation="nearest")
        ax.set_axis_off()

        # Plot bounding box + id at the centroid of each blob
        for idx, comp in enumerate(components, start=1):
            x1, y1, x2, y2 = comp["bounding_box"]
            w = x2 - x1 + 1
            h = y2 - y1 + 1

            # draw bounding box
            ax.add_patch(
                Rectangle(
                    (x1 - 0.5, y1 - 0.5),
                    w,
                    h,
                    edgecolor="white",
                    facecolor="none",
                    linewidth=1.2,
                )
            )

            # annotate with id number
            cx = x1 + w / 2.0
            cy = y1 + h / 2.0
            ax.text(
                cx,
                cy,
                str(idx),
                color="white",
                fontsize=8,
                ha="center",
                va="center",
                bbox=dict(
                    boxstyle="round,pad=0.2", facecolor="black", alpha=0.6, lw=0
                ),
            )

        if click_points is not None:
            for x, y in click_points:
                ax.plot(x, y, 'ro')

        plt.tight_layout()
        plt.savefig(save_path)

        # ---------------------------------------------------------------------
        # Console description
        # ---------------------------------------------------------------------
        for idx, comp in enumerate(components, start=1):
            bb = comp["bounding_box"]
            print(
                f"Component {idx}: "
                f"colour={comp['color']:>2}, "
                f"area={comp['area']:>4}, "
                f"bbox=(x1={bb[0]}, y1={bb[1]}, x2={bb[2]}, y2={bb[3]}), "
                f"rect={comp['is_rectangle']}, "
                f"twins={comp['number_of_twins']} "
                f"{'('+','.join(map(str,comp['twin_ids']))+')' if comp['twin_ids'] else ''}"
            )
    
    def hash_frame(self, frame: np.ndarray) -> str:
        """
        Deterministic 128-bit hash for an integer-valued NumPy array whose
        elements are in the range 0 … 15 (4 bits).

        • Compact: packs two elements per byte before hashing  
        • Stable: identical digest across Python versions & interpreter restarts  
        • Shape-aware: (m, n) and (n, m) views do NOT collide  
        • Dependency-free: only stdlib hashlib
        """
        # TODO: maybe just convert a matrix to a number and store it
        frame = np.asarray(frame, dtype=np.uint8, order='C')

        # ---- pack two 4-bit values into each byte ---------------------------
        flat = frame.ravel()
        if flat.size & 1:                       # pad to even length
            flat = np.concatenate([flat, np.zeros(1, dtype=np.uint8)])
        packed = (flat[0::2] << 4) | (flat[1::2] & 0x0F)
        payload = packed.tobytes()

        # ---- hash with Blake2B (128-bit digest) -----------------------------
        shape_tag = frame.shape.__repr__().encode()
        return hashlib.blake2b(payload,
                            digest_size=16,   # 128 bits
                            person=shape_tag  # embeds the shape
                            ).hexdigest()


    def frame_segments_to_action_groups(self, frame_segments: list[dict], n_groups: int) -> list[list[int]]:
        """
        Assign actions to groups
        """
        group_0_segments = set()
        group_1_segments = set()
        group_2_segments = set()
        group_3_segments = set()
        group_4_segments = set()

        for segment_id, segment in enumerate(frame_segments):
            x_width, y_width = segment["bounding_box"][2] - segment["bounding_box"][0] + 1, segment["bounding_box"][3] - segment["bounding_box"][1] + 1
            is_salient = segment["color"] in self.salient_color
            is_medium_width = self.minimal_width <= x_width <= self.maximal_width and self.minimal_width <= y_width <= self.maximal_width
            is_status_bar = segment["color"] == self.status_bar_color

            assert n_groups == 5, "Only 5 groups are supported for now"

            if is_salient and is_medium_width:
                group_0_segments.add(segment_id)
            elif is_medium_width:
                group_1_segments.add(segment_id)
            elif is_salient:
                group_2_segments.add(segment_id)
            elif not is_status_bar:
                group_3_segments.add(segment_id)
            else:
                group_4_segments.add(segment_id)

        groups2segments = [group_0_segments, group_1_segments, group_2_segments, group_3_segments, group_4_segments]
        # groups2segments = groups2segments[::-1] # NOTE: temporary to check the robustness 

        return groups2segments



# FIXME: hash keyerror when level_up
# TODO: check how hash decision-making generally works

# TODO then: add some value propagation with transitions

# TODO: switch strategies on resets, e.g.:
# - random action selection
# - favor new actions


# TODO: for an action that resulted in a game over, save that it creates a transition, but the frame should be `0`. And then maybe treat it as a basic transition?
# Hmm, but the distance should be indified or set to constant?

class MyAgent(Agent):
    """Faithful port of dolphin-in-a-coma HeuristicAgent.

    The reference splits state across choose_action AND its own main() loop:
      - main() wraps choose_action in try/except: on error it sets
        failed=True, level_up=True and REPLAYS last_action_object (NOT RESET).
      - main() sets level_up=True when score increases (status_bar_mask=None),
        else level_up=False once a status_bar_mask exists.

    The Kaggle/local base Agent.main() has none of that bookkeeping, so we
    replicate it inside choose_action. Level-up is detected via
    `levels_completed` (env adaptation) instead of `frame.score`.
    """

    MAX_ACTIONS = 1000000

    SIMPLE_ACTION_ID2GAME_ACTION = {
        1: GameAction.ACTION1,
        2: GameAction.ACTION2,
        3: GameAction.ACTION3,
        4: GameAction.ACTION4,
        5: GameAction.ACTION5,
    }

    N_GROUPS = 5

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        seed = int(time.time() * 1000000) + hash(self.game_id) % 1000000
        random.seed(seed)
        self.frame_processor = FrameProcessor()
        self.status_bar_mask = None
        self.hashed_frame2action_results = {}
        self.hashed_frame2transitions = {}
        self.last_hashed_frame = None
        self.last_action = None
        self.last_action_object = GameAction.RESET
        self.last_levels_completed = 0
        self.failed = False
        self.level_up = True
        self.last_transition_suspicious = False
        self.level_first_frame = None
        self.favor_frontier_search = True
        self.favor_new_actions = False
        self.graph_explorer = GraphExplorer(verbose_level=0, n_groups=self.N_GROUPS)
        # Circuit breaker: the reference's replay-on-exception path can
        # oscillate forever on some games (e.g. cd82) where the same edge
        # keeps re-asserting "Edge result must be untested". The reference
        # comment (heuristic_agent.py:469) calls this "re-initializing the
        # GraphExplorer" -- which its level_up rebuild path actually does.
        # We make that explicit: after N consecutive recovery failures, force
        # a clean level rebuild (level_up) instead of replaying indefinitely.
        self.consecutive_failures = 0
        self.max_consecutive_failures = 8

    def is_done(self, frames, latest_frame):
        try:
            return latest_frame.state is GameState.WIN
        except Exception:
            return True

    def get_frame_transition_data(self, hashed_frame, num_actions):
        # Mirror reference: zeros for results, [0]*n for transitions.
        curr = self.hashed_frame2action_results.get(hashed_frame, None)
        if curr is None:
            self.hashed_frame2action_results[hashed_frame] = np.zeros(num_actions)
            curr = self.hashed_frame2action_results[hashed_frame]
        trans = self.hashed_frame2transitions.get(hashed_frame, None)
        if trans is None:
            self.hashed_frame2transitions[hashed_frame] = [0] * num_actions
            trans = self.hashed_frame2transitions[hashed_frame]
        return curr, trans

    def choose_action(self, frames, latest_frame):
        # ---- replicate reference main()-loop level/exception bookkeeping ----
        # The reference main() set level_up from the score delta AFTER the
        # previous take_action. We do the equivalent here from levels_completed
        # observed on the incoming frame (env adaptation, intentionally kept).
        try:
            cur_levels = getattr(latest_frame, "levels_completed", 0) or 0
        except Exception:
            cur_levels = 0
        if cur_levels > self.last_levels_completed:
            # score went up == reference: level_up=True, status_bar_mask=None
            self.level_up = True
            self.status_bar_mask = None
            self.consecutive_failures = 0
        elif self.status_bar_mask is not None and not self.failed:
            # reference: elif status_bar_mask is not None -> level_up=False.
            # Reference quirk: its main() runs this elif even after an
            # exception, clobbering the recovery level_up. We additionally
            # guard with `not self.failed` so a pending recovery rebuild is
            # not silently cancelled (this is what made cd82 oscillate).
            self.level_up = False
        self.last_levels_completed = cur_levels

        try:
            action = self._explore_choose(frames, latest_frame)
            # Successful planning step: clear the circuit breaker and the
            # recovery flag (reference clears `failed` in its NOT_PLAYED
            # branch only; we also clear it on any clean step so the
            # level_up=False elif can resume normal operation).
            self.consecutive_failures = 0
            self.failed = False
            return action
        except Exception as e:
            # Reference main() recovery: failed=True, level_up=True, and
            # REPLAY last_action_object (NOT a forced RESET -- forcing RESET
            # throws away level progress and is the biggest level-loss bug).
            self.consecutive_failures += 1
            self.failed = True
            self.level_up = True
            if self.consecutive_failures >= self.max_consecutive_failures:
                # Bounded escalation: the replay loop is stuck. Drop the
                # corrupt graph + per-frame caches so the next call rebuilds
                # the level cleanly (reference's "re-initializing the
                # GraphExplorer"). RESET also breaks any game-side loop.
                logger.warning(
                    f"v24b stuck ({self.consecutive_failures} consecutive "
                    f"failures: {e}); hard level rebuild + RESET"
                )
                self.consecutive_failures = 0
                self.last_hashed_frame = None
                self.last_action = None
                self.hashed_frame2action_results = {}
                self.hashed_frame2transitions = {}
                self.status_bar_mask = None
                try:
                    self.graph_explorer.reset()
                except Exception:
                    pass
                self.last_action_object = GameAction.RESET
                return GameAction.RESET
            logger.warning(
                f"v24b choose_action error: {e}; re-init explorer (replay last action)"
            )
            return self.last_action_object

    def _explore_choose(self, frames, latest_frame):
        # ---- pre-frame state handling (reference choose_action top) --------
        if latest_frame.state in [GameState.NOT_PLAYED]:
            self.last_hashed_frame = None
            self.last_action = None
            if self.failed:
                self.level_up = True
                self.failed = False
            return GameAction.RESET

        if latest_frame.state in [GameState.GAME_OVER]:
            self.last_transition_suspicious = True
            return GameAction.RESET

        latest_frame_np = np.array(latest_frame.frame, dtype=np.uint8)
        if latest_frame_np.size == 0:
            # reference falls through to a fully random action on empty frame
            return random.choice([a for a in GameAction if a is not GameAction.RESET])

        num_frames = latest_frame_np.shape[0]
        latest_frame_np = latest_frame_np[-1].copy()

        if self.level_up:
            seg_for_status, segs_for_status = self.frame_processor.segment_frame(latest_frame_np)
            _, status_mask = self.frame_processor.identify_status_bars(
                seg_for_status, segs_for_status
            )
            self.status_bar_mask = status_mask
            self.hashed_frame2action_results = {}
            self.hashed_frame2transitions = {}

        # reference applies the mask unconditionally; guard only for the very
        # first call before any mask exists.
        if self.status_bar_mask is not None:
            latest_frame_np[self.status_bar_mask] = 16

        segmented_frame, frame_segments = self.frame_processor.segment_frame(latest_frame_np)
        available_actions = list(getattr(latest_frame, "available_actions", []) or [])

        num_arrow_actions = 0
        num_click_actions = 0
        num_actions = 0
        arrow_actions = []
        if 6 in available_actions:
            num_actions += len(frame_segments)
            num_click_actions += len(frame_segments)
            action_groups = self.frame_processor.frame_segments_to_action_groups(
                frame_segments, n_groups=self.N_GROUPS
            )
        else:
            action_groups = [set() for _ in range(self.N_GROUPS)]

        for action_id in available_actions:
            if action_id in self.SIMPLE_ACTION_ID2GAME_ACTION:
                arrow_actions.append(self.SIMPLE_ACTION_ID2GAME_ACTION[action_id])
                action_groups[0].add(num_actions)
                num_actions += 1
                num_arrow_actions += 1

        latest_frame_np[latest_frame_np == 16] = 0
        hashed_frame = self.frame_processor.hash_frame(latest_frame_np)

        if self.level_up and self.favor_frontier_search:
            self.level_first_frame = hashed_frame
            self.graph_explorer.reset()
            self.graph_explorer.initialize(
                start_node=hashed_frame,
                num_candidates=num_actions,
                group2remaining_candidate_ids=action_groups,
            )

        # suspicious_transition / transition are referenced later (recovery
        # block) even when the level_up branch is taken; init like reference.
        suspicious_transition = False
        transition = False

        if self.last_hashed_frame is not None and not self.level_up:
            transition = hashed_frame != self.last_hashed_frame
            suspicious_transition = hashed_frame == self.level_first_frame and num_frames > 1

            if self.last_transition_suspicious:
                suspicious_transition = True
                self.last_transition_suspicious = False

            old_value = self.hashed_frame2action_results[self.last_hashed_frame][self.last_action]

            if transition:
                self.hashed_frame2action_results[self.last_hashed_frame][self.last_action] = 1
                self.hashed_frame2transitions[self.last_hashed_frame][self.last_action] = hashed_frame
            else:
                self.hashed_frame2action_results[self.last_hashed_frame][self.last_action] = -1
                self.hashed_frame2transitions[self.last_hashed_frame][self.last_action] = None

            if self.favor_frontier_search:
                self.graph_explorer.record_test(
                    self.last_hashed_frame, self.last_action, transition, hashed_frame,
                    target_num_candidates=num_actions,
                    group2remaining_candidate_ids=action_groups,
                    suspicious_transition=suspicious_transition,
                )

            # Reference consistency check: a contradictory re-observation of an
            # already-recorded edge raises -> caught by choose_action -> clean
            # level rebuild (failed/level_up). Restored from reference l.297-298.
            new_value = self.hashed_frame2action_results[self.last_hashed_frame][self.last_action]
            if old_value != 0 and old_value != new_value:
                raise ValueError(f"Old value {old_value} != new value {new_value}")

        new_frame = hashed_frame not in self.hashed_frame2action_results
        curr_frame_action_results, curr_frame_transitions = self.get_frame_transition_data(
            hashed_frame, num_actions
        )

        if self.favor_frontier_search:
            # Recovery: if an exception occurred before the transition was
            # recorded, the node may be missing. Re-record with the REAL
            # transition value (reference passes `transition`, NOT a forced 1).
            if hashed_frame not in self.graph_explorer._nodes:
                self.graph_explorer.record_test(
                    self.last_hashed_frame, self.last_action, transition, hashed_frame,
                    target_num_candidates=num_actions,
                    group2remaining_candidate_ids=action_groups,
                    suspicious_transition=suspicious_transition,
                )

        # Reference clears level_up only AFTER the frame is fully processed and
        # a status_bar_mask exists (its main() does this via the `elif`). We've
        # now built/used everything for this frame, so clear it here.
        self.level_up = False

        available_actions_arr = np.where(curr_frame_action_results != -1)[0]
        new_actions = np.where(curr_frame_action_results == 0)[0]

        if len(available_actions_arr) == 0:
            # Reference raises ValueError here -> level-rebuild recovery.
            raise ValueError(f"No available actions found for frame {hashed_frame}")

        reasoning = ""
        if self.favor_frontier_search:
            action_id, reasoning = self.graph_explorer.choose_edge(
                hashed_frame, return_reasoning=True
            )
        elif len(new_actions) > 0 and self.favor_new_actions:
            action_id = random.choice(new_actions)
        else:
            action_id = random.choice(available_actions_arr)

        action_id = int(action_id)

        if action_id < num_click_actions:
            arrow_control = False
        else:
            arrow_control = True

        if not arrow_control:
            segment_mask = segmented_frame == action_id
            segment_points = np.argwhere(segment_mask)
            if len(segment_points) == 0:
                # Defensive: segment id not present (should not happen given
                # action_id < num_click_actions == len(frame_segments)).
                seg = frame_segments[action_id] if action_id < len(frame_segments) else None
                if seg is not None:
                    x1, y1, x2, y2 = seg["bounding_box"]
                    x, y = (x1 + x2) // 2, (y1 + y2) // 2
                else:
                    x, y = 32, 32
            else:
                segment_point = segment_points[random.randint(0, len(segment_points) - 1)]
                y, x = segment_point
            action = GameAction.ACTION6
            action.set_data({"x": int(x), "y": int(y)})
            reasoning += f"\nClicking on a segment {action_id}, x: {x}, y: {y}"
            action.reasoning = {
                "desired_action": f"{action.value}",
                "my_reason": reasoning,
            }
        else:
            action = arrow_actions[action_id - num_click_actions]
            reasoning += f"Arrow control: {action} for frame {hashed_frame}"
            action.reasoning = {
                "desired_action": f"{action.value}",
                "my_reason": reasoning,
            }

        self.last_hashed_frame = hashed_frame
        self.last_action = action_id
        self.last_action_object = action
        return action
