import math
import networkx as nx
import matplotlib.pyplot as plt

class Node:
    """
    Represents a node in a graph, with an ID, position, type, and neighbors.

    Attributes:
        id (any): Unique identifier for the node.
        position (tuple): Coordinates of the node, typically (x, y).
        typ (str): Type of the node, e.g., "line", "centroid".
        neighbors (set): A set of neighboring node IDs.
    """

    def __init__(self, node_id, position, typ):
        """
        Initializes a Node instance.

        Args:
            node_id (any): Unique identifier.
            position (tuple): Coordinates (x, y).
            typ (str): Type of the node.
        """
        self.id = node_id
        self.position = position
        self.typ = typ
        self.neighbors = set()

    def add_neighbor(self, other_id):
        """
        Adds another node ID to this node's neighbors.

        Args:
            other_id (any): The ID of the neighboring node.
        """
        self.neighbors.add(other_id)


class Graph:
    """
    Represents an undirected graph with nodes and edges.

    Nodes are stored in a dictionary and can be connected with edges.
    """

    def __init__(self):
        """Initializes an empty Graph."""
        self.nodes = {}

    def add_node(self, node_id, position, typ):
        """
        Adds a node to the graph.

        Args:
            node_id (any): Unique identifier.
            position (tuple): (x, y) position.
            typ (str): Type of the node.
        """
        if node_id not in self.nodes:
            self.nodes[node_id] = Node(node_id, position, typ)

    def add_edge(self, id1, id2):
        """
        Adds an undirected edge between two nodes.

        Args:
            id1 (any): First node ID.
            id2 (any): Second node ID.
        """
        if id1 == id2:
            return
        if id1 in self.nodes and id2 in self.nodes:
            self.nodes[id1].add_neighbor(id2)
            self.nodes[id2].add_neighbor(id1)

    def get_neighbors(self, node_id):
        """
        Returns the neighbors of a given node.

        Args:
            node_id (any): Node ID.

        Returns:
            set: Neighbor node IDs.
        """
        return self.nodes[node_id].neighbors

    def is_beginning_node(self, node_id):
        """
        Determines if a node is a starting point, defined by being a centroid or adjacent to one.

        Args:
            node_id (any): Node ID.

        Returns:
            bool: True if it's a beginning node.
        """
        if node_id not in self.nodes:
            return False
        if self.nodes[node_id].typ == "centroid":
            return True
        return any(self.nodes[n].typ == "centroid" for n in self.nodes[node_id].neighbors)

    def remove_node(self, node_id):
        """
        Removes a node and all references to it from its neighbors.

        Args:
            node_id (any): Node ID.
        """
        if node_id not in self.nodes:
            return
        neighbors = list(self.nodes[node_id].neighbors)
        for neighbor_id in neighbors:
            self.nodes[neighbor_id].neighbors.discard(node_id)
        del self.nodes[node_id]

    def __getitem__(self, node_id):
        """
        Enables dictionary-style access to nodes.

        Args:
            node_id (any): Node ID.

        Returns:
            Node: The requested node.
        """
        return self.nodes[node_id]

    def edges(self):
        """
        Yields all unique undirected edges in the graph.

        Yields:
            tuple: Pair of node IDs representing an edge.
        """
        seen = set()
        for node in self.nodes.values():
            for neighbor in node.neighbors:
                edge = tuple(sorted([node.id, neighbor]))
                if edge not in seen:
                    seen.add(edge)
                    yield edge

    def remove_edge(self, id1, id2):
        """
        Removes an undirected edge between two nodes.

        Args:
            id1 (any): First node ID.
            id2 (any): Second node ID.
        """
        if id1 in self.nodes:
            self.nodes[id1].neighbors.discard(id2)
        if id2 in self.nodes:
            self.nodes[id2].neighbors.discard(id1)

    def are_connected(self, start_id, other_id, visited=None):
        """
        Checks if two nodes are connected using DFS.

        Args:
            start_id (any): Starting node ID.
            other_id (any): Target node ID.
            visited (set, optional): Set of already visited node IDs.

        Returns:
            tuple: (visited set, boolean indicating connection).
        """
        if start_id == other_id:
            return visited, True
        if visited is None:
            visited = set()
        if start_id not in self.nodes:
            return visited, False

        visited.add(start_id)
        for neighbor_id in self.nodes[start_id].neighbors:
            if neighbor_id not in visited:
                visited, found = self.are_connected(neighbor_id, other_id, visited)
                if found:
                    return visited, True
        return visited, False

    def merge_close_nodes(self, threshold=5):
        """
        Merges nodes of type 'line' that are within a certain distance.

        Args:
            threshold (float): Maximum distance to consider nodes for merging.
        """
        visited = set()
        to_merge = []

        node_ids = list(self.nodes.keys())
        for i in range(len(node_ids)):
            id1 = node_ids[i]
            if id1 in visited or self.nodes[id1].typ == "centroid":
                continue

            group = [id1]
            x1, y1 = self.nodes[id1].position

            for j in range(i + 1, len(node_ids)):
                id2 = node_ids[j]
                if id2 in visited or self.nodes[id2].typ == "centroid":
                    continue
                x2, y2 = self.nodes[id2].position
                dist = math.hypot(x2 - x1, y2 - y1)
                if dist <= threshold:
                    group.append(id2)
                    visited.add(id2)

            if len(group) > 1:
                visited.update(group)
                to_merge.append(group)

        for group in to_merge:
            positions = [self.nodes[nid].position for nid in group]
            avg_x = int(sum(p[0] for p in positions) / len(positions))
            avg_y = int(sum(p[1] for p in positions) / len(positions))
            new_typ = "line"

            new_neighbors = set()
            for nid in group:
                new_neighbors.update(self.nodes[nid].neighbors)
            new_neighbors.difference_update(group)

            for nid in group:
                self.remove_node(nid)

            new_id = (avg_x, avg_y)
            self.add_node(new_id, (avg_x, avg_y), new_typ)
            for neighbor in new_neighbors:
                self.add_edge(new_id, neighbor)


def draw_graph_ignoring_position(custom_graph, k=0.25):
    """
    Visualizes the graph using spring layout (ignores actual node positions).

    Args:
        custom_graph (Graph): The custom graph object.
        k (float): Spring layout parameter controlling spacing.
    """
    G = nx.Graph()
    colors = []

    for node_id, node in custom_graph.nodes.items():
        G.add_node(node_id)
        if node.typ == "line":
            colors.append("blue")
        elif node.typ == "centroid":
            colors.append("red")
        else:
            colors.append("gray")

    for id1, id2 in custom_graph.edges():
        G.add_edge(id1, id2)

    pos = nx.spring_layout(G, seed=42, k=k)

    plt.figure(figsize=(8, 6))
    nx.draw(G, pos, with_labels=True, node_color=colors,
            node_size=500, font_size=10, edge_color='black')
    plt.title("Strukturbasierte Graph-Visualisierung")
    plt.axis("off")
    plt.show()


def draw_graph_considering_possition(custom_graph):
    """
    Visualizes the graph using actual node positions (e.g., for spatial graphs).

    Args:
        custom_graph (Graph): The custom graph object.
    """
    G = nx.Graph()
    pos = {}
    colors = []

    for node_id, node in custom_graph.nodes.items():
        G.add_node(node_id)
        pos[node_id] = node.position
        if node.typ == "line":
            colors.append("blue")
        elif node.typ == "arrowhead":
            colors.append("red")
        else:
            colors.append("gray")

    for id1, id2 in custom_graph.edges():
        G.add_edge(id1, id2)

    plt.figure(figsize=(8, 6))
    nx.draw(G, pos, with_labels=True, node_color=colors,
            node_size=500, font_size=10, edge_color='black')
    plt.title("Graph Visualisierung")
    plt.axis("equal")
    plt.show()
