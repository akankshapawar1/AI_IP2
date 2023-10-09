import heapq
import math
import random
import numpy as np
import sys


class Graph:
    """A graph connects nodes (vertices) by edges (links). Each edge can also
    have a length associated with it. The constructor call is something like:
        g = Graph({'A': {'B': 1, 'C': 2})
    this makes a graph with 3 nodes, A, B, and C, with an edge of length 1 from
    A to B,  and an edge of length 2 from A to C. You can also do:
        g = Graph({'A': {'B': 1, 'C': 2}, directed=False)
    This makes an undirected graph, so inverse links are also added. The graph
    stays undirected; if you add more links with g.connect('B', 'C', 3), then
    inverse link is also added. You can use g.nodes() to get a list of nodes,
    g.get('A') to get a dict of links out of A, and g.get('A', 'B') to get the
    length of the link from A to B. 'Lengths' can actually be any object at
    all, and nodes can be any hashable object."""

    def __init__(self, graph_dict=None, directed=True):
        self.graph_dict = graph_dict or {}
        self.directed = directed
        if not directed:
            self.make_undirected()

    def make_undirected(self):
        """Make a digraph into an undirected graph by adding symmetric edges."""
        for a in list(self.graph_dict.keys()):
            for (b, dist) in self.graph_dict[a].items():
                self.connect1(b, a, dist)

    def connect(self, A, B, distance=1):
        """Add a link from A and B of given distance, and also add the inverse
        link if the graph is undirected."""
        self.connect1(A, B, distance)
        if not self.directed:
            self.connect1(B, A, distance)

    def connect1(self, A, B, distance):
        """Add a link from A to B of given distance, in one direction only."""
        self.graph_dict.setdefault(A, {})[B] = distance

    def get(self, a, b=None):
        """Return a link distance or a dict of {node: distance} entries.
        .get(a,b) returns the distance or None;
        .get(a) returns a dict of {node: distance} entries, possibly {}."""
        links = self.graph_dict.setdefault(a, {})
        if b is None:
            return links
        else:
            return links.get(b)

    def nodes(self):
        """Return a list of nodes in the graph."""
        s1 = set([k for k in self.graph_dict.keys()])
        s2 = set([k2 for v in self.graph_dict.values() for k2, v2 in v.items()])
        nodes = s1.union(s2)
        return list(nodes)


def UndirectedGraph(graph_dict=None):
    """Build a Graph where every edge (including future ones) goes both ways."""
    return Graph(graph_dict=graph_dict, directed=False)


romania_map = UndirectedGraph(dict(
    Arad=dict(Zerind=75, Sibiu=140, Timisoara=118),
    Bucharest=dict(Urziceni=85, Pitesti=101, Giurgiu=90, Fagaras=211),
    Craiova=dict(Drobeta=120, Rimnicu=146, Pitesti=138),
    Drobeta=dict(Mehadia=75),
    Eforie=dict(Hirsova=86),
    Fagaras=dict(Sibiu=99),
    Hirsova=dict(Urziceni=98),
    Iasi=dict(Vaslui=92, Neamt=87),
    Lugoj=dict(Timisoara=111, Mehadia=70),
    Oradea=dict(Zerind=71, Sibiu=151),
    Pitesti=dict(Rimnicu=97),
    Rimnicu=dict(Sibiu=80),
    Urziceni=dict(Vaslui=142)))
romania_map.locations = dict(
    Arad=(91, 492), Bucharest=(400, 327), Craiova=(253, 288),
    Drobeta=(165, 299), Eforie=(562, 293), Fagaras=(305, 449),
    Giurgiu=(375, 270), Hirsova=(534, 350), Iasi=(473, 506),
    Lugoj=(165, 379), Mehadia=(168, 339), Neamt=(406, 537),
    Oradea=(131, 571), Pitesti=(320, 368), Rimnicu=(233, 410),
    Sibiu=(207, 457), Timisoara=(94, 410), Urziceni=(456, 350),
    Vaslui=(509, 444), Zerind=(108, 531))


# print(romania_map.nodes())

class SimpleProblemSolvingAgent:

    def probability(self, p):
        """Return true with probability p."""
        return p > random.uniform(0.0, 1.0)

    def h(self, p1, p2):
        return int(math.sqrt((p2[0] - p1[0]) ** 2 + (p2[1] - p1[1]) ** 2))

    def exp_schedule(self, k=20, lam=0.005, limit=100):
        """One possible schedule function for simulated annealing"""
        return lambda t: (k * np.exp(-lam * t) if t < limit else 0)

    def value(self, city, goal, locations):
        """Value function for a city: negative of heuristic distance to goal."""
        return -self.h(locations[city], locations[goal])

    def print_path_dict(self, path_dict, start, goal):
        current = goal
        path = [current]

        while current != start:
            current = path_dict[current]
            path.append(current)

        path.reverse()
        return path

    def greedy_best_first_search(self, start, goal, graph, locations):
        """Search the nodes with the lowest f scores first.
            You specify the function f(node) that you want to minimize; for example,
            if f is a heuristic estimate to the goal, then we have greedy best
            first search; if f is node.depth then we have breadth-first search.
            There is a subtlety: the line "f = memoize(f, 'f')" means that the f
            values will be cached on the nodes as they are computed. So after doing
            a best first search you can examine the f values of the path returned."""
        visited = set()
        queue = [(0, start)]
        path_dict = {start: None}
        distance = {start: 0}

        while queue:
            (priority, current) = heapq.heappop(queue)

            if current == goal:
                return self.print_path_dict(path_dict, start, goal), distance[goal]

            if current not in visited:
                visited.add(current)

                for neighbor, dist in graph.get(current, {}).items():
                    new_distance = distance[current] + dist

                    if neighbor not in visited and (neighbor not in distance or new_distance < distance[neighbor]):
                        path_dict[neighbor] = current
                        distance[neighbor] = new_distance
                        heuristic_value = self.h(locations[neighbor], locations[goal])
                        heapq.heappush(queue, (heuristic_value, neighbor))

        return None, 0

    def a_star(self, start, goal, graph, locations):
        visited = set()
        queue = [(0, start)]
        path_dict = {start: None}
        g_value = {start: 0}

        while queue:
            (priority, current) = heapq.heappop(queue)

            if current == goal:
                return self.print_path_dict(path_dict, start, goal), g_value[goal]

            if current not in visited:
                visited.add(current)

                for neighbor, dist in graph.get(current, {}).items():
                    tentative_g_value = g_value[current] + dist

                    if neighbor not in visited and (neighbor not in g_value or tentative_g_value < g_value[neighbor]):
                        path_dict[neighbor] = current
                        g_value[neighbor] = tentative_g_value
                        f_value = tentative_g_value + self.h(locations[neighbor], locations[goal])
                        heapq.heappush(queue, (f_value, neighbor))
        return None, 0

    def hill_climbing(self, start, goal, graph, locations):
        current = start
        path = [current]

        while current != goal:
            neighbors = graph.get(current, {})

            if not neighbors:
                return None, 0

            # Sort neighbors based on heuristic values
            sorted_neighbors = sorted(neighbors.keys(), key=lambda x: self.h(locations[x], locations[goal]))

            # If the best neighbor is not better than the current position
            if self.h(locations[sorted_neighbors[0]], locations[goal]) >= self.h(locations[current], locations[goal]):
                break

            current = sorted_neighbors[0]
            path.append(current)

        # Calculate the total distance of the path
        distance = sum(graph[path[i]][path[i + 1]] for i in range(len(path) - 1))
        return path, distance

    def simulated_annealing(self, start, goal, graph, locations, schedule=None):
        if schedule is None:
            schedule = self.exp_schedule()
        current = start
        path = [current]

        for t in range(sys.maxsize):
            T = schedule(t)
            if T == 0:
                # Compute the total distance of the path before returning
                total_distance = sum(graph[path[i]][path[i + 1]] for i in range(len(path) - 1))
                return path, total_distance

            neighbors = list(graph.get(current, {}).keys())
            if not neighbors:
                total_distance = sum(graph[path[i]][path[i + 1]] for i in range(len(path) - 1))
                return path, total_distance

            next_city = random.choice(neighbors)
            delta_e = self.value(next_city, goal, locations) - self.value(current, goal, locations)

            if delta_e > 0 or self.probability(np.exp(delta_e / T)):
                current = next_city
                path.append(current)

        total_distance = sum(graph[path[i]][path[i + 1]] for i in range(len(path) - 1))
        return path, total_distance

    def find_path(self, start_city, goal_city, graph, locations):
        print("Greedy Best First Search")
        path, total_distance = self.greedy_best_first_search(start_city, goal_city, graph, locations)

        if path:
            print(' -> '.join(path))
            print(f"Total cost: {total_distance}")
        else:
            print(f"No path found from {start_city} to {goal_city}.")

        print("A* Search")
        path, total_distance = self.a_star(start_city, goal_city, graph, locations)

        if path:
            print(' -> '.join(path))
            print(f"Total cost: {total_distance}")
        else:
            print(f"No path found from {start_city} to {goal_city}.")

        print("Hill Climbing")
        path, total_distance = self.hill_climbing(start_city, goal_city, graph, locations)

        if path:
            print(' -> '.join(path))
            print(f"Total cost: {total_distance}")
        else:
            print(f"No path found from {start_city} to {goal_city}.")

        print("Simulated Annealing")
        path, total_distance = self.simulated_annealing(start_city, goal_city, graph, locations, self.exp_schedule())

        if path:
            print(' -> '.join(path))
            print(f"Total cost: {total_distance}")
        else:
            print(f"No path found from {start_city} to {goal_city}.")
