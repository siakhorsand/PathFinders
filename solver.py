import math
import random
from collections import deque, defaultdict
import heapq
import numpy as np

random.seed(42)

###############################################################################
#                                Node Class                                   #
###############################################################################

class Node:
    """
    Represents a graph node with an undirected adjacency list.
    'value' can store (row, col), or any unique identifier.
    'neighbors' is a list of connected Node objects (undirected).
    """
    def __init__(self, value):
        self.value = value
        self.neighbors = []

    def add_neighbor(self, node):
        """
        Adds an undirected edge between self and node:
         - self includes node in self.neighbors
         - node includes self in node.neighbors (undirected)
        """
        # TODO: Implement adding a neighbor in an undirected manner
        if node not in self.neighbors: 
            self.neighbors.append(node)
            node.neighbors.append(self)
        if node in self.neighbors:
            node.neighbors.append(self)
    

    def __repr__(self):
        return f"Node({self.value})"
    
    def __lt__(self, other):
        return self.value < other.value


###############################################################################
#                   Maze -> Graph Conversion (Undirected)                     #
###############################################################################

def parse_maze_to_graph(maze):
    """
    Converts a 2D maze (numpy array) into an undirected graph of Node objects.
    maze[r][c] == 0 means open cell; 1 means wall/blocked.

    Returns:
        nodes_dict: dict[(r, c): Node] mapping each open cell to its Node
        start_node : Node corresponding to (0, 0), or None if blocked
        goal_node  : Node corresponding to (rows-1, cols-1), or None if blocked
    """
    rows, cols = maze.shape
    nodes_dict = {}
    
    # TODO: Implement the logic to build nodes and link neighbors
    
    for r in range(rows):
        for c in range(cols):
            if maze[r][c]== 0:
                nodes_dict[(r,c)] = Node((r,c))
            else:   
                continue
    for r in range(rows):
        for c in range(cols):
            if maze[r][c] == 0:
                if r-1 >= 0 and maze[r-1][c] == 0:
                    nodes_dict[(r,c)].add_neighbor(nodes_dict[(r-1,c)])
                if r+1 < rows and maze[r+1][c] == 0:
                    nodes_dict[(r,c)].add_neighbor(nodes_dict[(r+1,c)])
                if c-1 >= 0 and maze[r][c-1] == 0:
                    nodes_dict[(r,c)].add_neighbor(nodes_dict[(r,c-1)])
                if c+1 < cols and maze[r][c+1] == 0:
                    nodes_dict[(r,c)].add_neighbor(nodes_dict[(r,c+1)])
                       
    start_node = None
    goal_node = None


    # TODO: Assign start_node and goal_node if they exist in nodes_dict
    if (0,0) in nodes_dict:
        start_node = nodes_dict[(0,0)]
    if (rows-1,cols-1) in nodes_dict:
        goal_node = nodes_dict[(rows-1,cols-1)]
    
    return nodes_dict, start_node, goal_node


###############################################################################
#                         BFS (Graph-based)                                    #
###############################################################################

def bfs(start_node, goal_node):
    """
    Breadth-first search on an undirected graph of Node objects.
    Returns a list of (row, col) from start to goal, or None if no path.

    Steps (suggested):
      1. Use a queue (collections.deque) to hold nodes to explore.
      2. Track visited nodes so you don’t revisit.
      3. Also track parent_map to reconstruct the path once goal_node is reached.
    """
    # TODO: Implement BFS
    queue = deque()
    visited = set()
    parent_map = {}
    queue.append(start_node)
    visited.add(start_node)
    while queue:
        curr = queue.popleft()
        if curr == goal_node:
            return reconstruct_path(goal_node, parent_map)
        for neighbor in curr.neighbors:
            if neighbor not in visited:
                visited.add(neighbor)
                parent_map[neighbor] = curr
                queue.append(neighbor)
    return None


###############################################################################
#                          DFS (Graph-based)                                   #
###############################################################################

def dfs(start_node, goal_node):
    """
    Depth-first search on an undirected graph of Node objects.
    Returns a list of (row, col) from start to goal, or None if no path.

    Steps (suggested):
      1. Use a stack (Python list) to hold nodes to explore.
      2. Keep track of visited nodes to avoid cycles.
      3. Reconstruct path via parent_map if goal_node is found.
    """
    # TODO: Implement DFS
    
    stack = []
    visited = set()
    parent_map = {}
    stack.append(start_node)
    visited.add(start_node)
    while stack:
        curr = stack.pop()
        if curr == goal_node:
            return reconstruct_path(goal_node, parent_map)
        for neighbor in curr.neighbors:
            if neighbor not in visited:
                visited.add(neighbor)
                parent_map[neighbor] = curr
                stack.append(neighbor)
    return None


###############################################################################
#                    A* (Graph-based with Manhattan)                           #
###############################################################################

def astar(start_node, goal_node):
    """
    A* search on an undirected graph of Node objects.
    Uses manhattan_distance as the heuristic, assuming node.value = (row, col).
    Returns a path (list of (row, col)) or None if not found.

    Steps (suggested):
      1. Maintain a min-heap/priority queue (heapq) where each entry is (f_score, node).
      2. f_score[node] = g_score[node] + heuristic(node, goal_node).
      3. g_score[node] is the cost from start_node to node.
      4. Expand the node with the smallest f_score, update neighbors if a better path is found.
    """
    # TODO: Implement A*
    if start_node is None or goal_node is None:
        return None

    open_heap = []
    visited = set()
    parent_map = {}
    g_score = defaultdict(lambda: float('inf'))
    g_score[start_node] = 0

    f_score = manhattan_distance(start_node, goal_node)
    heapq.heappush(open_heap, (f_score, start_node))

    while open_heap:
        curr_f_score, curr = heapq.heappop(open_heap)

        if curr== goal_node:
            return reconstruct_path(goal_node, parent_map)
        if curr in visited:
            continue
        visited.add(curr)

        for neighbor in curr.neighbors:
            cost = g_score[curr] + 1
            if cost < g_score[neighbor]:
                g_score[neighbor] = cost
                f_score = cost + manhattan_distance(neighbor, goal_node)
                parent_map[neighbor] = curr
                heapq.heappush(open_heap, (f_score, neighbor))



    return None

def manhattan_distance(node_a, node_b):
    """
    Helper: Manhattan distance between node_a.value and node_b.value 
    if they are (row, col) pairs.
    """
    # TODO: Return |r1 - r2| + |c1 - c2|
    r1, c1 = node_a.value
    r2, c2 = node_b.value
    r = abs(r1 - r2)
    c = abs(c1 - c2)
    return r + c 


###############################################################################
#                 Bidirectional Search (Graph-based)                          #
###############################################################################

def bidirectional_search(start_node, goal_node):
    """
    Bidirectional search on an undirected graph of Node objects.
    Returns list of (row, col) from start to goal, or None if not found.

    Steps (suggested):
      1. Maintain two frontiers (queues), one from start_node, one from goal_node.
      2. Alternate expansions between these two queues.
      3. If the frontiers intersect, reconstruct the path by combining partial paths.
    """
    # TODO: Implement bidirectional search
    if start_node is None or goal_node is None:
        return None

 
    fq = deque([start_node])
    bq = deque([goal_node])


    fp = {start_node: None}
    bp = {goal_node: None}

    while fq and bq:
        curr_f_size = len(fq)
        for _ in range(curr_f_size):
            curr = fq.popleft()
            if curr in bp:
                forward = reconstruct_path(curr, fp)
                backward = reconstruct_path(curr, bp)
                return forward + backward[1:]

            for neighbor in curr.neighbors:
                if neighbor not in fp:
                    fp[neighbor] = curr
                    fq.append(neighbor)

        curr_b_size = len(bq)
        for _ in range(curr_b_size):
            curr = bq.popleft()
            if curr in fp:
                forward = reconstruct_path(curr, fp)
                backward = reconstruct_path(curr, bp)
                return forward + backward[1:]
            for neighbor in curr.neighbors:
                if neighbor not in bp:
                    bp[neighbor] = curr
                    bq.append(neighbor)
    return None
    


###############################################################################
#             Simulated Annealing (Graph-based)                               #
###############################################################################

def simulated_annealing(start_node, goal_node, temperature=1.0, cooling_rate=0.99, min_temperature=0.01):
    """
    A basic simulated annealing approach on an undirected graph of Node objects.
    - The 'cost' is the manhattan_distance to the goal.
    - We randomly choose a neighbor and possibly move there.
    Returns a list of (row, col) from start to goal (the path traveled), or None if not reached.

    Steps (suggested):
      1. Start with 'current' = start_node, compute cost = manhattan_distance(current, goal_node).
      2. Pick a random neighbor. Compute next_cost.
      3. If next_cost < current_cost, move. Otherwise, move with probability e^(-cost_diff / temperature).
      4. Decrease temperature each step by cooling_rate until below min_temperature or we reach goal_node.
    """
    # TODO: Implement simulated annealing
    
    if start_node is None or goal_node is None:
        return None
    
    curr_node = start_node
    curr_cost = manhattan_distance(curr_node, goal_node)
    path = [curr_node.value] 
    
    while temperature > min_temperature:
        if curr_node == goal_node:
            return path 

        if not curr_node.neighbors:
            break  
        next_node = random.choice(curr_node.neighbors)
        
        next_cost = manhattan_distance(next_node, goal_node)

        cost_diff = next_cost - curr_cost
        if cost_diff < 0 or random.random() < math.exp(-cost_diff / temperature):
            curr_node = next_node
            curr_cost = next_cost
            path.append(curr_node.value)
        temperature *= cooling_rate
    
    return None


###############################################################################
#                           Helper: Reconstruct Path                           #
###############################################################################

def reconstruct_path(end_node, parent_map):
    """
    Reconstructs a path by tracing parent_map up to None.
    Returns a list of node.value from the start to 'end_node'.

    'parent_map' is typically dict[Node, Node], where parent_map[node] = parent.

    Steps (suggested):
      1. Start with end_node, follow parent_map[node] until None.
      2. Collect node.value, reverse the list, return it.
    """
    # TODO: Implement path reconstruction
    path = []
    current = end_node
    while current:
        path.append(current.value)
        current = parent_map.get(current)   
    if path:
        return path[::-1]
    else:
        return None
        


###############################################################################
#                              Demo / Testing                                 #
###############################################################################
if __name__ == "__main__":

    random.seed(42)
    np.random.seed(42)

    # Example small maze: 0 => open, 1 => wall
    maze_data = np.array([
        [0, 0, 1],
        [0, 0, 0],
        [1, 0, 0]
    ])

    # Parse into an undirected graph
    nodes_dict, start_node, goal_node = parse_maze_to_graph(maze_data)
    print("Created graph with", len(nodes_dict), "nodes.")
    print("Start Node:", start_node)
    print("Goal Node :", goal_node)
 
    # Test BFS (will return None until implemented)
    path_bfs = bfs(start_node, goal_node)
    print("BFS Path:", path_bfs)

    # Similarly test DFS, A*, etc.
    path_dfs = dfs(start_node, goal_node)
    path_astar = astar(start_node, goal_node)
    path_bidirectional = bidirectional_search(start_node, goal_node)
    print("DFS Path:", path_dfs)
    print("A* Path:", path_astar)
    print("Bidirectional Path:", path_bidirectional)

###############################################################################