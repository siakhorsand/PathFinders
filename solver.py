# solver.py - Main algorithm implementation file

import math
import random
from collections import deque, defaultdict
import heapq
import numpy as np
from typing import Dict, List, Set, Tuple, Optional
import time

# Constants for algorithm limits and safety
MAX_PATH_LENGTH = 1000  # Maximum allowed path length
MAX_ITERATIONS = 10000  # Maximum iterations for iterative algorithms
TIMEOUT_SECONDS = 5     # Maximum execution time

class MazeSolverError(Exception):
    """Custom exception for maze solver specific errors"""
    pass

class Node:
    def __init__(self, value: Tuple[int, int]):
        self.value = value
        self.neighbors: Set['Node'] = set()

    def add_neighbor(self, node: 'Node') -> None:
        self.neighbors.add(node)
        node.neighbors.add(self)


    def __repr__(self) -> str:
        return f"Node({self.value})"
    
    def __lt__(self, other: 'Node') -> bool:
        return self.value < other.value

def parse_maze_to_graph(maze: np.ndarray) -> Tuple[Dict[Tuple[int, int], Node], Optional[Node], Optional[Node]]:
    """
    Converts a 2D maze into an undirected graph.
    Added validation and error handling.
    """
    if maze is None or maze.size == 0:
        raise MazeSolverError("Invalid maze: Empty or None")
        
    rows, cols = maze.shape
    nodes_dict: Dict[Tuple[int, int], Node] = {}
    
    # Create nodes for open cells
    for r in range(rows):
        for c in range(cols):
            if maze[r, c] == 0:  # Open cell
                nodes_dict[(r, c)] = Node((r, c))
    
    # Connect neighboring nodes
    for (r, c), node in nodes_dict.items():
        for dr, dc in [(0, 1), (1, 0), (0, -1), (-1, 0)]:  # 4-directional
            nr, nc = r + dr, c + dc
            if (nr, nc) in nodes_dict:
                node.add_neighbor(nodes_dict[(nr, nc)])
    
    start_node = nodes_dict.get((0, 0))
    goal_node = nodes_dict.get((rows-1, cols-1))
    
    # Validate start and goal nodes
    if not start_node or not goal_node:
        raise MazeSolverError("Start or goal position blocked")
        
    return nodes_dict, start_node, goal_node

def manhattan_distance(node_a: Node, node_b: Node) -> int:
    """Calculate Manhattan distance between two nodes"""
    r1, c1 = node_a.value
    r2, c2 = node_b.value
    return abs(r1 - r2) + abs(c1 - c2)

def reconstruct_path(end_node: Node, parent_map: Dict[Node, Node]) -> Optional[List[Tuple[int, int]]]:
    """
    Reconstructs path from parent map.
    Added validation and length checking.
    """
    path = []
    current = end_node
    
    while current and len(path) < MAX_PATH_LENGTH:
        path.append(current.value)
        current = parent_map.get(current)
        
    if len(path) >= MAX_PATH_LENGTH:
        raise MazeSolverError("Path exceeds maximum length")
        
    return path[::-1] if path else None

def bfs(start_node: Node, goal_node: Node) -> Optional[List[Tuple[int, int]]]:
    """
    Breadth-first search with timeout and validation.
    """
    if not start_node or not goal_node:
        raise MazeSolverError("Invalid start or goal node")
        
    start_time = time.time()
    queue = deque([(start_node, 0)])  # (node, depth)
    visited = {start_node}
    parent_map = {}
    
    while queue and len(visited) < MAX_ITERATIONS:
        if time.time() - start_time > TIMEOUT_SECONDS:
            raise MazeSolverError("BFS timeout")
            
        current, depth = queue.popleft()
        if current == goal_node:
            return reconstruct_path(goal_node, parent_map)
            
        for neighbor in current.neighbors:
            if neighbor not in visited:
                visited.add(neighbor)
                parent_map[neighbor] = current
                queue.append((neighbor, depth + 1))
                
    return None

def dfs(start_node: Node, goal_node: Node) -> Optional[List[Tuple[int, int]]]:
    """
    Depth-first search with depth limiting and timeout.
    """
    if not start_node or not goal_node:
        raise MazeSolverError("Invalid start or goal node")
        
    start_time = time.time()
    stack = [(start_node, 0)]  # (node, depth)
    visited = {start_node}
    parent_map = {}
    max_depth = MAX_PATH_LENGTH
    
    while stack:
        if time.time() - start_time > TIMEOUT_SECONDS:
            raise MazeSolverError("DFS timeout")
            
        current, depth = stack.pop()
        if current == goal_node:
            return reconstruct_path(goal_node, parent_map)
            
        if depth < max_depth:
            for neighbor in current.neighbors:
                if neighbor not in visited:
                    visited.add(neighbor)
                    parent_map[neighbor] = current
                    stack.append((neighbor, depth + 1))
                    
    return None

def astar(start_node: Node, goal_node: Node) -> Optional[List[Tuple[int, int]]]:
    """
    A* search with improved heuristic and timeout.
    """
    if not start_node or not goal_node:
        raise MazeSolverError("Invalid start or goal node")
        
    start_time = time.time()
    open_heap = [(0, start_node)]
    g_score = {start_node: 0}
    closed = set()
    parent_map = {}
    
    while open_heap and len(closed) < MAX_ITERATIONS:
        if time.time() - start_time > TIMEOUT_SECONDS:
            raise MazeSolverError("A* timeout")
            
        current_f, current = heapq.heappop(open_heap)
        
        if current == goal_node:
            return reconstruct_path(goal_node, parent_map)
            
        if current in closed:
            continue
            
        closed.add(current)
        
        for neighbor in current.neighbors:
            if neighbor in closed:
                continue
                
            tentative_g = g_score[current] + 1
            
            if neighbor not in g_score or tentative_g < g_score[neighbor]:
                g_score[neighbor] = tentative_g
                f_score = tentative_g + manhattan_distance(neighbor, goal_node)
                parent_map[neighbor] = current
                heapq.heappush(open_heap, (f_score, neighbor))
                
    return None

def bidirectional_search(start_node: Node, goal_node: Node) -> Optional[List[Tuple[int, int]]]:
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
    


def reconstruct_bidirectional_path(meeting_node: Node, 
                                 forward_visited: Dict[Node, Node],
                                 backward_visited: Dict[Node, Node]) -> List[Tuple[int, int]]:
    """Reconstructs the full path from start to goal through the meeting node."""

    forward_path = []
    current = meeting_node
    while current:
        forward_path.append(current.value)
        current = forward_visited.get(current)
    forward_path = forward_path[::-1] 

    backward_path = []
    current = backward_visited.get(meeting_node)
    while current:
        backward_path.append(current.value)
        current = backward_visited.get(current)
    # No reversal here since backward_visited stores parent references from goal
    
    # Combine paths (exclude meeting_node duplicate)
    return forward_path[::-1] + backward_path[1:]

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


def solve_maze(maze: np.ndarray, algorithm: str) -> Optional[List[Tuple[int, int]]]:
    try:
        nodes_dict, start_node, goal_node = parse_maze_to_graph(maze)
        
        solvers = {
            "bfs": lambda: bfs(start_node, goal_node),
            "dfs": lambda: dfs(start_node, goal_node),
            "astar": lambda: astar(start_node, goal_node),
            "bidirectional": lambda: bidirectional_search(start_node, goal_node),
            "simulated_annealing": lambda: simulated_annealing(
                start_node, goal_node, temperature=1000, cooling_rate=0.95
            )
        }
        
        solver = solvers.get(algorithm)
        if not solver:
            raise MazeSolverError(f"Unknown algorithm: {algorithm}")
            
        return solver()
        
    except MazeSolverError as e:
        print(f"Maze solving error: {e}")
        return None
    except Exception as e:
        print(f"Unexpected error: {e}")
        return None

# Test cases for the implementation
def run_tests():
    """
    Comprehensive test suite for the maze solver.
    """
    # Test case 1: Simple 3x3 maze
    simple_maze = np.array([
        [0, 0, 1],
        [0, 0, 0],
        [1, 0, 0]
    ])
    
    # Test case 2: No solution maze
    no_solution_maze = np.array([
        [0, 1, 0],
        [1, 1, 1],
        [0, 1, 0]
    ])
    
    # Test case 3: Large maze (20x20)
    large_maze = np.random.choice(
        [0, 1],
        size=(20, 20),
        p=[0.7, 0.3]  # 70% chance of open cells
    )
    large_maze[0, 0] = large_maze[-1, -1] = 0  # Ensure start/end are open
    
    # Run tests for each maze and algorithm
    mazes = [simple_maze, no_solution_maze, large_maze]
    algorithms = ["bfs", "dfs", "astar", "bidirectional", "simulated_annealing"]
    
    for i, maze in enumerate(mazes):
        print(f"\nTesting maze {i+1}:")
        for algo in algorithms:
            try:
                start_time = time.time()
                path = solve_maze(maze, algo)
                end_time = time.time()
                
                print(f"  {algo:20}: {'Success' if path else 'No path'} ({end_time - start_time:.3f}s)")

            except MazeSolverError as e:
                print(f"  {algo:20}: Error - {e}")
            except Exception as e:
                print(f"  {algo:20}: Unexpected error - {e}")

if __name__ == "__main__":
    run_tests()