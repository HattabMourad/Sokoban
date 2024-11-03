from typing import List, Tuple, Set, Optional
from collections import deque
import copy
import heapq


class SokobanPuzzle:
    def __init__(self, board: List[List[str]]):
        """Initialize the Sokoban puzzle state.
        
        Args:
            board: 2D list representing the game board
        """
        self.board = board
        self.rows = len(board)
        self.cols = len(board[0])
        self.player_pos = self._find_player()
        
    def _find_player(self) -> Tuple[int, int]:
        """Find the player position on the board."""
        for i in range(self.rows):
            for j in range(self.cols):
                if self.board[i][j] in ['R', '.']:
                    return (i, j)
        raise ValueError("No player found on board")
    
    def isGoal(self) -> bool:
        """Check if current state is a goal state (all boxes on target spaces)."""
        # Count boxes not on targets
        for i in range(self.rows):
            for j in range(self.cols):
                if self.board[i][j] == 'B':  # Box not on target
                    return False
        return True
    
    def _is_valid_move(self, pos: Tuple[int, int]) -> bool:
        """Check if a position is valid (within bounds and not a wall)."""
        row, col = pos
        return (0 <= row < self.rows and 
                0 <= col < self.cols and 
                self.board[row][col] != 'O')
    
    def _get_cell(self, pos: Tuple[int, int]) -> str:
        """Get the content of a cell at given position."""
        return self.board[pos[0]][pos[1]]
    
    def get_boxes_and_targets(self) -> Tuple[List[Tuple[int, int]], List[Tuple[int, int]]]:
        """Get positions of all boxes and targets."""
        boxes = []
        targets = []
        for i in range(self.rows):
            for j in range(self.cols):
                if self.board[i][j] in ['B', '*']:
                    boxes.append((i, j))
                if self.board[i][j] in ['S', '.', '*']:
                    targets.append((i, j))
        return boxes, targets
    
    def successorFunction(self) -> List[Tuple[str, 'SokobanPuzzle']]:
        """Generate all possible successor states and their corresponding actions."""
        moves = {
            'UP': (-1, 0),
            'DOWN': (1, 0),
            'LEFT': (0, -1),
            'RIGHT': (0, 1)
        }
        
        successors = []
        
        for action, (dx, dy) in moves.items():
            # Calculate new position
            new_pos = (self.player_pos[0] + dx, self.player_pos[1] + dy)
            
            # Check if move is valid
            if not self._is_valid_move(new_pos):
                continue
                
            # Create new state
            new_state = copy.deepcopy(self)
            
            # Get current cell content
            curr_cell = self._get_cell(self.player_pos)
            new_cell = self._get_cell(new_pos)
            
            # Handle different cases
            if new_cell in [' ', 'S']:  # Moving to empty space or target
                # Update player position
                new_state.board[self.player_pos[0]][self.player_pos[1]] = 'S' if curr_cell == '.' else ' '
                new_state.board[new_pos[0]][new_pos[1]] = '.' if new_cell == 'S' else 'R'
                new_state.player_pos = new_pos
                successors.append((action, new_state))
                
            elif new_cell in ['B', '*']:  # Pushing a box
                # Calculate position after box
                box_new_pos = (new_pos[0] + dx, new_pos[1] + dy)
                
                if self._is_valid_move(box_new_pos):
                    box_new_cell = self._get_cell(box_new_pos)
                    
                    if box_new_cell in [' ', 'S']:  # Can push box
                        # Update box position
                        new_state.board[box_new_pos[0]][box_new_pos[1]] = '*' if box_new_cell == 'S' else 'B'
                        # Update player position
                        new_state.board[self.player_pos[0]][self.player_pos[1]] = 'S' if curr_cell == '.' else ' '
                        new_state.board[new_pos[0]][new_pos[1]] = '.' if new_cell == '*' else 'R'
                        new_state.player_pos = new_pos
                        successors.append((action, new_state))
        
        return successors

class Node:
    def __init__(self, state: SokobanPuzzle, parent: 'Node' = None, action: str = None):
        """Initialize a search node."""
        self.state = state
        self.parent = parent
        self.action = action
        self.g = 0 if parent is None else parent.g + 1
        self.f = 0
        
    def __lt__(self, other: 'Node') -> bool:
        """Comparison method for priority queue."""
        return self.f < other.f
    
    def getPath(self) -> List[SokobanPuzzle]:
        """Return the path from root to this node."""
        path = []
        current = self
        while current:
            path.append(current.state)
            current = current.parent
        return list(reversed(path))
    
    def getSolution(self) -> List[str]:
        """Return the sequence of actions from root to this node."""
        actions = []
        current = self
        while current.parent:
            actions.append(current.action)
            current = current.parent
        return list(reversed(actions))
    
    def setF(self, h_value: int):
        """Set the f-value for A* search."""
        self.f = self.g + h_value

def manhattan_distance(pos1: Tuple[int, int], pos2: Tuple[int, int]) -> int:
    """Calculate Manhattan distance between two positions."""
    return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])

def h1(state: SokobanPuzzle) -> int:
    """Heuristic 1: Number of boxes not on target spaces."""
    count = 0
    for i in range(state.rows):
        for j in range(state.cols):
            if state.board[i][j] == 'B':  # Box not on target
                count += 1
    return count

def h2(state: SokobanPuzzle) -> int:
    """Heuristic 2: Combination of boxes not on target and Manhattan distances."""
    boxes, targets = state.get_boxes_and_targets()
    
    # Count boxes not on targets
    not_on_target = sum(1 for i, j in boxes if state.board[i][j] == 'B')
    
    # Calculate minimum Manhattan distances
    total_distance = 0
    for box in boxes:
        if state.board[box[0]][box[1]] == 'B':  # Only consider boxes not on targets
            min_distance = min(manhattan_distance(box, target) for target in targets)
            total_distance += min_distance
    
    return 2 * not_on_target + total_distance

def h3 (state: SokobanPuzzle) -> int:
    unplaced_boxes = h1(state)
    total_manhattan_distance = h2(state)
    
    # Combine the two heuristics, giving more weight to the unplaced boxes
    return 3 * unplaced_boxes + total_manhattan_distance

def bfs_search(s: SokobanPuzzle):
    Open = deque()
    Closed = set()  # Use a set for faster lookups
    node_count = 0  # Initialize the node counter

    init_node = Node(state=s, parent=None, action=None)

    # Check if the initial state is the goal
    if s.isGoal():
        return init_node, node_count
    
    Open.append(init_node)
    Closed.add(tuple(tuple(row) for row in s.board))  # Convert board to a hashable form

    while Open:
        current = Open.popleft()
        node_count += 1  # Increment node counter

        # Generate successors (valid moves from current state)
        for action, successor in current.state.successorFunction():
            child = Node(state=successor, parent=current, action=action)

            # Convert child state to a hashable form for comparison
            child_state_tuple = tuple(tuple(row) for row in child.state.board)

            # Check if child state is not in Closed
            if child_state_tuple not in Closed:
                if child.state.isGoal():
                    return child, node_count
                Open.append(child)
                Closed.add(child_state_tuple)  # Mark this state as visited

    return None, node_count




def get_solution_path(goal_node):
    actions = []
    current = goal_node
    while current.parent is not None:
        actions.append(current.action)
        current = current.parent
    actions.reverse()
    return actions

import heapq
from typing import Optional

def astar_search(s: SokobanPuzzle, h) -> Tuple[Optional[Node], int]:
    Open = []  # Priority queue (min-heap)
    Open_dict = {}  # Dictionary to track the best nodes in Open
    Closed = set()  # Set for visited states
    node_count = 0  # Initialize the node counter
    
    # Initialize the root node
    init_node = Node(state=s, parent=None, action=None)
    init_node.g = 0
    init_node.f = h(init_node.state)
    
    # Insert the initial node into Open with its f-value
    heapq.heappush(Open, (init_node.f, init_node))
    Open_dict[tuple(tuple(row) for row in s.board)] = init_node  # Track the initial state
    
    while Open:
        # Get the node with the lowest f value
        _, current = heapq.heappop(Open)
        node_count += 1  # Increment node counter
        current_state_tuple = tuple(tuple(row) for row in current.state.board)
        
        # Check if the current node is the goal
        if current.state.isGoal():
            return current, node_count
        
        # Mark this state as explored
        Closed.add(current_state_tuple)
        
        # Remove from Open_dict if it exists
        if current_state_tuple in Open_dict:
            del Open_dict[current_state_tuple]  # Remove from Open_dict
            
        # Generate successors
        for action, successor in current.state.successorFunction():
            child = Node(state=successor, parent=current, action=action)
            child.g = current.g + 1  # Assume each move has a cost of 1
            child.f = child.g + h(child.state)
            
            child_state_tuple = tuple(tuple(row) for row in child.state.board)
            
            # If the child state has not been visited
            if child_state_tuple not in Closed:
                if child_state_tuple not in Open_dict or child.f < Open_dict[child_state_tuple].f:
                    # Add to Open
                    heapq.heappush(Open, (child.f, child))
                    Open_dict[child_state_tuple] = child  # Add or update the best node in Open

    return None, node_count
def is_deadlock(state: SokobanPuzzle) -> bool:
    """
    Detects if there are any boxes in deadlock positions (corner or line deadlocks).
    Returns True if a deadlock is found, False otherwise.
    """
    boxes, targets = state.get_boxes_and_targets()
    deadlock_positions = set()

    def is_corner_deadlock(row: int, col: int) -> bool:
        """Check if position forms a corner deadlock."""
        if (row, col) in targets:  # Position is a target, not a deadlock
            return False
            
        # Check adjacent walls in all four corner configurations
        corners = [
            ((row-1, col), (row, col-1)),  # top-left
            ((row-1, col), (row, col+1)),  # top-right
            ((row+1, col), (row, col-1)),  # bottom-left
            ((row+1, col), (row, col+1))   # bottom-right
        ]
        
        for wall1, wall2 in corners:
            if (0 <= wall1[0] < state.rows and 
                0 <= wall1[1] < state.cols and 
                0 <= wall2[0] < state.rows and 
                0 <= wall2[1] < state.cols):
                if (state.board[wall1[0]][wall1[1]] == 'O' and 
                    state.board[wall2[0]][wall2[1]] == 'O'):
                    return True
        return False

    def find_line_deadlocks() -> Set[Tuple[int, int]]:
        """Find all line deadlock positions."""
        line_deadlocks = set()
        
        # Find horizontal line deadlocks
        for row in range(1, state.rows - 1):
            wall_sequence = False
            start_col = -1
            
            for col in range(state.cols):
                if state.board[row][col] == 'O':
                    if not wall_sequence:
                        wall_sequence = True
                        start_col = col
                elif wall_sequence:
                    # Check if this section forms a line deadlock
                    end_col = col - 1
                    if (all(state.board[row-1][c] == 'O' for c in range(start_col+1, end_col+1)) or
                        all(state.board[row+1][c] == 'O' for c in range(start_col+1, end_col+1))):
                        for c in range(start_col+1, end_col+1):
                            if (row, c) not in targets:
                                line_deadlocks.add((row, c))
                    wall_sequence = False

        # Find vertical line deadlocks
        for col in range(1, state.cols - 1):
            wall_sequence = False
            start_row = -1
            
            for row in range(state.rows):
                if state.board[row][col] == 'O':
                    if not wall_sequence:
                        wall_sequence = True
                        start_row = row
                elif wall_sequence:
                    # Check if this section forms a line deadlock
                    end_row = row - 1
                    if (all(state.board[r][col-1] == 'O' for r in range(start_row+1, end_row+1)) or
                        all(state.board[r][col+1] == 'O' for r in range(start_row+1, end_row+1))):
                        for r in range(start_row+1, end_row+1):
                            if (r, col) not in targets:
                                line_deadlocks.add((r, col))
                    wall_sequence = False
                    
        return line_deadlocks

    # Find all deadlock positions
    for row in range(state.rows):
        for col in range(state.cols):
            if is_corner_deadlock(row, col):
                deadlock_positions.add((row, col))
    
    deadlock_positions.update(find_line_deadlocks())
    
    # Check if any box is in a deadlock position
    for box in boxes:
        if box in deadlock_positions and state.board[box[0]][box[1]] in ['B', '*']:
            return True
            
    return False

def astar_search_with_deadlock(s: SokobanPuzzle, h) -> Tuple[Optional[Node], int]:
    """A* search algorithm with deadlock detection."""
    Open = []  # Priority queue (min-heap)
    Open_dict = {}  # Dictionary to track nodes in Open
    Closed = set()  # Set of explored states
    node_count = 0
    
    # Initialize root node
    init_node = Node(state=s, parent=None, action=None)
    init_node.setF(h(init_node.state))
    
    # Check initial state for deadlock
    if is_deadlock(init_node.state):
        return None, 0
    
    heapq.heappush(Open, (init_node.f, init_node))
    Open_dict[tuple(tuple(row) for row in s.board)] = init_node
    
    while Open:
        _, current = heapq.heappop(Open)
        node_count += 1
        
        if current.state.isGoal():
            return current, node_count
            
        current_state_tuple = tuple(tuple(row) for row in current.state.board)
        Closed.add(current_state_tuple)
        
        if current_state_tuple in Open_dict:
            del Open_dict[current_state_tuple]
            
        for action, successor in current.state.successorFunction():
            # Skip if successor state has a deadlock
            if is_deadlock(successor):
                continue
                
            child = Node(state=successor, parent=current, action=action)
            child.setF(h(child.state))
            
            child_state_tuple = tuple(tuple(row) for row in child.state.board)
            
            if child_state_tuple not in Closed:
                if child_state_tuple not in Open_dict or child.f < Open_dict[child_state_tuple].f:
                    heapq.heappush(Open, (child.f, child))
                    Open_dict[child_state_tuple] = child
    
    return None, node_count


def solve_puzzle(board: List[List[str]]):
    """Solve puzzle using both BFS and A* with different heuristics."""
    #puzzle = SokobanPuzzle(board)

    puzzle = SokobanPuzzle(board)
    """ solution, nodes = astar_search_with_deadlock(puzzle, h3)

    if solution:
        print(f"Solution found in {solution.g} steps")
        print(f"Nodes expanded: {nodes}")
        print("Actions:", solution.getSolution())
    else:
        print(f"No solution found. Nodes expanded: {nodes}") """
    
    print("Solving with BFS...")
    bfs_solution, bfs_nodes_expanded = bfs_search(puzzle)
    if bfs_solution:
        print(f"BFS Solution found in {bfs_solution.g} steps with {bfs_nodes_expanded} nodes expanded")
        print("Actions:", bfs_solution.getSolution())
    else:
        print("No BFS solution found!")
        
    print("\nSolving with A* (h1)...")
    astar_h1_solution, astar_h1_nodes_expanded = astar_search(puzzle, h1)
    if astar_h1_solution:
        print(f"A* (h1) Solution found in {astar_h1_solution.g} steps with {astar_h1_nodes_expanded} nodes expanded")
        print("Actions:", astar_h1_solution.getSolution())
    else:
        print("No A* (h1) solution found!")
        
    print("\nSolving with A* (h2)...")
    astar_h2_solution, astar_h2_nodes_expanded = astar_search(puzzle, h2)
    if astar_h2_solution:
        print(f"A* (h2) Solution found in {astar_h2_solution.g} steps with {astar_h2_nodes_expanded} nodes expanded")
        print("Actions:", astar_h2_solution.getSolution())
    else:
        print("No A* (h2) solution found!")
    
    print("\nSolving with A* (h3)...")
    astar_h3_solution, astar_h3_nodes_expanded = astar_search(puzzle, h3)
    if astar_h3_solution:
        print(f"A* (h3) Solution found in {astar_h3_solution.g} steps with {astar_h3_nodes_expanded} nodes expanded")
        print("Actions:", astar_h3_solution.getSolution())
    else:
        print("No A* (h3) solution found!")
    solution, nodes = astar_search_with_deadlock(puzzle, h3)

    if solution:
        print(f"Solution found in {solution.g} steps")
        print(f"Nodes expanded: {nodes}")
        print("Actions:", solution.getSolution())
    else:
        print(f"No solution found. Nodes expanded: {nodes}")
    """ solution_node, nodes_expanded = astar_search_with_deadlock(puzzle, h3)
        
    if solution_node:
        print(f"Solution found for Test in {solution_node.g} steps with {nodes_expanded} nodes expanded")
        print("Solution Actions:", solution_node.getSolution())
    else:
        print(f"No solution found for Test. Nodes expanded: {nodes_expanded}") """

initial_board = [
    ['O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O'],
    ['O', 'O', 'O', ' ', ' ', 'O', 'O', 'O', 'O'],
    ['O', 'O', 'O', ' ', ' ', 'O', 'O', 'O', 'O'],
    ['O', 'O', 'O', ' ', '*', ' ', ' ', 'O', 'O'],
    ['O', 'O', 'O', 'B', 'O', 'B', ' ', 'O', 'O'],
    ['O', 'O', ' ', 'S', 'R', 'S', ' ', 'O', 'O'],
    ['O', 'O', ' ', ' ', ' ', ' ', 'O', 'O', 'O'],
    ['O', 'O', 'O', 'O', ' ', ' ', 'O', 'O', 'O'],
    ['O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O']
]
""" initial_board = [
    ['O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O'],
    ['O', ' ', ' ', ' ', ' ', ' ', ' ', ' ', 'O'],
    ['O', ' ', ' ', ' ', ' ', ' ', ' ', ' ', 'O'],
    ['O', ' ', ' ', 'O', 'O', 'O', ' ', ' ', 'O'],
    ['O', ' ', ' ', ' ', ' ', 'O', '.', ' ', 'O'],
    ['O', ' ', ' ', ' ', ' ', ' ', 'O', ' ', 'O'],
    ['O', ' ', ' ', 'B', ' ', ' ', 'O', ' ', 'O'],
    ['O', ' ', ' ', ' ', ' ', ' ', 'O', ' ', 'O'],
    ['O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O']
] """


# Solve using all methods
#solve_puzzle(initial_board)