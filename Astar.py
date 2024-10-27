from typing import List, Tuple, Set
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

def bfs_search(initial_state: SokobanPuzzle) -> Node:
    """Perform BFS search to find solution."""
    frontier = deque([Node(initial_state)])
    explored = set()
    
    while frontier:
        node = frontier.popleft()
        
        if node.state.isGoal():
            return node
            
        board_tuple = tuple(tuple(row) for row in node.state.board)
        if board_tuple in explored:
            continue
            
        explored.add(board_tuple)
        
        for action, successor_state in node.state.successorFunction():
            successor_node = Node(successor_state, node, action)
            frontier.append(successor_node)
    
    return None

def astar_search(initial_state: SokobanPuzzle, heuristic) -> Node:
    """Perform A* search to find solution.
    
    Args:
        initial_state: Initial puzzle state
        heuristic: Heuristic function to use (h1 or h2)
        
    Returns:
        Solution node if found, None otherwise
    """
    # Create initial node and set its f-value
    initial_node = Node(initial_state)
    initial_node.setF(heuristic(initial_state))
    
    # Initialize frontier as priority queue and explored set
    frontier = [initial_node]
    explored = set()
    
    while frontier:
        # Get node with lowest f-value
        node = heapq.heappop(frontier)
        
        if node.state.isGoal():
            return node
            
        board_tuple = tuple(tuple(row) for row in node.state.board)
        if board_tuple in explored:
            continue
            
        explored.add(board_tuple)
        
        # Generate and add successors to frontier
        for action, successor_state in node.state.successorFunction():
            successor_node = Node(successor_state, node, action)
            successor_node.setF(heuristic(successor_state))
            heapq.heappush(frontier, successor_node)
    
    return None

# Example usage
def solve_puzzle(board: List[List[str]]):
    """Solve puzzle using both BFS and A* with different heuristics."""
    puzzle = SokobanPuzzle(board)
    
    print("Solving with BFS...")
    bfs_solution = bfs_search(puzzle)
    if bfs_solution:
        print(f"BFS Solution found in {bfs_solution.g} steps")
        print("Actions:", bfs_solution.getSolution())
    else:
        print("No BFS solution found!")
        
    print("\nSolving with A* (h1)...")
    astar_h1_solution = astar_search(puzzle, h1)
    if astar_h1_solution:
        print(f"A* (h1) Solution found in {astar_h1_solution.g} steps")
        print("Actions:", astar_h1_solution.getSolution())
    else:
        print("No A* (h1) solution found!")
        
    print("\nSolving with A* (h2)...")
    astar_h2_solution = astar_search(puzzle, h2)
    if astar_h2_solution:
        print(f"A* (h2) Solution found in {astar_h2_solution.g} steps")
        print("Actions:", astar_h2_solution.getSolution())
    else:
        print("No A* (h2) solution found!")

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

# Solve using all methods
solve_puzzle(initial_board)