import pygame
import time
from typing import List, Dict, Optional

from Astar import Node, SokobanPuzzle, astar_search, bfs_search, h1, h2

class SokobanGUI:
    # Colors
    COLORS = {
        'WHITE': (255, 255, 255),
        'BLACK': (0, 0, 0),
        'GRAY': (128, 128, 128),
        'BROWN': (139, 69, 19),
        'GREEN': (34, 139, 34),
        'YELLOW': (255, 215, 0),
        'BLUE': (65, 105, 225),
        'BACKGROUND': (240, 240, 240)
    }
    
    def __init__(self, cell_size: int = 60):
        """Initialize the Sokoban GUI.
        
        Args:
            cell_size: Size of each cell in pixels
        """
        pygame.init()
        
        self.cell_size = cell_size
        self.margin = 2
        self.info_height = 150  # Height of information panel
        
        # Font initialization
        self.font = pygame.font.Font(None, 36)
        self.small_font = pygame.font.Font(None, 24)
        
        # Initialize step counter and current solution
        self.current_step = 0
        self.current_solution = None
        self.solution_path = None
        self.paused = False
        
    def init_display(self, board: List[List[str]]):
        """Initialize the display based on board size."""
        self.board_height = len(board) * self.cell_size
        self.board_width = len(board[0]) * self.cell_size
        
        # Set up the display
        self.screen = pygame.display.set_mode((
            self.board_width,
            self.board_height + self.info_height
        ))
        pygame.display.set_caption('Sokoban Puzzle Solver')
        
    def draw_cell(self, x: int, y: int, color: tuple):
        """Draw a colored cell at the specified position."""
        pygame.draw.rect(self.screen, color,
            (x * self.cell_size + self.margin,
             y * self.cell_size + self.margin,
             self.cell_size - 2 * self.margin,
             self.cell_size - 2 * self.margin))
    
    def draw_player(self, x: int, y: int):
        """Draw the player character."""
        center_x = x * self.cell_size + self.cell_size // 2
        center_y = y * self.cell_size + self.cell_size // 2
        radius = self.cell_size // 3
        pygame.draw.circle(self.screen, self.COLORS['BLUE'], (center_x, center_y), radius)
        
    def draw_box(self, x: int, y: int):
        """Draw a box."""
        pygame.draw.rect(self.screen, self.COLORS['BROWN'],
            (x * self.cell_size + self.margin * 3,
             y * self.cell_size + self.margin * 3,
             self.cell_size - 6 * self.margin,
             self.cell_size - 6 * self.margin))
    
    def draw_target(self, x: int, y: int):
        """Draw a target space."""
        center_x = x * self.cell_size + self.cell_size // 2
        center_y = y * self.cell_size + self.cell_size // 2
        radius = self.cell_size // 4
        pygame.draw.circle(self.screen, self.COLORS['GREEN'], (center_x, center_y), radius, 3)
    
    def draw_board(self, board: List[List[str]]):
        """Draw the complete board state."""
        self.screen.fill(self.COLORS['BACKGROUND'])
        
        # Draw board elements
        for y in range(len(board)):
            for x in range(len(board[0])):
                cell = board[y][x]
                
                # Draw base cell
                if cell == 'O':  # Wall
                    self.draw_cell(x, y, self.COLORS['GRAY'])
                else:  # Floor
                    self.draw_cell(x, y, self.COLORS['WHITE'])
                
                # Draw game elements
                if cell in ['S', '.', '*']:  # Target space
                    self.draw_target(x, y)
                if cell in ['B', '*']:  # Box
                    self.draw_box(x, y)
                if cell in ['R', '.']:  # Player
                    self.draw_player(x, y)
    
    def draw_info_panel(self, solution_info: Dict):
        """Draw the information panel below the board."""
        info_y = self.board_height
        
        # Background
        pygame.draw.rect(self.screen, self.COLORS['WHITE'],
            (0, info_y, self.board_width, self.info_height))
        
        # Draw solution information
        y_offset = info_y + 10
        line_height = 30
        
        for algorithm, info in solution_info.items():
            text = f"{algorithm}: {info['steps']} steps"
            text_surface = self.small_font.render(text, True, self.COLORS['BLACK'])
            self.screen.blit(text_surface, (10, y_offset))
            y_offset += line_height
        
        # Draw current step
        if self.solution_path:
            step_text = f"Step: {self.current_step + 1}/{len(self.solution_path)}"
            step_surface = self.small_font.render(step_text, True, self.COLORS['BLACK'])
            self.screen.blit(step_surface, (10, y_offset))
    
    def simulate_solution(self, initial_board: List[List[str]], solution_path: List[SokobanPuzzle],
                         solution_info: Dict, delay: float = 0.5):
        """Simulate the solution with animation."""
        self.init_display(initial_board)
        self.solution_path = solution_path
        self.current_step = 0
        
        running = True
        last_update = time.time()
        
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_SPACE:
                        self.paused = not self.paused
                    elif event.key == pygame.K_LEFT:
                        self.current_step = max(0, self.current_step - 1)
                    elif event.key == pygame.K_RIGHT:
                        self.current_step = min(len(self.solution_path) - 1, self.current_step + 1)
            
            # Update board state if not paused
            current_time = time.time()
            if not self.paused and current_time - last_update >= delay:
                if self.current_step < len(self.solution_path) - 1:
                    self.current_step += 1
                    last_update = current_time
            
            # Draw current state
            self.draw_board(self.solution_path[self.current_step].board)
            self.draw_info_panel(solution_info)
            pygame.display.flip()
        
        pygame.quit()

def visualize_solution(initial_board: List[List[str]], bfs_solution: Optional[Node] = None,
                      astar_h1_solution: Optional[Node] = None, astar_h2_solution: Optional[Node] = None):
    """Visualize the solution using the GUI."""
    # Create solution info dictionary
    solution_info = {}
    if bfs_solution:
        solution_info['BFS'] = {'steps': bfs_solution.g}
    if astar_h1_solution:
        solution_info['A* (h1)'] = {'steps': astar_h1_solution.g}
    if astar_h2_solution:
        solution_info['A* (h2)'] = {'steps': astar_h2_solution.g}
    
    # Use the solution with the fewest steps for visualization
    solutions = [(s, name) for s, name in [
        (bfs_solution, 'BFS'),
        (astar_h1_solution, 'A* (h1)'),
        (astar_h2_solution, 'A* (h2)')
    ] if s is not None]
    
    if not solutions:
        print("No solutions to visualize!")
        return
    
    best_solution, algorithm = min(solutions, key=lambda x: x[0].g)
    print(f"Visualizing {algorithm} solution with {best_solution.g} steps")
    
    # Create and run GUI
    gui = SokobanGUI()
    gui.simulate_solution(
        initial_board,
        best_solution.getPath(),
        solution_info
    )

# Example usage
if __name__ == "__main__":
    # Example board
    test_board = [
        ['O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O'],
        ['O', ' ', ' ', ' ', ' ', ' ', ' ', ' ', 'O'],
        ['O', ' ', ' ', ' ', ' ', ' ', ' ', ' ', 'O'],
        ['O', ' ', ' ', 'O', 'O', 'O', ' ', ' ', 'O'],
        ['O', ' ', ' ', ' ', ' ', 'O', '.', ' ', 'O'],
        ['O', ' ', ' ', ' ', ' ', ' ', 'O', ' ', 'O'],
        ['O', ' ', ' ', 'B', ' ', ' ', 'O', ' ', 'O'],
        ['O', ' ', ' ', ' ', ' ', ' ', 'O', ' ', 'O'],
        ['O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O']
    ]
    
    # Solve puzzle
    puzzle = SokobanPuzzle(test_board)
    bfs_solution = bfs_search(puzzle)
    astar_h1_solution = astar_search(puzzle, h1)
    astar_h2_solution = astar_search(puzzle, h2)
    
    # Visualize solution
    visualize_solution(test_board, bfs_solution, astar_h1_solution, astar_h2_solution)