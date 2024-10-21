class SokobanGame:
    def __init__(self, grid):
        self.grid = grid
        self.player_pos = self.find_player()
        self.box_pos = self.find_boxes()
        self.target_pos = self.find_targets()

    def find_player(self):
        for row in range(len(self.grid)):
            for col in range(len(self.grid[0])):
                if self.grid[row][col] == 'R':
                    return (row, col)
        return None

    def find_boxes(self):
        boxes = []
        for row in range(len(self.grid)):
            for col in range(len(self.grid[0])):
                if self.grid[row][col] == 'B':
                    boxes.append((row, col))
        return boxes

    def find_targets(self):
        targets = []
        for row in range(len(self.grid)):
            for col in range(len(self.grid[0])):
                if self.grid[row][col] == 'T':
                    targets.append((row, col))
        return targets

    def is_goal(self):
        return all(box in self.target_pos for box in self.box_pos)

    def is_valid_move(self, direction):
        row_change, col_change = direction
        new_player_pos = (self.player_pos[0] + row_change, self.player_pos[1] + col_change)

        if self.grid[new_player_pos[0]][new_player_pos[1]] == 'O':
            return False

        if new_player_pos in self.box_pos:
            new_box_pos = (new_player_pos[0] + row_change, new_player_pos[1] + col_change)
            if self.grid[new_box_pos[0]][new_box_pos[1]] == 'O' or new_box_pos in self.box_pos:
                return False

        return True

    def move(self, direction):
        if self.is_valid_move(direction):
            row_change, col_change = direction
            new_player_pos = (self.player_pos[0] + row_change, self.player_pos[1] + col_change)

            new_box_positions = self.box_pos.copy()

            if new_player_pos in self.box_pos:
                new_box_pos = (new_player_pos[0] + row_change, new_player_pos[1] + col_change)
                new_box_positions.remove(new_player_pos)
                new_box_positions.append(new_box_pos)

            new_state = SokobanGame(self.grid)
            new_state.player_pos = new_player_pos
            new_state.box_pos = new_box_positions
            return new_state
        
        return None
    
    def successor_function(self):
        directions = {
            'UP': (-1, 0),
            'DOWN': (1, 0),
            'LEFT': (0, -1),
            'RIGHT': (0, 1)
        }

        successors = []
        for action, direction in directions.items():
            new_state = self.move(direction)
            if new_state is not None:
                successors.append((action, new_state))

        return successors
    
class Node:
    def _init(self, state, parent=None, action=None,g=0):
        self.state = state
        self.parent = parent
        self.action = action
        self.g = g
        self.f = 0

    def get_path(self):
        path = []
        current_node = self
        while current_node is not None:
            path.append(current_node)
            current_node = current_node.parent
        path.reverse()
        return path
    
    def get_solution(self):
        solution = []
        current_node = self
        while current_node is not None:
            solution.append(current_node.action)
            current_node = current_node.parent
        solution.reverse()
        return solution
    
    def set_f(self, heuristic):
        self.f = self.g + heuristic(self.state)