import pygame
import numpy as np
from typing import List, Set, Optional, Tuple, Dict
from enum import Enum
import math
import sys
from solver import (
    parse_maze_to_graph, bfs, dfs, astar,
    bidirectional_search, simulated_annealing, solve_maze, MazeSolverError
)

# Constants for visualization
SCREEN_SIZE = 1000
GRID_SIZE = 50
BLOCK_SIZE = SCREEN_SIZE // GRID_SIZE
FPS = 60

# Enhanced color palette with alpha support
class Colors:
    DARK_GRADIENT = (18, 23, 29)
    MAZE_BG = (34, 41, 49)
    WALL_COLOR = (255, 60, 60)  # Neon red
    PATH_COLOR = (100, 255, 220)
    START_COLOR = (100, 255, 150)
    GOAL_COLOR = (255, 100, 150)
    NEON_GLOW = (203, 249, 255, 50)
    UI_COLOR = (40, 45, 52)
    UI_ACCENT = (80, 90, 100)
    TEXT_COLOR = (200, 200, 200)
    ERROR_COLOR = (255, 80, 80)

class UIState:
    """Manages UI state and transitions"""
    def __init__(self):
        self.solving = False
        self.error_message: Optional[str] = None
        self.error_timer = 0
        self.animation_progress = 0.0
        self.current_path: Optional[List[Tuple[int, int]]] = None
        self.hover_buttons: Set[int] = set()
        self.dropdown_open = False
        self.selected_algorithm = "bfs"
        self.algorithms = ["bfs", "dfs", "astar", "bidirectional", "simulated_annealing", "dijkstra"]

    def show_error(self, message: str, duration: float = 3.0):
        """Display error message for specified duration"""
        self.error_message = message
        self.error_timer = duration
        self.solving = False

    def update(self, dt: float):
        """Update UI state with time delta"""
        if self.error_timer > 0:
            self.error_timer -= dt
            if self.error_timer <= 0:
                self.error_message = None

        if self.solving and self.current_path:
            self.animation_progress = min(1.0, self.animation_progress + dt)
            if self.animation_progress >= 1.0:
                self.solving = False

class MazeVisualizer:
    """Handles maze visualization and user interaction"""
    def __init__(self):
        pygame.init()
        self.screen = pygame.display.set_mode((SCREEN_SIZE, SCREEN_SIZE))
        pygame.display.set_caption("Enhanced Maze Solver")
        self.clock = pygame.time.Clock()
        
        # Load and scale fonts
        try:
            self.font = pygame.font.Font("Minecraft.ttf", 24)
            self.small_font = pygame.font.Font("Minecraft.ttf", 18)
        except pygame.error:
            # Fallback to system font if custom font not found
            self.font = pygame.font.SysFont("Arial", 24)
            self.small_font = pygame.font.SysFont("Arial", 18)

        self.ui_state = UIState()
        self.maze = self.generate_maze()
        self.glow_surfaces = self.create_glow_surfaces()

    def generate_maze(self) -> np.ndarray:
        """Generate a new random maze with guaranteed start/end access"""
        maze = np.random.choice([0, 1], size=(GRID_SIZE, GRID_SIZE), p=[0.7, 0.3])
        maze[0, 0] = maze[GRID_SIZE-1, GRID_SIZE-1] = 0
        # Ensure path exists between start and end
        self.ensure_path_exists(maze)
        return maze

    def ensure_path_exists(self, maze: np.ndarray):
        """Ensure there's at least one valid path through the maze"""
        if not solve_maze(maze, "bfs"):
            # Create a simple path if none exists
            for i in range(GRID_SIZE):
                maze[i, 0] = 0
            for j in range(GRID_SIZE):
                maze[GRID_SIZE-1, j] = 0

    def create_glow_surfaces(self) -> Dict[str, pygame.Surface]:
        """Create pre-rendered glow effects for better performance"""
        def create_glow(color: Tuple[int, ...], radius: int) -> pygame.Surface:
            surface = pygame.Surface((radius*2, radius*2), pygame.SRCALPHA)
            for i in range(radius, 0, -1):
                alpha = int(100 * (i/radius))
                pygame.draw.circle(surface, (*color[:3], alpha), (radius, radius), i)
            return surface

        return {
            'start': create_glow(Colors.START_COLOR, 15),
            'goal': create_glow(Colors.GOAL_COLOR, 15),
            'path': create_glow(Colors.PATH_COLOR, 15)
        }

    def draw_cyber_background(self):
        """Draw cyberpunk-style background with enhanced grid effects"""
        self.screen.fill(Colors.DARK_GRADIENT)
        
        # Draw perspective grid
        vanishing_point = (SCREEN_SIZE // 2, SCREEN_SIZE // 2)
        for i in range(0, SCREEN_SIZE, BLOCK_SIZE):
            alpha = 30 if i % (BLOCK_SIZE*5) == 0 else 10
            
            # Horizontal lines with perspective
            start_y = i
            pygame.draw.line(self.screen, (*Colors.UI_ACCENT[:3], alpha),
                           (0, start_y),
                           (SCREEN_SIZE, start_y + (start_y - vanishing_point[1])//4))
            
            # Vertical lines with perspective
            start_x = i
            pygame.draw.line(self.screen, (*Colors.UI_ACCENT[:3], alpha),
                           (start_x, 0),
                           (start_x + (start_x - vanishing_point[0])//4, SCREEN_SIZE))

    def draw_maze(self):
        """Draw maze with enhanced visual effects"""
        for y in range(GRID_SIZE):
            for x in range(GRID_SIZE):
                rect = pygame.Rect(x * BLOCK_SIZE, y * BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE)
                
                if self.maze[y, x] == 0:  # Open cell
                    pygame.draw.rect(self.screen, Colors.MAZE_BG, rect, border_radius=3)
                else:  # Wall with neon effect
                    pygame.draw.rect(self.screen, Colors.WALL_COLOR, 
                                   rect.inflate(-2, -2), border_radius=2)
                    glow = self.glow_surfaces['path' if (x+y) % 2 == 0 else 'start']
                    self.screen.blit(glow, glow.get_rect(center=rect.center))

        # Animated markers for start and goal
        t = pygame.time.get_ticks() / 1000
        pulse = abs(math.sin(t * 2)) * 3  # Smooth pulse effect
        
        for pos, glow in [((0, 0), 'start'), 
                         ((GRID_SIZE-1, GRID_SIZE-1), 'goal')]:
            rect = pygame.Rect(pos[1] * BLOCK_SIZE, pos[0] * BLOCK_SIZE, 
                             BLOCK_SIZE, BLOCK_SIZE)
            pygame.draw.circle(self.screen, 
                             Colors.START_COLOR if glow == 'start' else Colors.GOAL_COLOR,
                             rect.center, BLOCK_SIZE//3 + int(pulse))
            self.screen.blit(self.glow_surfaces[glow], 
                           self.glow_surfaces[glow].get_rect(center=rect.center))

    def draw_path(self):
        """Draw solution path with animated effects"""
        if not self.ui_state.current_path:
            return

        progress = self.ui_state.animation_progress
        path_length = len(self.ui_state.current_path)
        visible_length = int(path_length * progress)

        for i, (row, col) in enumerate(self.ui_state.current_path[:visible_length]):
            alpha = 150 + int(105 * (i / path_length))
            color = (*Colors.PATH_COLOR, alpha)
            
            rect = pygame.Rect(col * BLOCK_SIZE, row * BLOCK_SIZE, 
                             BLOCK_SIZE, BLOCK_SIZE)
            pygame.draw.rect(self.screen, color, rect.inflate(-4, -4), border_radius=2)
            self.screen.blit(self.glow_surfaces['path'], 
                           self.glow_surfaces['path'].get_rect(center=rect.center))

            # Animated tracer effect at path end
            if i == visible_length - 1:
                t = pygame.time.get_ticks() / 1000
                size = BLOCK_SIZE // 2 + int(5 * abs(math.sin(t * 4)))
                tracer = pygame.Rect(0, 0, size, size)
                tracer.center = rect.center
                pygame.draw.rect(self.screen, Colors.NEON_GLOW, tracer, border_radius=2)

    def draw_ui(self):
        """Draw enhanced UI with error handling and visual feedback"""
        panel_height = 150
        panel_rect = pygame.Rect(10, SCREEN_SIZE - panel_height - 10, 200, panel_height)

        # Main panel background with glow effect
        pygame.draw.rect(self.screen, Colors.UI_COLOR, panel_rect, border_radius=8)
        pygame.draw.rect(self.screen, Colors.UI_ACCENT, panel_rect, 2, border_radius=8)

        # Button definitions
        buttons = [
            ("Algorithm: " + self.ui_state.selected_algorithm, 
             (panel_rect.centerx, panel_rect.top + 30), 0),
            ("Solve", (panel_rect.centerx, panel_rect.top + 70), 1),
            ("New Maze", (panel_rect.centerx, panel_rect.top + 110), 2)
        ]

        # Draw buttons with hover effects
        for label, pos, btn_id in buttons:
            btn_rect = pygame.Rect(pos[0] - 80, pos[1] - 15, 160, 30)
            color = Colors.UI_ACCENT
            if btn_id in self.ui_state.hover_buttons:
                color = tuple(min(c + 40, 255) for c in color)
            pygame.draw.rect(self.screen, color, btn_rect, border_radius=5)
            text = self.small_font.render(label, True, Colors.TEXT_COLOR)
            self.screen.blit(text, text.get_rect(center=btn_rect.center))

        # Draw algorithm dropdown if open
        if self.ui_state.dropdown_open:
            self.draw_algorithm_dropdown(panel_rect)

        # Draw error message if present
        if self.ui_state.error_message:
            self.draw_error_message()

    def draw_algorithm_dropdown(self, panel_rect: pygame.Rect):
        """Draw dropdown menu for algorithm selection"""
        dropdown_height = len(self.ui_state.algorithms) * 30
        dropdown_rect = pygame.Rect(panel_rect.left + 10,
                                  panel_rect.top - dropdown_height - 10,
                                  180,
                                  dropdown_height)

        pygame.draw.rect(self.screen, Colors.UI_COLOR, dropdown_rect, border_radius=5)
        pygame.draw.rect(self.screen, Colors.UI_ACCENT, dropdown_rect, 2, border_radius=5)

        for i, algo in enumerate(self.ui_state.algorithms):
            option_rect = pygame.Rect(dropdown_rect.left + 5,
                                    dropdown_rect.top + 5 + i*30,
                                    dropdown_rect.width - 10,
                                    25)
            
            if option_rect.collidepoint(pygame.mouse.get_pos()):
                pygame.draw.rect(self.screen, (100, 150, 200), option_rect, border_radius=3)
            
            text = self.small_font.render(algo, True, Colors.TEXT_COLOR)
            self.screen.blit(text, text.get_rect(center=option_rect.center))

    def draw_error_message(self):
        """Draw error message with fade effect"""
        alpha = int(255 * min(1.0, self.ui_state.error_timer))
        text = self.font.render(self.ui_state.error_message, True, Colors.ERROR_COLOR)
        text.set_alpha(alpha)
        self.screen.blit(text, (SCREEN_SIZE//2 - text.get_width()//2, 20))

    def handle_events(self) -> bool:
        """Handle user input events, returns False if should quit"""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False
            
            if event.type == pygame.MOUSEBUTTONDOWN:
                self.handle_mouse_click(event.pos)
            
            if event.type == pygame.MOUSEMOTION:
                self.update_hover_states(event.pos)

        return True

    def handle_mouse_click(self, pos: Tuple[int, int]):
        """Handle mouse click events"""
        panel_rect = pygame.Rect(10, SCREEN_SIZE - 160, 200, 150)
        
        if panel_rect.collidepoint(pos):
            y_rel = pos[1] - panel_rect.top
            
            if 15 <= y_rel <= 45:  # Algorithm button
                self.ui_state.dropdown_open = not self.ui_state.dropdown_open
            elif 55 <= y_rel <= 85:  # Solve button
                self.solve_current_maze()
            elif 95 <= y_rel <= 125:  # New Maze button
                self.new_maze()
        
        # Handle algorithm selection from dropdown
        if self.ui_state.dropdown_open:
            dropdown_rect = pygame.Rect(15, 
                                      SCREEN_SIZE - 160 - len(self.ui_state.algorithms)*30 - 15,
                                      180, 
                                      len(self.ui_state.algorithms)*30)
            if dropdown_rect.collidepoint(pos):
                self.handle_algorithm_selection(pos[1] - dropdown_rect.top)

    def handle_algorithm_selection(self, y_offset: float):
        """Handle algorithm selection from dropdown"""
        selected_index = int(y_offset // 30)
        if 0 <= selected_index < len(self.ui_state.algorithms):
            self.ui_state.selected_algorithm = self.ui_state.algorithms[selected_index]
            self.ui_state.dropdown_open = False
            self.ui_state.current_path = None
            self.ui_state.animation_progress = 0.0

    def update_hover_states(self, pos: Tuple[int, int]):
        """Update button hover states based on mouse position"""
        panel_rect = pygame.Rect(10, SCREEN_SIZE - 160, 200, 150)
        self.ui_state.hover_buttons.clear()
        
        if panel_rect.collidepoint(pos):
            y_rel = pos[1] - panel_rect.top
            if 15 <= y_rel <= 45:
                self.ui_state.hover_buttons.add(0)
            elif 55 <= y_rel <= 85:
                self.ui_state.hover_buttons.add(1)
            elif 95 <= y_rel <= 125:
                self.ui_state.hover_buttons.add(2)

    def solve_current_maze(self):
        """Attempt to solve the current maze with selected algorithm"""
        if self.ui_state.solving:
            return

        try:
            path = solve_maze(self.maze, self.ui_state.selected_algorithm)
            if path:
                self.ui_state.current_path = path
                self.ui_state.animation_progress = 0.0
                self.ui_state.solving = True
            else:
                self.ui_state.show_error("No solution found!")
        except MazeSolverError as e:
            self.ui_state.show_error(str(e))
        except Exception as e:
            self.ui_state.show_error(f"Unexpected error: {str(e)}")

    def new_maze(self):
        """Generate a new maze and reset visualization state"""
        self.maze = self.generate_maze()
        self.ui_state.current_path = None
        self.ui_state.animation_progress = 0.0
        self.ui_state.solving = False
        self.ui_state.error_message = None

    def run(self):
        """Main program loop with proper error handling"""
        try:
            running = True
            while running:
                dt = self.clock.tick(FPS) / 1000.0  # Convert to seconds
                
                # Handle events and update state
                running = self.handle_events()
                self.ui_state.update(dt)
                
                # Draw everything
                self.draw_cyber_background()
                self.draw_maze()
                if self.ui_state.current_path:
                    self.draw_path()
                self.draw_ui()
                
                # Update display
                pygame.display.flip()

        except Exception as e:
            print(f"Critical error: {e}")
        finally:
            pygame.quit()

def main():
    """Program entry point with error handling"""
    try:
        visualizer = MazeVisualizer()
        visualizer.run()
    except Exception as e:
        print(f"Failed to start application: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()