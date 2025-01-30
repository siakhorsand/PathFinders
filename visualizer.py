import pygame
import numpy as np
import random
from solver import (
    parse_maze_to_graph,
    bfs,
    dfs,
    astar,
    bidirectional_search,
    simulated_annealing
)

# Constants
SCREEN_SIZE = 1000
GRID_SIZE = 50
BLOCK_SIZE = SCREEN_SIZE // GRID_SIZE
FPS = 60

# Color Palette
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

# Glow effect parameters
GLOW_RADIUS = 15
GLOW_ALPHA = 100

# Initialize Pygame
pygame.init()
screen = pygame.display.set_mode((SCREEN_SIZE, SCREEN_SIZE))
pygame.display.set_caption("Maze Solver")
clock = pygame.time.Clock()
font = pygame.font.Font("Minecraft.ttf", 24)  # Use a retro font
small_font = pygame.font.Font("Minecraft.ttf", 18)

def create_glow_surface(radius, color):
    """Creates a glow surface for lighting effects"""
    glow = pygame.Surface((radius*2, radius*2), pygame.SRCALPHA)
    for i in range(radius, 0, -1):
        alpha = int(GLOW_ALPHA * (i/radius))
        pygame.draw.circle(glow, (*color, alpha), (radius, radius), i)
    return glow

# Pre-create glow surfaces
start_glow = create_glow_surface(GLOW_RADIUS, START_COLOR[:3])
goal_glow = create_glow_surface(GLOW_RADIUS, GOAL_COLOR[:3])
path_glow = create_glow_surface(GLOW_RADIUS, PATH_COLOR[:3])

def draw_cyber_background():
    """Draws a cyberpunk-style background with grid lines"""
    screen.fill(DARK_GRADIENT)
    
    # Draw grid lines
    for i in range(0, SCREEN_SIZE, BLOCK_SIZE):
        alpha = 30 if i % (BLOCK_SIZE*5) == 0 else 10
        pygame.draw.line(screen, (255, 255, 255, alpha), (i, 0), (i, SCREEN_SIZE))
        pygame.draw.line(screen, (255, 255, 255, alpha), (0, i), (SCREEN_SIZE, i))
    
    # Add scanline effect
    scanline = pygame.Surface((SCREEN_SIZE, SCREEN_SIZE), pygame.SRCALPHA)
    for y in range(0, SCREEN_SIZE, 4):
        pygame.draw.line(scanline, (0, 0, 0, 30), (0, y), (SCREEN_SIZE, y))
    screen.blit(scanline, (0, 0))

def draw_maze(maze):
    """Draws the maze with cyberpunk aesthetic"""
    for y in range(GRID_SIZE):
        for x in range(GRID_SIZE):
            rect = pygame.Rect(x * BLOCK_SIZE, y * BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE)
            
            if maze[y][x] == 0:  # Open cell
                pygame.draw.rect(screen, MAZE_BG, rect, border_radius=3)
            else:  # Wall with neon effect
                # Wall base
                pygame.draw.rect(screen, WALL_COLOR, rect.inflate(-2, -2), border_radius=2)
                # Glow effect
                glow_rect = start_glow.get_rect(center=rect.center)
                screen.blit(start_glow if (x+y) % 2 == 0 else goal_glow, glow_rect)

    # Animated start and goal markers
    t = pygame.time.get_ticks() / 1000
    pulse = abs(3 * (t % 1 - 0.5))  # 0-1.5-0 pulse wave
    
    # Start marker
    start_rect = pygame.Rect(0, 0, BLOCK_SIZE, BLOCK_SIZE)
    pygame.draw.circle(screen, START_COLOR, start_rect.center, BLOCK_SIZE//3 + int(pulse * 2))
    screen.blit(start_glow, start_glow.get_rect(center=start_rect.center))
    
    # Goal marker
    goal_rect = pygame.Rect((GRID_SIZE-1)*BLOCK_SIZE, (GRID_SIZE-1)*BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE)
    pygame.draw.circle(screen, GOAL_COLOR, goal_rect.center, BLOCK_SIZE//3 + int(pulse * 2))
    screen.blit(goal_glow, goal_glow.get_rect(center=goal_rect.center))

def draw_path(path, progress=1.0):
    """Draws the solution path with cybernetic effects"""
    if not path:
        return
    
    full_length = len(path)
    draw_length = int(full_length * progress)
    
    for i, (row, col) in enumerate(path[:draw_length]):
        alpha = 150 + int(105 * (i / full_length))
        color = (*PATH_COLOR, alpha)
        
        rect = pygame.Rect(col * BLOCK_SIZE, row * BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE)
        # Path segment
        pygame.draw.rect(screen, color, rect.inflate(-4, -4), border_radius=2)
        # Glow effect
        screen.blit(path_glow, path_glow.get_rect(center=rect.center))
        
        # Moving tracer effect
        if i == draw_length - 1:
            tracer_size = BLOCK_SIZE // 2 + int(5 * abs((pygame.time.get_ticks() % 1000)/500 - 1))
            tracer_rect = pygame.Rect(0, 0, tracer_size, tracer_size)
            tracer_rect.center = rect.center
            pygame.draw.rect(screen, NEON_GLOW, tracer_rect, border_radius=2)

def draw_control_panel(algorithm, hover_buttons, dropdown_open, algorithms):
    """Draws the control panel with algorithm selection dropdown above"""
    panel_height = 150
    panel_rect = pygame.Rect(10, SCREEN_SIZE - panel_height - 10, 200, panel_height)
    
    # Main panel
    pygame.draw.rect(screen, UI_COLOR, panel_rect, border_radius=8)
    pygame.draw.rect(screen, UI_ACCENT, panel_rect, 2, border_radius=8)
    
    # Panel buttons
    buttons = [
        ("Choose Algorithm", (panel_rect.centerx, panel_rect.top + 30), 0),
        ("Solve", (panel_rect.centerx, panel_rect.top + 70), 1),
        ("New Maze", (panel_rect.centerx, panel_rect.top + 110), 2),
        ("Quit", (panel_rect.centerx, panel_rect.top + 150), 3)
    ]
    
    # Draw main buttons
    for label, pos, btn_id in buttons:
        btn_rect = pygame.Rect(pos[0] - 80, pos[1] - 15, 160, 30)
        color = (100, 150, 200) if btn_id in hover_buttons else UI_ACCENT
        pygame.draw.rect(screen, color, btn_rect, border_radius=5)
        text = small_font.render(label, True, TEXT_COLOR)
        screen.blit(text, (btn_rect.centerx - text.get_width()//2, btn_rect.centery - text.get_height()//2))

    # Draw algorithm dropdown if open
    if dropdown_open:
        dropdown_height = len(algorithms) * 30
        dropdown_rect = pygame.Rect(panel_rect.left + 10, 
                                  panel_rect.top - dropdown_height - 10, 
                                  180, 
                                  dropdown_height)
        

        pygame.draw.rect(screen, UI_COLOR, dropdown_rect, border_radius=5)
        pygame.draw.rect(screen, UI_ACCENT, dropdown_rect, 2, border_radius=5)
        

        for i, algo in enumerate(algorithms):
            option_rect = pygame.Rect(dropdown_rect.left + 5,
                                    dropdown_rect.top + 5 + i*30,
                                    dropdown_rect.width - 10,
                                    25)
            
            if option_rect.collidepoint(pygame.mouse.get_pos()):
                pygame.draw.rect(screen, (100, 150, 200), option_rect, border_radius=3)
            
            text = small_font.render(algo, True, NEON_GLOW)
            screen.blit(text, (option_rect.centerx - text.get_width()//2, 
                             option_rect.centery - text.get_height()//2))

def solve_maze(maze, algorithm):
    """Converts maze to graph and runs selected solver"""
    nodes_dict, start_node, goal_node = parse_maze_to_graph(maze)
    
    # Debug: 
    if not start_node or not goal_node:
        print("Error: Start or goal node not found!")
        return None

    solvers = {
        "bfs": bfs,
        "dfs": dfs,
        "astar": astar,
        "bidirectional": bidirectional_search,
        "simulated_annealing": lambda: simulated_annealing(start_node, goal_node, temp=1000, cooling_rate=0.99)
    }
    
    solver = solvers.get(algorithm)
    if not solver:
        print(f"Error: Unknown algorithm {algorithm}")
        return None
        
    path = solver()
    print(f"Solved with {algorithm}. Path found: {path is not None}")  # Debug print
    return path


if __name__ == "__main__":
    maze = np.random.choice([0, 1], size=(GRID_SIZE, GRID_SIZE), p=[0.7, 0.3])
    maze[0][0] = maze[GRID_SIZE-1][GRID_SIZE-1] = 0

    running = True
    maze_solved = False
    path = None
    animation_progress = 0.0
    hover_buttons = set()
    dropdown_open = False
    algorithms = ["bfs", "dfs", "astar", "bidirectional", "simulated_annealing"]
    selected_algorithm = "bfs"

    while running:
        dt = clock.tick(FPS) / 1000
        mouse_pos = pygame.mouse.get_pos()
        hover_buttons.clear()
        panel_rect = pygame.Rect(10, SCREEN_SIZE - 160, 200, 150)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

            if event.type == pygame.MOUSEBUTTONDOWN:
                if panel_rect.collidepoint(mouse_pos):
                    y_pos = mouse_pos[1] - panel_rect.top
                    
                    # Main buttons
                    if 15 <= y_pos <= 45:  # Choose Algorithm
                        dropdown_open = not dropdown_open
                    elif 55 <= y_pos <= 85:  # Solve
                        maze_solved = False  # Trigger new solve
                    elif 95 <= y_pos <= 125:  # New Maze
                        maze = np.random.choice([0, 1], size=(GRID_SIZE, GRID_SIZE), p=[0.7, 0.3])
                        maze[0][0] = maze[GRID_SIZE-1][GRID_SIZE-1] = 0
                        maze_solved = False
                        path = None
                        animation_progress = 0.0
                    elif 135 <= y_pos <= 165:  # Quit
                        running = False

                # Handle algorithm selection
                if dropdown_open:
                    dropdown_rect = pygame.Rect(15, SCREEN_SIZE - 160 - len(algorithms)*30 - 15, 
                                              180, len(algorithms)*30)
                    if dropdown_rect.collidepoint(mouse_pos):
                        rel_y = mouse_pos[1] - dropdown_rect.top
                        selected_index = rel_y // 30
                        if 0 <= selected_index < len(algorithms):
                            selected_algorithm = algorithms[selected_index]
                            dropdown_open = False
                            maze_solved = False
                            path = None
                            animation_progress = 0.0

        # Update button hover states
        if panel_rect.collidepoint(mouse_pos):
            y_pos = mouse_pos[1] - panel_rect.top
            if 15 <= y_pos <= 45:
                hover_buttons.add(0)
            elif 55 <= y_pos <= 85:
                hover_buttons.add(1)
            elif 95 <= y_pos <= 125:
                hover_buttons.add(2)
            elif 135 <= y_pos <= 165:
                hover_buttons.add(3)

        # Drawing
        draw_cyber_background()
        draw_maze(maze)
        
        if path and maze_solved:
            draw_path(path, animation_progress)

        draw_control_panel(selected_algorithm, hover_buttons, dropdown_open, algorithms)
        pygame.display.flip()

    pygame.quit()