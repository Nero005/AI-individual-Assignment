import sys
from collections import deque

def display_grid(rows, cols, markers=None, goals=None, obstacles=None, path=None):
    """Function to display the grid based on given rows and columns.
       Markers is a list of (col, row) coordinates to place an 'X'.
       Goals is a list of (col, row) coordinates to place a 'G'.
       Obstacles is a list of cells (col, row) to mark with '#'.
       Path is a list of cells to mark the path of the solution."""
    for row in range(rows):
        row_display = []
        for col in range(cols):
            if markers and (col, row) in markers:
                row_display.append('X')  # Mark the initial state with 'X'
            elif goals and (col, row) in goals:
                row_display.append('G')  # Mark the goal state with 'G'
            elif obstacles and (col, row) in obstacles:
                row_display.append('#')  # Mark the obstacle with '#'
            elif path and (col, row) in path:
                row_display.append('P')  # Mark the path with 'P'
            else:
                row_display.append('.')
        print(' '.join(row_display))

def parse_input(input_file):
    """Parse the input file and return grid dimensions, initial state, goals, and obstacles."""
    try:
        with open(input_file, 'r') as file:
            # First line: grid dimensions
            first_line = file.readline().strip()
            grid_line = first_line.split(']')[0] + ']'
            grid_dimensions = grid_line.strip('[]').split(',')
            rows, cols = int(grid_dimensions[0].strip()), int(grid_dimensions[1].strip())

            # Second line: initial state coordinates (column index, row index)
            second_line = file.readline().strip()
            coord_line = second_line.split(')')[0] + ')'  # Remove comments after the coordinates
            coordinates = coord_line.strip('()').split(',')
            initial_state = (int(coordinates[0].strip()), int(coordinates[1].strip()))

            # Third line: goal states coordinates
            third_line = file.readline().strip()
            # Remove comments after the goal coordinates
            goal_line = third_line.split('//')[0]
            goal_coordinates = goal_line.split('|')
            goals = []
            for goal in goal_coordinates:
                goal = goal.strip().strip('()')  # Remove parentheses
                goal_col, goal_row = map(int, goal.split(','))
                goals.append((goal_col, goal_row))

            # Fourth to Tenth lines: Obstacles
            obstacles = set()
            for _ in range(7):  # Read the next 7 lines for obstacle definitions
                obstacle_line = file.readline().strip()
                obstacle_data = obstacle_line.split('//')[0].strip()
                obstacle_data = obstacle_data.strip('()')
                obstacle_values = list(map(int, obstacle_data.split(',')))

                # Extract top-left corner and size
                top_left_col, top_left_row = obstacle_values[0], obstacle_values[1]
                width, height = obstacle_values[2], obstacle_values[3]

                # Mark all cells covered by this obstacle
                for i in range(width):
                    for j in range(height):
                        obstacles.add((top_left_col + i, top_left_row + j))

            return rows, cols, initial_state, goals, obstacles
    except FileNotFoundError:
        print(f"Error: The file '{input_file}' was not found.")
        sys.exit(1)
    except ValueError as e:
        print(f"Error: Unable to parse grid dimensions or coordinates. {e}")
        sys.exit(1)
    except Exception as e:
        print(f"An error occurred: {e}")
        sys.exit(1)

def bfs(start, goals, rows, cols, obstacles):
    """Perform Breadth-First Search (BFS) to find the shortest path from start to any of the goals."""
    # Define the possible movements: up, down, left, right
    movements = [(-1, 0), (1, 0), (0, -1), (0, 1)]

    # Queue for BFS: holds tuples of (current position, path to this position)
    queue = deque([(start, [start])])
    visited = set([start])

    while queue:
        current_position, path = queue.popleft()

        # Check if we have reached a goal
        if current_position in goals:
            return path  # Return the path to the goal

        # Explore neighbors
        for move in movements:
            neighbor = (current_position[0] + move[0], current_position[1] + move[1])

            # Check if the neighbor is within bounds, not an obstacle, and not visited
            if (0 <= neighbor[0] < cols and 0 <= neighbor[1] < rows and 
                neighbor not in obstacles and neighbor not in visited):
                visited.add(neighbor)
                queue.append((neighbor, path + [neighbor]))

    return None  # No path found

def main():
    if len(sys.argv) < 3:
        print("Usage: python script.py <input_file> <algorithm>")
        print("Example: python script.py input.txt BFS")
        sys.exit(1)

    input_file = sys.argv[1]
    algorithm = sys.argv[2]

    # Parse input file
    rows, cols, initial_state, goals, obstacles = parse_input(input_file)

    # Print grid information
    print(f"Grid Dimensions: {rows} rows, {cols} columns")
    print(f"Initial State: {initial_state}")
    print(f"Goal States: {goals}")
    print(f"Obstacles: {obstacles}")

    if algorithm.upper() == "BFS":
        print("\nPerforming Breadth-First Search (BFS)...")

        # Execute BFS to find the path
        path = bfs(initial_state, goals, rows, cols, obstacles)
        
        if path:
            print("Path found:", path)
            print("\nGenerated Grid with Path:")
            display_grid(rows, cols, markers=[initial_state], goals=goals, obstacles=obstacles, path=path)
        else:
            print("No path found using BFS.")

if __name__ == "__main__":
    main()
