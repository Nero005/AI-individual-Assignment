import sys
from collections import deque

DIRECTIONS = [
    (-1, 0, 'UP'),
    (0, -1, 'LEFT'),
    (1, 0, 'DOWN'),
    (0, 1, 'RIGHT')
]

def parse_input_file(filename):
    with open(filename, 'r') as f:
        lines = [line.strip() for line in f if line.strip()]

    grid_size = eval(lines[0])  # [N, M]
    rows, cols = grid_size

    start = eval(lines[1])  # (x, y)
    goals = [eval(g.strip()) for g in lines[2].split('|')]
    walls = [eval(line) for line in lines[3:]]

    return rows, cols, start, goals, walls

def build_grid(rows, cols, start, goals, walls):
    grid = [[' ' for _ in range(cols)] for _ in range(rows)]

    for (x, y, w, h) in walls:
        for dx in range(h):
            for dy in range(w):
                if 0 <= x + dx < rows and 0 <= y + dy < cols:
                    grid[x + dx][y + dy] = '#'

    for (x, y) in goals:
        grid[x][y] = 'G'

    sx, sy = start
    grid[sx][sy] = 'S'

    return grid

def reconstruct_path(parent, end):
    path = []
    while parent[end] is not None:
        end, move = parent[end]
        path.append(move)
    return list(reversed(path))

def bfs(grid, start, goals, rows, cols):
    queue = deque()
    visited = set()
    parent = {}

    queue.append(start)
    visited.add(start)
    parent[start] = None

    nodes_created = 1

    while queue:
        current = queue.popleft()

        if current in goals:
            path = reconstruct_path(parent, current)
            return current, nodes_created, path

        x, y = current
        for dx, dy, move in DIRECTIONS:
            nx, ny = x + dx, y + dy
            neighbor = (nx, ny)

            if (0 <= nx < rows and 0 <= ny < cols and
                neighbor not in visited and grid[nx][ny] != '#'):

                queue.append(neighbor)
                visited.add(neighbor)
                parent[neighbor] = (current, move)
                nodes_created += 1

    return None, nodes_created, None

def dfs(grid, start, goals, rows, cols):
    stack = []
    visited = set()
    parent = {}

    stack.append(start)
    visited.add(start)
    parent[start] = None

    nodes_created = 1

    while stack:
        current = stack.pop()

        if current in goals:
            path = reconstruct_path(parent, current)
            return current, nodes_created, path

        x, y = current
        for dx, dy, move in reversed(DIRECTIONS):
            nx, ny = x + dx, y + dy
            neighbor = (nx, ny)

            if (0 <= nx < rows and 0 <= ny < cols and
                neighbor not in visited and grid[nx][ny] != '#'):

                stack.append(neighbor)
                visited.add(neighbor)
                parent[neighbor] = (current, move)
                nodes_created += 1

    return None, nodes_created, None

if __name__ == '__main__':
    if len(sys.argv) < 3:
        print("Usage: python search.py <filename> <method>")
    else:
        filename = sys.argv[1]
        method = sys.argv[2].upper()

        rows, cols, start, goals, walls = parse_input_file(filename)
        grid = build_grid(rows, cols, start, goals, walls)

        if method == 'BFS':
            goal, nodes_created, path = bfs(grid, start, set(goals), rows, cols)
        elif method == 'DFS':
            goal, nodes_created, path = dfs(grid, start, set(goals), rows, cols)
        else:
            print(f"Method {method} not implemented yet.")
            sys.exit()

        print(f"{filename} {method}")
        if goal:
            print(f"{goal} {nodes_created}")
            print(','.join(path))
        else:
            print(f"No goal is reachable; {nodes_created}")

