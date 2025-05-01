import sys
import heapq
from collections import deque

DIRECTIONS = [
    (1, 0, 'DOWN'),
    (0, 1, 'RIGHT'),
    (0, -1, 'LEFT'),
    (-1, 0, 'UP')
]

def parse_input_file(filename):
    with open(filename, 'r') as f:
        lines = [line.strip() for line in f if line.strip()]

    # Read declared grid size
    declared_rows, declared_cols = eval(lines[0])

    start = eval(lines[1])
    goals = [eval(g.strip()) for g in lines[2].split('|')]
    walls = [eval(line) for line in lines[3:]]

    # Calculate required grid dimensions
    max_row = max(
        start[0],
        *(g[0] for g in goals),
        *(x + h - 1 for x, y, w, h in walls)
    )
    max_col = max(
        start[1],
        *(g[1] for g in goals),
        *(y + w - 1 for x, y, w, h in walls)
    )

    rows = max(declared_rows, max_row + 1)
    cols = max(declared_cols, max_col + 1)

    return rows, cols, start, goals, walls

def build_grid(rows, cols, start, goals, walls):
    grid = [['' for _ in range(cols)] for _ in range(rows)]

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
        end, move = parent [end]
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

def heuristic(a, goals):
    return min(abs(a[0] - g[0]) + abs(a[1] - g[1]) for g in goals)

def gbfs(grid, start, goals, rows, cols):
    heap = []
    visited = set()
    parent = {}

    h = heuristic(start, goals)
    heapq.heappush(heap, (h, start))
    visited.add(start)
    parent[start] = None

    nodes_created = 1

    while heap:
        _, current = heapq.heappop(heap)

        if current in goals:
            path = reconstruct_path(parent, current)
            return current, nodes_created, path

        x, y = current
        for dx, dy, move in DIRECTIONS:
            nx, ny = x + dx, y + dy
            neighbor = (nx, ny)

            if (0 <= nx < rows and 0 <= ny < cols and
                neighbor not in visited and grid[nx][ny] != '#'):

                h = heuristic(neighbor, goals)
                heapq.heappush(heap, (h, neighbor))
                visited.add(neighbor)
                parent[neighbor] = (current, move)
                nodes_created += 1

    return None, nodes_created, None

def asearch(grid, start, goals, rows, cols):
    heap = []
    visited = {}
    parent = {}

    h = heuristic(start, goals)
    heapq.heappush(heap, (h, 0, start))
    visited[start] = 0
    parent[start] = None

    node_created = 1

    while heap:
        f, g, current = heapq.heappop(heap)

        if current in goals:
            path = reconstruct_path(parent, current)
            return current,  node_created, path

        x, y = current
        for dx, dy, move in DIRECTIONS:
            nx, ny = x + dx, y + dy
            neighbor = (nx, ny)

            if (0 <= nx < rows and 0 <= ny < cols and grid[nx][ny] != '#'):
                new_g = g + 1

                if neighbor not in visited or new_g < visited[neighbor]:
                    visited[neighbor] = new_g
                    h = heuristic(neighbor, goals)
                    f = new_g + h
                    heapq.heappush(heap, (f, new_g, neighbor))
                    parent[neighbor] = (current, move)
                    node_created += 1

    return None, node_created, None

def CUS1(grid, start, goals, rows, cols, depth_limit=10):
    stack = [(start, 0)]
    visited = set()
    parent = {start: None}
    nodes_created = 1

    while stack:
        current, depth = stack.pop()

        if current in goals:
            path = reconstruct_path(parent, current)
            return current, nodes_created, path

        if depth >= depth_limit:
            continue

        x, y = current
        for dx, dy, move in reversed(DIRECTIONS):
            nx, ny = x + dx, y + dy
            neighbor = (nx, ny)

            if (0 <= nx < rows and 0 <= ny < cols and
                neighbor not in visited and grid[nx][ny] != '#'):

                visited.add(neighbor)
                parent[neighbor] = (current, move)
                stack.append((neighbor, depth + 1))
                nodes_created += 1

    return None, nodes_created, None

def CUS2(grid, start, goals, rows, cols, epsilon=2):
    open_list = []
    heapq.heappush(open_list, (0, 0, start))
    visited = set()
    parent = {start: None}
    g_cost = {start: 0}
    nodes_created = 1
    count = 0

    while open_list:
        _, _, current = heapq.heappop(open_list)

        if current in goals:
            path = reconstruct_path(parent, current)
            return current, nodes_created, path

        visited.add(current)
        x, y = current

        for dx, dy, move in DIRECTIONS:
            nx, ny = x + dx, y + dy
            neighbor = (nx, ny)

            if (0 <= nx < rows and 0 <= ny < cols and
                grid[nx][ny] != '#' and neighbor not in visited):

                tentative_g = g_cost[current] + 1

                if neighbor not in g_cost or tentative_g < g_cost[neighbor]:
                    g_cost[neighbor] = tentative_g
                    f = tentative_g + epsilon * heuristic(neighbor, goals)
                    count += 1
                    heapq.heappush(open_list, (f, count, neighbor))
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
        elif method == 'GBFS':
            goal, nodes_created, path = gbfs(grid, start, set(goals), rows, cols)
        elif method == 'AS':
            goal, nodes_created, path = asearch(grid, start, set(goals), rows, cols)
        elif method == 'CUS1':
            goal, nodes_created, path = CUS1(grid, start, set(goals), rows, cols)
        elif method == 'CUS2':
            goal, nodes_created, path = CUS2(grid, start, set(goals), rows, cols)
        else:
            print(f"Method {method} not implemented yet.")
            sys.exit()

        print(f"{filename} {method}")
        if goal:
            print(f"{goal} {nodes_created}")
            print(','.join(action.lower()for action in path))
        else:
            print(f"No goal is reachable; {nodes_created}")