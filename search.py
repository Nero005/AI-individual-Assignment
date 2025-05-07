import sys
import heapq
from collections import deque

DIRECTIONS = [
    (0, 1, 'DOWN'),
    (1, 0, 'RIGHT'),
    (-1, 0, 'LEFT'),
    (0, -1, 'UP')
]

def parse_input_file(filename):
    with open(filename, 'r') as f:
        lines = [line.strip() for line in f if line.strip()]

    declared_rows, declared_cols = eval(lines[0])
    start = eval(lines[1])
    goals = [eval(g.strip()) for g in lines[2].split('|')]
    walls = [eval(line) for line in lines[3:]]

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
        end, move = parent[end]
        path.append(move)
    return list(reversed(path))

def bfs(grid, start, goals, rows, cols):
    queue = deque([start])
    visited = set([start])
    visited_order = [start]
    parent = {start: None}
    nodes_explored = 0

    while queue:
        current = queue.popleft()
        nodes_explored += 1

        if current in goals:
            path = reconstruct_path(parent, current)
            return current, nodes_explored, path, visited_order

        x, y = current
        for dx, dy, move in DIRECTIONS:
            neighbor = (x + dx, y + dy)
            if 0 <= neighbor[0] < rows and 0 <= neighbor[1] < cols and neighbor not in visited and grid[neighbor[0]][neighbor[1]] != '#':
                queue.append(neighbor)
                visited.add(neighbor)
                visited_order.append(neighbor)
                parent[neighbor] = (current, move)

    return None, nodes_explored, None, visited_order

def dfs(grid, start, goals, rows, cols):
    stack = [start]
    visited = set()
    visited_order = []
    parent = {start: None}
    nodes_explored = 0

    while stack:
        current = stack.pop()
        if current in visited:
            continue

        visited.add(current)
        visited_order.append(current)
        nodes_explored += 1

        if current in goals:
            path = reconstruct_path(parent, current)
            return current, nodes_explored, path, visited_order

        x, y = current
        for dx, dy, move in reversed(DIRECTIONS):
            nx, ny = x + dx, y + dy
            if 0 <= ny < rows and 0 <= nx < cols and (nx, ny) not in visited and grid[ny][nx] != '#':
                stack.append((nx, ny))
                parent[(nx, ny)] = (current, move)

    return None, nodes_explored, None, visited_order


def heuristic(a, goals):
    return min(abs(a[0] - g[0]) + abs(a[1] - g[1]) for g in goals)

def gbfs(grid, start, goals, rows, cols):
    heap = [(heuristic(start, goals), start)]
    visited = set([start])
    visited_order = [start]
    parent = {start: None}
    nodes_explored = 0

    while heap:
        _, current = heapq.heappop(heap)
        nodes_explored += 1

        if current in goals:
            path = reconstruct_path(parent, current)
            return current, nodes_explored, path, visited_order

        x, y = current
        for dx, dy, move in DIRECTIONS:
            neighbor = (x + dx, y + dy)
            if 0 <= neighbor[0] < rows and 0 <= neighbor[1] < cols and neighbor not in visited and grid[neighbor[0]][neighbor[1]] != '#':
                heapq.heappush(heap, (heuristic(neighbor, goals), neighbor))
                visited.add(neighbor)
                visited_order.append(neighbor)
                parent[neighbor] = (current, move)

    return None, nodes_explored, None, visited_order

def asearch(grid, start, goals, rows, cols):
    heap = [(heuristic(start, goals), 0, start)]
    visited = {start: 0}
    visited_order = [start]
    parent = {start: None}
    nodes_explored = 0

    while heap:
        f, g, current = heapq.heappop(heap)
        nodes_explored += 1

        if current in goals:
            path = reconstruct_path(parent, current)
            return current, nodes_explored, path, visited_order

        x, y = current
        for dx, dy, move in DIRECTIONS:
            neighbor = (x + dx, y + dy)
            if 0 <= neighbor[0] < rows and 0 <= neighbor[1] < cols and grid[neighbor[0]][neighbor[1]] != '#':
                new_g = g + 1
                if neighbor not in visited or new_g < visited[neighbor]:
                    visited[neighbor] = new_g
                    visited_order.append(neighbor)
                    heapq.heappush(heap, (new_g + heuristic(neighbor, goals), new_g, neighbor))
                    parent[neighbor] = (current, move)

    return None, nodes_explored, None, visited_order

def BID_BFS(grid, start, goals, rows, cols):

    if not goals:
        return None, 0, None, []

    goal = next(iter(goals)) 

    forward_queue = deque([start])
    forward_visited = {start: None}
    forward_path = []

    backward_queue = deque([goal])
    backward_visited = {goal: None}
    backward_path = []

    visited_order = [start, goal]
    nodes_created = 2

    while forward_queue and backward_queue:
        current_forward = forward_queue.popleft()
        fx, fy = current_forward
        for dx, dy, move in DIRECTIONS:
            nx, ny = fx + dx, fy + dy
            neighbor = (nx, ny)
            if (0 <= nx < rows and 0 <= ny < cols and grid[nx][ny] != '#' and neighbor not in forward_visited):
                forward_queue.append(neighbor)
                forward_visited[neighbor] = (current_forward, move)
                visited_order.append(neighbor)
                nodes_created += 1
                if neighbor in backward_visited:
                    meet_point = neighbor
                    fpath = []
                    n = meet_point
                    while forward_visited[n] is not None:
                        n, m = forward_visited[n]
                        fpath.append(m)
                    fpath.reverse()
                    bpath = []
                    n = meet_point
                    while backward_visited[n] is not None:
                        n, m = backward_visited[n]
                        bpath.append(m)
                    direction_map = {'UP': 'DOWN', 'DOWN': 'UP', 'LEFT': 'RIGHT', 'RIGHT': 'LEFT'}
                    bpath = [direction_map[m] for m in bpath]
                    return meet_point, nodes_created, fpath + bpath, visited_order

        current_backward = backward_queue.popleft()
        bx, by = current_backward
        for dx, dy, move in DIRECTIONS:
            nx, ny = bx + dx, by + dy
            neighbor = (nx, ny)
            if (0 <= nx < rows and 0 <= ny < cols and grid[nx][ny] != '#' and neighbor not in backward_visited):
                backward_queue.append(neighbor)
                backward_visited[neighbor] = (current_backward, move)
                visited_order.append(neighbor)
                nodes_created += 1
                if neighbor in forward_visited:
                    meet_point = neighbor
                    fpath = []
                    n = meet_point
                    while forward_visited[n] is not None:
                        n, m = forward_visited[n]
                        fpath.append(m)
                    fpath.reverse()
                    bpath = []
                    n = meet_point
                    while backward_visited[n] is not None:
                        n, m = backward_visited[n]
                        bpath.append(m)
                    direction_map = {'UP': 'DOWN', 'DOWN': 'UP', 'LEFT': 'RIGHT', 'RIGHT': 'LEFT'}
                    bpath = [direction_map[m] for m in bpath]
                    return meet_point, nodes_created, fpath + bpath, visited_order

    return None, nodes_created, None, visited_order

def reconstruct_path_bidir(parent_start, parent_goal, meet_point):
    path_start = []
    node = meet_point
    while parent_start[node] is not None:
        node, move = parent_start[node]
        path_start.append(move)
    path_start.reverse()

    path_goal = []
    node = meet_point
    while parent_goal[node] is not None:
        node, move = parent_goal[node]
        direction_map = {'UP': 'DOWN', 'DOWN': 'UP', 'LEFT': 'RIGHT', 'RIGHT': 'LEFT'}
        path_goal.append(direction_map[move])

    return path_start + path_goal   

def BID_GBFS(grid, start, goals, rows, cols, epsilon=2):
    goal = list(goals)[0]
    open_start = [(0, 0, start)]
    open_goal = [(0, 0, goal)]
    g_start = {start: 0}
    g_goal = {goal: 0}
    parent_start = {start: None}
    parent_goal = {goal: None}
    visited_start = set()
    visited_goal = set()
    visited_order = []
    nodes_created = 2
    count = 0

    while open_start and open_goal:
        _, _, current_s = heapq.heappop(open_start)
        visited_start.add(current_s)
        visited_order.append(current_s)

        if current_s in visited_goal:
            path = reconstruct_path_bidir(parent_start, parent_goal, current_s)
            return current_s, nodes_created, path, visited_order

        x, y = current_s
        for dx, dy, move in DIRECTIONS:
            nx, ny = x + dx, y + dy
            neighbor = (nx, ny)
            if (0 <= nx < rows and 0 <= ny < cols and grid[nx][ny] != '#' and neighbor not in visited_start):
                tentative_g = g_start[current_s] + 1
                if neighbor not in g_start or tentative_g < g_start[neighbor]:
                    g_start[neighbor] = tentative_g
                    f = tentative_g + epsilon * heuristic(neighbor, [goal])
                    count += 1
                    heapq.heappush(open_start, (f, count, neighbor))
                    parent_start[neighbor] = (current_s, move)
                    nodes_created += 1

        _, _, current_g = heapq.heappop(open_goal)
        visited_goal.add(current_g)
        visited_order.append(current_g)

        if current_g in visited_start:
            path = reconstruct_path_bidir(parent_start, parent_goal, current_g)
            return current_g, nodes_created, path, visited_order

        x, y = current_g
        for dx, dy, move in DIRECTIONS:
            nx, ny = x + dx, y + dy
            neighbor = (nx, ny)
            if (0 <= nx < rows and 0 <= ny < cols and grid[nx][ny] != '#' and neighbor not in visited_goal):
                tentative_g = g_goal[current_g] + 1
                if neighbor not in g_goal or tentative_g < g_goal[neighbor]:
                    g_goal[neighbor] = tentative_g
                    f = tentative_g + epsilon * heuristic(neighbor, [start])
                    count += 1
                    heapq.heappush(open_goal, (f, count, neighbor))
                    parent_goal[neighbor] = (current_g, move)
                    nodes_created += 1

    return None, nodes_created, None, visited_order


if __name__ == '__main__':
    if len(sys.argv) < 3:
        print("Usage: python search.py <filename> <method>")
    else:
        filename = sys.argv[1]
        method = sys.argv[2].upper()

        rows, cols, start, goals, walls = parse_input_file(filename)
        grid = build_grid(rows, cols, start, goals, walls)

        if method == 'BFS':
            goal, nodes_created, path, visited_order = bfs(grid, start, set(goals), rows, cols)
        elif method == 'DFS':
            goal, nodes_created, path, visited_order = dfs(grid, start, set(goals), rows, cols)
        elif method == 'GBFS':
            goal, nodes_created, path, visited_order = gbfs(grid, start, set(goals), rows, cols)
        elif method == 'AS':
            goal, nodes_created, path, visited_order = asearch(grid, start, set(goals), rows, cols)
        elif method == 'BID_BFS':
            goal, nodes_created, path, visited_order = BID_BFS(grid, start, set(goals), rows, cols)
        elif method == 'BID_GBFS':
            goal, nodes_created, path, visited_order = BID_GBFS(grid, start, set(goals), rows, cols)
        else:
            print(f"Method {method} not implemented yet.")
            sys.exit()

        print(f"{filename} {method}")
        if goal:
            print(f"{goal} {nodes_created}")
            print(','.join(action.lower() for action in path))
            print("Visited nodes:")
            print("visited:", visited_order)
        else:
            print(f"No goal is reachable; {nodes_created}")