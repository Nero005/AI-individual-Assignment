import sys
import re
from collections import deque

def parse_input(filename):
    with open(filename, 'r') as f:
        lines = [line.strip() for line in f.readlines() if line.strip()]

    grid_size = eval(lines[0])
    rows, cols = grid_size

    start = eval(lines[1])

    goals_line = lines[2]
    goals = [eval(g.strip()) for g in goals_line.split('|')]

    walls = []
    for line in lines[3:]:
        wall = eval(line)
        walls.append(wall)

    return rows, cols, start, goals, walls

def build_grid(rows, cols, start, goals, walls):
    grid = [[' ' for _ in range(cols)] for _ in range(rows) ]

    for (x, y, w, h) in walls:
        for dx in range(h):
            for dy in range(w):
                if 0 <= x + dx < rows and 0 <= y + dy < cols:
                    grid[x + dx][y + dy] = '#'
    
    for (x,y) in goals:
        grid[x][y] = 'G'

    sx, sy = start
    grid[sx][sy] = "S"

    return grid

def print_grid(grid):
    for now in grid:
        print(' '.join(row)) # type: ignore
    print()

