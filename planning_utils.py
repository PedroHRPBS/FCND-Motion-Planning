from enum import Enum
from queue import PriorityQueue, Queue
import numpy as np
from bresenham import bresenham
from math import sqrt, inf


def create_grid(data, drone_altitude, safety_distance):
    
    north_min = np.floor(np.min(data[:, 0] - data[:, 3]))
    north_max = np.ceil(np.max(data[:, 0] + data[:, 3]))

    east_min = np.floor(np.min(data[:, 1] - data[:, 4]))
    east_max = np.ceil(np.max(data[:, 1] + data[:, 4]))

    
    north_size = int(np.ceil(north_max - north_min))
    east_size = int(np.ceil(east_max - east_min))

    grid = np.zeros((north_size, east_size))

    # Populate the grid with obstacles
    for i in range(data.shape[0]):
        north, east, alt, d_north, d_east, d_alt = data[i, :]
        if alt + d_alt + safety_distance > drone_altitude:
            obstacle = [
                int(np.clip(north - d_north - safety_distance - north_min, 0, north_size-1)),
                int(np.clip(north + d_north + safety_distance - north_min, 0, north_size-1)),
                int(np.clip(east - d_east - safety_distance - east_min, 0, east_size-1)),
                int(np.clip(east + d_east + safety_distance - east_min, 0, east_size-1)),
            ]
            grid[obstacle[0]:obstacle[1]+1, obstacle[2]:obstacle[3]+1] = 1

    return grid, int(north_min), int(east_min)

def create_grid_bfs(data, drone_altitude, safety_distance):
    north_min = np.floor(np.amin(data[:, 0] - data[:, 3]))
    north_max = np.ceil(np.amax(data[:, 0] + data[:, 3]))
    
    east_min = np.floor(np.amin(data[:, 1] - data[:, 4]))
    east_max = np.ceil(np.amax(data[:, 1] + data[:, 4]))

  
    north_size = int(np.ceil(north_max - north_min)) + 2
    east_size = int(np.ceil(east_max - east_min)) + 2
 
    grid = np.zeros((north_size, east_size))

    north_min_center = np.min(data[:, 0])
    east_min_center = np.min(data[:, 1])

    for i in range(north_size):
        grid[i,0] = 1
        grid[i,-1] = 1

    for i in range(east_size):
        grid[0,i] = 1
        grid[-1,i] = 1


    for i in range(data.shape[0]):
        north, east, alt, d_north, d_east, d_alt = data[i, :]
        if alt + d_alt + safety_distance > drone_altitude:
            obstacle = [
                int(np.clip(north - d_north - safety_distance - north_min, 0, north_size-1)),
                int(np.clip(north + d_north + safety_distance - north_min, 0, north_size-1)),
                int(np.clip(east - d_east - safety_distance - east_min, 0, east_size-1)),
                int(np.clip(east + d_east + safety_distance - east_min, 0, east_size-1)),
            ]
            grid[obstacle[0]:obstacle[1]+1, obstacle[2]:obstacle[3]+1] = 1
        

    return grid, int(north_min), int(east_min)

class Action(Enum):
    LEFT = (0, -1, 1)
    RIGHT = (0, 1, 1)
    UP = (-1, 0, 1)
    DOWN = (1, 0, 1)
    LEFTUP = (-1, -1, sqrt(2))
    LEFTDOWN = (1, -1, sqrt(2))
    RIGHTUP = (-1, 1, sqrt(2))
    RIGHTDOWN = (1, 1, sqrt(2))
    
    
    @property
    def cost(self):
        return self.value[2]
    
    @property
    def delta(self):
        return (self.value[0], self.value[1])
   
def valid_actions(grid, current_node):
    valid = [Action.UP, Action.LEFT, Action.RIGHT, Action.DOWN, Action.LEFTUP, Action.LEFTDOWN, Action.RIGHTUP, Action.RIGHTDOWN]
    n, m = grid.shape[0] - 1, grid.shape[1] - 1
    x, y = current_node
    
    if x - 1 < 0 or grid[x-1, y] == 1:
        valid.remove(Action.UP)
    if x + 1 > n or grid[x+1, y] == 1:
        valid.remove(Action.DOWN)
    if y - 1 < 0 or grid[x, y-1] == 1:
        valid.remove(Action.LEFT)
    if y + 1 > m or grid[x, y+1] == 1:
        valid.remove(Action.RIGHT)
    if (x - 1 < 0 or y - 1 < 0) or grid[x-1, y-1] == 1:
        valid.remove(Action.LEFTUP)
    if (x + 1 > n or y - 1 < 0) or grid[x+1, y-1] == 1:
        valid.remove(Action.LEFTDOWN)
    if (x - 1 < 0 or y + 1 > m) or grid[x-1, y+1] == 1:
        valid.remove(Action.RIGHTUP)
    if (x + 1 > n or y + 1 > m) or grid[x+1, y+1] == 1:
        valid.remove(Action.RIGHTDOWN)
        
    return valid

def valid_actions_bfs(grid, current_node):
    valid = [Action.UP, Action.LEFT, Action.RIGHT, Action.DOWN, Action.LEFTUP, Action.LEFTDOWN, Action.RIGHTUP, Action.RIGHTDOWN]
    n, m = grid.shape[0] - 1, grid.shape[1] - 1
    x, y = current_node
    
    if grid[x-1, y] == 1:
        valid.remove(Action.UP)
    if grid[x+1, y] == 1:
        valid.remove(Action.DOWN)
    if grid[x, y-1] == 1:
        valid.remove(Action.LEFT)
    if grid[x, y+1] == 1:
        valid.remove(Action.RIGHT)
    if grid[x-1, y-1] == 1:
        valid.remove(Action.LEFTUP)
    if grid[x+1, y-1] == 1:
        valid.remove(Action.LEFTDOWN)
    if grid[x-1, y+1] == 1:
        valid.remove(Action.RIGHTUP)
    if grid[x+1, y+1] == 1:
        valid.remove(Action.RIGHTDOWN)
        
    return valid

def a_star(grid, h, start, goal):

    path = []
    path_cost = 0
    queue = PriorityQueue()
    queue.put((0, start))
    visited = set(start)

    branch = {}
    found = False
    
    while not queue.empty():
        item = queue.get()
        current_node = item[1]
        if current_node == start:
            current_cost = 0.0
        else:              
            current_cost = branch[current_node][0]
            
        if current_node == goal:        
            print('Found a path.')
            found = True
            break
        else:
            for action in valid_actions(grid, current_node):
                da = action.delta
                next_node = (current_node[0] + da[0], current_node[1] + da[1])
                branch_cost = action.cost + current_cost
                queue_cost = branch_cost + h(next_node,goal)
                
                if next_node not in visited:                
                    visited.add(next_node)               
                    branch[next_node] = (branch_cost, current_node, action)
                    queue.put((queue_cost, next_node))    
                    
    if found:
        n = goal
        path_cost = branch[n][0]
        path.append(goal)
        while branch[n][1] != start:
            path.append(branch[n][1])
            n = branch[n][1]
        path.append(branch[n][1])
 
    else:
        print('**********************')
        print('Failed to find a path!')
        print('**********************')
        
    return path[::-1], path_cost

def prune_path_bres(path, grid):
    
    pruned_path = [p for p in path]
    i = 0
    
    while i < len(pruned_path) - 2:
        p1 = pruned_path[i]
        p2 = pruned_path[i+2]
        k = 0
        
        cells = list(bresenham(int(p1[0]), int(p1[1]), int(p2[0]), int(p2[1])))
        
        for c in cells:
            k += grid[c]
            
        if k > 0:
            i += 1
        else:
            pruned_path.remove(pruned_path[i+1])
            
    return pruned_path          
        
def heuristic(position, goal_position):
    return np.linalg.norm(np.array(position) - np.array(goal_position))

def find_start_goal(skel, start, goal):
    
    skel_cells = np.transpose(skel.nonzero())
    start_min_dist = np.linalg.norm(np.array(start) - np.array(skel_cells), axis=1).argmin()
    near_start = skel_cells[start_min_dist]
    goal_min_dist = np.linalg.norm(np.array(goal) - np.array(skel_cells), axis=1).argmin()
    near_goal = skel_cells[goal_min_dist]
    
    return near_start, near_goal

def a_star_bfs(grid, h, start, goal):

    path = []
    path_cost = 0
    queue = PriorityQueue()
    queue.put((0, start))
    visited = set(start)
    width = np.shape(grid)[1]

    branch = {}
    found = False
    
    while not queue.empty():
        item = queue.get()
        current_node = item[1]
        if current_node == start:
            current_cost = 0.0
        else:              
            current_cost = branch[current_node][0]
            
        if current_node == goal:        
            print('Found a path.')
            found = True
            break
        else:
            for action in valid_actions_bfs(grid, current_node):
                da = action.delta
                next_node = (current_node[0] + da[0], current_node[1] + da[1])
                branch_cost = action.cost + current_cost
                queue_cost = branch_cost + h[next_node[0] + next_node[1] * width]
                
                if next_node not in visited:                
                    visited.add(next_node)               
                    branch[next_node] = (branch_cost, current_node, action)
                    queue.put((queue_cost, next_node))    
                    
    if found:
        n = goal
        path_cost = branch[n][0]
        while branch[n][1] != start:
            path.append(branch[n][1])
            n = branch[n][1]
        path.append(branch[n][1])
 
    else:
        print('**********************')
        print('Failed to find a path!')
        print('**********************')
        
    return path[::-1], path_cost

def breadth_first(grid, start, goal):
    q = Queue()
    q.put((0, start))
    visited = set(start)
    branch = {}
    found = False
    width = np.shape(grid)[1]
    mapa = np.zeros(np.shape(grid)[0]-1 + (np.shape(grid)[1]-1) * width + 1)
    mapa[mapa > -1] = inf 

    while not q.empty(): 
        item = q.get()
        current_node = item[1]
        
        if current_node == start:
            current_cost = 0.0
        else:              
            current_cost = branch[current_node][0]
        
        if current_node == goal: 
            print('Found a path.')
            found = True
            break
        else:
            valid = valid_actions_bfs(grid, current_node)
            for action in valid:
                da = action.value
                next_node = (current_node[0] + da[0], current_node[1] + da[1])
                branch_cost = action.cost + current_cost
                mapa[current_node[0] + current_node[1] * width] = current_cost
                if next_node not in visited:
                    visited.add(next_node)
                    q.put((branch_cost, next_node))    
                    branch[next_node] = (branch_cost, current_node, action)
   
    return mapa
