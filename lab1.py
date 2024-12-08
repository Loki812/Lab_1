from PIL import Image
import sys
from typing import List, Dict, Tuple
import math 

# ------------- Global Variables ------------

height = 500 # pixels
width = 395 # pixels

longitude = 10.29 # meters constant per pixel
latitude = 7.55 # meters constant per pixel

# determines the traveling penalty for colored pixels
rbg_weights: Dict[Tuple, float] = {
    (248, 148, 18): 0.0, # open land value
    (255, 192, 0) : 50.0, # rough meadow value
    (255, 255, 255) : 20.0, # easy movement forest
    (2, 208, 60) : 30.0, # slow run forest value
    (2, 136, 40) : 50.0, # walk forest value
    (5, 73, 24) : 500.0, # impassible vegetation value
    (0, 0, 255) : 250.0, # lake/swamp/marsh value
    (71, 51, 3) : 0.0, # paved road value
    (0, 0, 0) : 0, # footpath value
    (205, 0, 101): math.inf, # out of bounds value
}

# ----------------------------------------------

class PriorityQueue():
    def __init__(self):
        # keep both of these same size, keep indexs in line
        self.cost_list: List[float] = []
        self.cord_list: List[Coordinate] = []

    def isEmpty(self) -> bool:
        return len(self.cost_list) == 0

    def insert(self, cord, cost):
        self.cord_list.append(cord)
        self.cost_list.append(cost)

    def pop(self) -> str:
        """
        return the least costly state to travel to while still getting
        us closer to the goal.
        cost derived from f(n) = h(n) + g + w

        h(n) = straight line 3d distance from goal state
        g = distance to move to coordinate (10.29 or 7.55)
        w = weight dependent on terrain value
        """
        min_val = 0
        for i in range(len(self.cost_list)):
            if self.cost_list[i] < self.cost_list[min_val]:
                min_val = i
        coordinate = self.cord_list[min_val]
        del self.cord_list[min_val]
        del self.cost_list[min_val]        
        return coordinate



class Coordinate:
    def __init__(self, x, y):
        self.x: int = x
        self.y: int = y

    def __eq__(self, other):
        if isinstance(other, Coordinate):
             return self.x == other.x and self.y == other.y
        return False

    def __hash__(self):
        return hash((self.x, self.y))

    def __repr__(self):
        return f"Coordinate(x={self.x}, y={self.y})"  
           


def calc_3d_distance(current_state: Coordinate, goal_state: Coordinate, elev_map: Dict[Coordinate, float]) -> float:
    """
    Calculate the 3d distance between 2 coordinates A = (x, y, z) and B = (x, y, z).
    """
    difference_x = abs(current_state.x - goal_state.x) * longitude
    difference_y = abs(current_state.y - goal_state.y) * latitude
    difference_z = abs(elev_map[current_state] - elev_map[goal_state])

    sum_of_differences = (difference_x ** 2) + (difference_y ** 2) + (difference_z ** 2)

    return math.sqrt(sum_of_differences)



def calc_total_length(path: List[Coordinate], elev_map: Dict[Coordinate, float]) -> float:
    """
    Calculates the total distance traveled between all landmarks
    """
    total: float = 0.0
    for i in range(1, len(path)):
        total += calc_3d_distance(path[i - 1], path[i], elev_map)
    return total    



def init_terrain_weights(file_name: str) -> Dict[Coordinate, float]:
    """
    Initializes a 2d array of integer weights corresponding to their
    terrain type.
    """
    terrain_weights: Dict[Coordinate, float] = dict()

    try:
        image = Image.open(file_name)
        image = image.convert('RGB')
        for x in range(width):
            for y in range(height):
                terrain_weights[Coordinate(x, y)] = rbg_weights.get(image.getpixel((x, y)))

        return terrain_weights        
    except FileNotFoundError:
        print(f"Error: Could not find image with name {file_name}")
        sys.exit(1)
    except IOError:
        print(f"Error: A problem occured while reading {file_name}")        



def init_elevation(file_name: str) -> Dict[Coordinate, float]:
    """
    Initializes a 2d array of floats representing the elevation at each pixel.

    Assumption: The elevation is absolute (ex. relative to sea level) and not based
    on where you are in the map
    """
    elevation_map: Dict[Coordinate, float] = dict()

    try:
        with open(file_name, 'r') as file:
            
            for y, line in enumerate(file):
                 height_values = line.strip().split()
                 # due to the line containing 400 whitespace seperated values,
                 # we must splice the array to get the first 395 values
                 for x, weight in enumerate(height_values[:-5]):
                    elevation_map[Coordinate(x, y)] = float(weight)
            return elevation_map          
    except FileNotFoundError:
            print(f"Error: Could not find a text file with the name {file_name}")
    except IOError:
            print(f"Error: A problem occured attempting to read {file_name}")        



def init_path(file_name: str) -> List[Coordinate]:
    """
    Initializes the ordered list of landmarks that the path must go through.
    """
    path = []

    try:
        with open(file_name, 'r') as file:
            for line in file:
                x, y = line.strip().split()
                path.append(Coordinate(int(x), int(y)))

        return path    
    except FileNotFoundError:
            print(f"Error: Could not find a text file with the name {file_name}")
    except IOError:
            print(f"Error: A problem occured attempting to read {file_name}")



def modify_image(path: List[Coordinate], input_name: str, output_name: str):
    """
    Adds the path to the output image using the specified path color (118, 63, 231)
    """
    try:
        # image to be manipulated
        img = Image.open(input_name)
        img = img.convert("RGB")

        pixels = img.load()

        for i in range(len(path)):
            pixels[path[i].x, path[i].y] = (118, 63, 231)

        img.save(output_name)    
    except FileNotFoundError:
            print(f"Error: Could not find a text file with the name {input_name}")
    except IOError:
            print(f"Error: A problem occured attempting to read or create a file")              



def generate_neighbors(state: Coordinate, terrain_weights: Dict[Coordinate, float], 
                       elevation: Dict[Coordinate, float]) -> Dict[Coordinate, float]:
    """
    Generate the 4 neighbors and their associated cost to travel to (heuristic not included)
    """
    neighbors = dict()

    # generate northern neighbor
    if state.y != height - 1:
        north = Coordinate(state.x, state.y + 1)
        neighbors[north] = longitude + terrain_weights[north] + abs(elevation[state] - elevation[north])
    # generate eastern neighbor
    if state.x != width - 1:
        east = Coordinate(state.x + 1, state.y)
        neighbors[east] = latitude + terrain_weights[east] + abs(elevation[state] - elevation[east])
    # generate southern neighbor
    if state.y != 0:
        south = Coordinate(state.x, state.y - 1)
        neighbors[south] = longitude + terrain_weights[south] + abs(elevation[state] - elevation[south])
    # generate western neighbor    
    if state.x != 0:
        west = Coordinate(state.x - 1, state.y)
        neighbors[west] = latitude + terrain_weights[west] + abs(elevation[state] - elevation[west])
    return neighbors                    
         
            

def astar(initial_state: Coordinate, goal_state: Coordinate,
           terrain_weights: Dict[Coordinate, float], 
           elevation_map: Dict[Coordinate, float]) -> List[Coordinate]:
    """
    Maps the most optimal path from current state to goal state.
    """
    frontier = PriorityQueue()
    frontier.insert(initial_state, 0)
    # caching straight line 3d distance to goal so we do not have to calculate multiple times..
    heuristic: Dict[Coordinate, float] = dict()

    came_from: Dict[Coordinate, Coordinate] = dict()
    cost_so_far: Dict[Coordinate, float] = dict()

    came_from[initial_state] = None
    cost_so_far[initial_state] = 0

    while not frontier.isEmpty():
        current: Coordinate = frontier.pop()

        if current == goal_state:
            break

        neighbors = generate_neighbors(current, terrain_weights, elevation_map)

        for n_coordinate, n_cost in neighbors.items():
            new_cost = cost_so_far[current] + n_cost
            if n_coordinate not in cost_so_far or new_cost < cost_so_far[n_coordinate]:
                cost_so_far[n_coordinate] = new_cost
                # add neighbor to priority queue with new_cost + heuristic
                if n_coordinate not in heuristic:
                     heuristic[n_coordinate] = calc_3d_distance(n_coordinate, goal_state, elevation_map)
                frontier.insert(n_coordinate, new_cost + heuristic[n_coordinate])  
                came_from[n_coordinate] = current

    path: List[Coordinate] = []
    current = goal_state             
    # building the path based off of the came_from dict
    while current != initial_state:
        path.append(current)
        try:
            current = came_from[current]
        except KeyError:
            print(f"No solution")
            sys.exit(1)
    path.append(initial_state)
    path.reverse()
    return path



def main():
    if (len(sys.argv) != 5):
        print(f"Error: Expected 4 arguements but received {len(sys.argv) - 1}.")
        sys.exit(1)
    else:
        terrain_weights = init_terrain_weights(sys.argv[1])

        elevation_map = init_elevation(sys.argv[2])

        landmarks = init_path(sys.argv[3])

        total_path: List[Coordinate] = []
        
        for i in range(1, len(landmarks)):
            path = astar(landmarks[i - 1], landmarks[i], terrain_weights, elevation_map)
            total_path.extend(path)

        total_length = calc_total_length(total_path, elevation_map)
        print(total_length)

        modify_image(total_path, sys.argv[1], sys.argv[4])    

        # TODO manipulate output image and save under new file name






if __name__ == "__main__":
    main()
