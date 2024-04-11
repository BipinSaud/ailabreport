#1. Write a program in python to implement Breadth First Search (BFS).

'''
graph = {
         'A': ['B', 'C'],
         'B': ['D', 'E'],
         'C': ['F'],
         'D': ['G', 'H'],
         'E': ['I'],
         'F': [],
         'G': [],
         'H': [],
         'I': []
}
visited = []  # List for visited nodes.
queue = []  # Initialize a queue
def bfs(visited, graph, node):  # function for BFS
         visited.append(node)
         queue.append(node)
         while queue:          # Creating loop to visit each
    # node
             m = queue.pop(0)
             # print '->' after each node except the last one
             print(m, end='->' if m != 'I' else '')
             for neighbour in graph[m]:
                 if neighbour not in visited:
                     visited.append(neighbour)
                     queue.append(neighbour)
     # Driver Code
print("Breadth-First Search travels through : ")
bfs(visited, graph, 'A')

'''

#2. Write a program in python to implement Depth First Search(DFS).

'''

graph = {
      'A': ['B', 'C'],
      'B': ['D', 'E'],
      'C': ['F'],
      'D': ['G', 'H'],
      'E': ['I'],
      'F': [],
      'G': [],
      'H': [],
      'I': []
}

visited = set()  # Set to keep track of visited nodes of graph.
def dfs(visited, graph, node):  # function for dfs
    if node not in visited:
        # print '->' after each node except the last one
        print(node, end='->' if node != 'F' else '')
        visited.add(node)
        for neighbour in graph[node]:
            dfs(visited, graph, neighbour)
# Driver Code
print("Depth-First Search travels  through: ")
dfs(visited, graph, 'A')

'''

#3. Write a program in python to implement Depth Limited Search(DLS).
'''
graph = {
      'A':['B','C'],
      'B':['D','E'],
      'C':['F','G'],
      'D':['H','I'],
      'E':['J','K'],
      'F':['L','M'],
      'G':['N','O'],
      'H':[],
      'I':[],
      'J':[],
      'K':[],
      'L':[],
      'M':[],
      'N':[],
      'O':[]
}
def DLS(start,goal,path,level,maxD):
    print('\nCurrent level-->',level)
    print('Goal node testing for',start)
    path.append(start)
    if start == goal:
      print("Goal test successful")
      return path
    print('Goal node testing failed')
    if level==maxD:
      return False
    print('\nExpanding the current node',start)
    for child in graph[start]:
      if DLS(child,goal,path,level+1,maxD):
        return path
      path.pop()
    return False
start = 'A'
goal = input('Enter the goal node:-')
maxD = int(input("Enter the maximum depth limit:-"))
print()
path = list()
res = DLS(start,goal,path,0,maxD)
if(res):
    print("Path to goal node available")
    print("Path",path)
else:
    print("No path available for the goal node in given depth limit")

'''

#4. Write a program in python to implement Iterative Deepening Depth First Search.

'''

graph = {
    'A': ['B', 'C'],
    'B': ['D', 'E'],
    'C': ['F', 'G'],
    'D': [],
    'E': [],
    'F': [],
    'G': []
}
nodes_visited_level = {}
current_level = -1

def dfs(node, goal, d_limit, start, visited, path):
    global current_level
    if start == d_limit:
        current_level += 1
        nodes_visited_level[current_level] = []
    nodes_visited_level[current_level].append(node)
    if node == goal:
        return "FOUND", path + [node]
    elif d_limit == 0:
        return "NOT_FOUND", None
    else:
        visited.add(node)
        for child in graph[node]:
            if child not in visited:
                result, tv_path = dfs(child, goal, d_limit - 1, start, visited, path + [node])
                if result == 'FOUND':
                    return "FOUND", tv_path
        return "NOT_FOUND", None

def iddfs(root, goal):
    d_limit = 0
    while True:
        visited = set()
        start = d_limit
        result, tv_path = dfs(root, goal, d_limit, start, visited, [])
        if result == "FOUND":
            return "Goal node found! Traversed path:" + '->'.join(tv_path)
        elif result == "NOT_FOUND":
            d_limit += 1

root = input("Enter the start node: ")
goal = input("Enter the goal node: ")
result = iddfs(root, goal)

for level, nodes in nodes_visited_level.items():
    print("Depth limit: " + str(level) + " Traversed path: ", end="")
    for node in nodes:
        print(node, end="")
        if node != nodes[-1]:
            print("->", end="")
    print()

print(result)


'''

#5. Write a program to implement Greedy Best First Search.

'''

graph = {
    'S': {'A': 3, 'B': 2},
    'A': {'C': 4, 'D': 1},
    'B': {'E': 3, 'F': 1},
    'C': {},
    'D': {},
    'E': {'H': 5},
    'F': {'I': 2, 'G': 3},
    'G': {},
    'H': {},
    'I': {},
}

heuristic = {
    'S': 13,
    'A': 12,
    'B': 4,
    'C': 7,
    'D': 3,
    'E': 8,
    'F': 2,
    'G': 0,
    'H': 4,
    'I': 9,
}

def gbfs(graph, heuristic, start, goal):
    visited = set()
    queue = [(heuristic[start], [start])]
    while queue:
        (h, path) = queue.pop(0)
        current_node = path[-1]
        if current_node == goal:
            return path
        visited.add(current_node)
        for neighbor, distance in graph[current_node].items():
            if neighbor not in visited:
                new_path = path + [neighbor]
                queue.append((heuristic[neighbor], new_path))
        queue.sort()  # Sort based on heuristic value
    return None

print("Greedy Best First Search")
start = input("Enter the start node: ")
goal = input("Enter the goal node: ")
traversed_path = gbfs(graph, heuristic, start, goal)
if traversed_path:
    print(f"Goal node found: {traversed_path}")
else:
    print("Goal node not found")

'''

#6. Write a program to implement A* Search.

'''
# Defining the graph nodes in dict with given costs to traverse
adj_list = {
    's': [('a', 1), ('g', 10)],
    'a': [('b', 2), ('c', 1)],
    'b': [('d', 5)],
    'c': [('d', 3), ('g', 4)],
    'd': [('g', 2)],
    'g': []
}

# Defining heuristic values for each node
heuristic = {
    's': 5,
    'a': 3,
    'b': 4,
    'c': 2,
    'd': 6,
    'g': 0
}

# A Star Search Algorithm
def astar_search(adj_list, heuristic, start_node, goal_node):
    open_list = set([start_node])
    closed_list = set([])
    g = {}
    g[start_node] = 0
    parents = {}
    parents[start_node] = start_node

    def get_neighbors(node):
        return adj_list[node]

    def h(node):
        return heuristic[node]

    while len(open_list) > 0:
        n = None
        for v in open_list:
            if n == None or g[v] + h(v) < g[n] + h(n):
                n = v

        if n == None:
            print('Path does not exist!')
            return None

        if n == goal_node:
            reconst_path = []
            while parents[n] != n:
                reconst_path.append(n)
                n = parents[n]
            reconst_path.append(start_node)
            reconst_path.reverse()
            print('Path found: {}'.format(reconst_path))
            return reconst_path

        for (m, weight) in get_neighbors(n):
            if m not in open_list and m not in closed_list:
                open_list.add(m)
                parents[m] = n
                g[m] = g[n] + weight
            else:
                if g[m] > g[n] + weight:
                    g[m] = g[n] + weight
                    parents[m] = n
                    if m in closed_list:
                        closed_list.remove(m)
                        open_list.add(m)

        open_list.remove(n)
        closed_list.add(n)

    print('Path does not exist!')
    return None

print("----- A star search -----")
start_node = input("Enter the start node: ")
goal_node = input("Enter the goal node: ")
astar_search(adj_list, heuristic, start_node, goal_node)

'''

#7. Write a program to implement Uniform Cost Search.
'''

def uniform_cost_search(goal, start):
    global graph, cost
    answer = []
    queue = []
    for i in range(len(goal)):
        answer.append(float('inf'))  # Initialize answer vector with infinity

    queue.append([0, start])  # Insert the starting index
    visited = set()  # Use a set to store visited nodes
    count = 0

    while queue:
        queue.sort()  # Sort the queue based on cost
        p = queue.pop(0)
        cost_to_current_node, current_node = p[0], p[1]

        if current_node in visited:
            continue

        visited.add(current_node)

        if current_node in goal:
            index = goal.index(current_node)
            if answer[index] == float('inf'):
                count += 1
            if answer[index] > cost_to_current_node:
                answer[index] = cost_to_current_node

            if count == len(goal):
                return answer

        for neighbor, edge_cost in graph[current_node]:
            if neighbor not in visited:
                queue.append([cost_to_current_node + edge_cost, neighbor])

    return answer

if __name__ == '__main__':
    graph, cost = [[] for _ in range(8)], {}
    graph[0].append((1, 2))
    graph[0].append((3, 5))
    graph[3].append((1, 5))
    graph[3].append((6, 6))
    graph[3].append((4, 2))
    graph[1].append((6, 1))
    graph[4].append((2, 4))
    graph[4].append((5, 3))
    graph[2].append((1, 4))
    graph[5].append((2, 6))
    graph[5].append((6, 3))
    graph[6].append((4, 7))

    cost[(0, 1)] = 2
    cost[(0, 3)] = 5
    cost[(1, 6)] = 1
    cost[(3, 1)] = 5
    cost[(3, 6)] = 6
    cost[(3, 4)] = 2
    cost[(2, 1)] = 4
    cost[(4, 2)] = 4
    cost[(4, 5)] = 3
    cost[(5, 2)] = 6
    cost[(5, 6)] = 3
    cost[(6, 4)] = 7

    goal = [6]

    answer = uniform_cost_search(goal, 0)

    print("Minimum cost from 0 to 6 is = ", answer[0])

'''

#8. Write a program to implement Alpha-Beta Pruning Algorithm.
'''

tree = [
    [[5, 1, 2], [8, -8, -9]],
    [[9, 4, 5], [-3, 4, 3]]
]  # Tree to search
root = 0  # Root depth
pruned = 0  # Times pruned

# Function to search tree
def children(branch, depth, alpha, beta):
    global root, pruned  # Global variables
    i = 0  # Index of child
    for child in branch:
        if type(child) is list:
            # If child is a list, call children function recursively
            (nalpha, nbeta) = children(child, depth + 1, alpha, beta)
            if depth % 2 == 1:
                beta = min(beta, nalpha)
            else:
                alpha = max(alpha, nbeta)
            branch[i] = nalpha if depth % 2 == 0 else nbeta
            i += 1
        else:
            if depth % 2 == 0 and alpha < child:
                alpha = child
            if depth % 2 == 1 and beta > child:
                beta = child
            if alpha >= beta:
                pruned += 1
                break
    return (alpha, beta)

# Function to call search
def alphabeta(branch=tree, depth=root, alpha=-15, beta=15):
    global pruned
    (alpha, beta) = children(branch, depth, alpha, beta)
    if depth == root:
        best_move = max(branch) if depth % 2 == 0 else min(branch)
        print("(alpha, beta): ", alpha, beta)
        print("Result: ", best_move)
        print("Times pruned: ", pruned)
    return (alpha, beta, branch, pruned)

if __name__ == "__main__":
    alphabeta()

'''

#9. Write a program that implements water jug problem .
'''
from collections import deque

# Function to find all possible states from the current state
def get_next_states(current_state, jug1_capacity, jug2_capacity):
    next_states = []

    # Empty Jug 1
    next_states.append((0, current_state[1], "Empty Jug 1"))

    # Empty Jug 2
    next_states.append((current_state[0], 0, "Empty Jug 2"))

    # Fill Jug 1
    next_states.append((jug1_capacity, current_state[1], "Fill Jug 1"))

    # Fill Jug 2
    next_states.append((current_state[0], jug2_capacity, "Fill Jug 2"))

    # Pour water from Jug 1 to Jug 2
    pour_amount = min(current_state[0], jug2_capacity - current_state[1])
    next_states.append((current_state[0] - pour_amount, current_state[1] + pour_amount, "Pour from Jug 1 to Jug 2"))

    # Pour water from Jug 2 to Jug 1
    pour_amount = min(jug1_capacity - current_state[0], current_state[1])
    next_states.append((current_state[0] + pour_amount, current_state[1] - pour_amount, "Pour from Jug 2 to Jug 1"))

    return next_states

# Breadth-First Search to find the solution
def water_jug_problem(jug1_capacity, jug2_capacity, target_amount):
    visited = set()  # Set to keep track of visited states
    queue = deque([(0, 0, [])])  # Starting state: (0, 0)
    visited.add((0, 0))

    while queue:
        current_state = queue.popleft()

        if current_state[0] == target_amount or current_state[1] == target_amount:
            return current_state[2]

        next_states = get_next_states(current_state[:2], jug1_capacity, jug2_capacity)

        for state in next_states:
            if state[:2] not in visited:
                visited.add(state[:2])
                new_path = current_state[2] + [state[2]]  # Append the action to the current path
                queue.append((state[0], state[1], new_path))

    return None  # Solution not found

# Main function to run the program
if __name__ == "__main__":
    jug1_capacity = int(input("Enter the capacity of Jug 1: "))
    jug2_capacity = int(input("Enter the capacity of Jug 2: "))
    target_amount = int(input("Enter the target amount of water: "))

    solution = water_jug_problem(jug1_capacity, jug2_capacity, target_amount)

    if solution:
        print("Solution found:")
        for step in solution:
            print(step)
    else:
        print("Solution not found.")
'''

#10. Write a program that implements Tic Tac Toe  using the Minimax algorithm.
'''
import random

board = [' ' for x in range(9)]
player = 1
Win = 1
Draw = -1
Running = 0
Stop = 1
###########################
Game = Running
Mark = 'X'


# This Function Draws Game Board
def DrawBoard():
    print(" %c | %c | %c " % (board[0], board[1], board[2]))
    print("___|___|___")
    print(" %c | %c | %c " % (board[3], board[4], board[5]))
    print("___|___|___")
    print(" %c | %c | %c " % (board[6], board[7], board[8]))
    print("   |   |   ")


# This Function Checks position is empty or not
def CheckPosition(x):
    if (board[x] == ' '):
        return True
    else:
        return False


# This Function Checks player has won or not
def CheckWin():
    global Game
    # Horizontal winning condition
    if (board[0] == board[1] and board[1] == board[2] and board[0] != ' '):
        Game = Win
    elif (board[3] == board[4] and board[4] == board[5] and board[3] != ' '):
        Game = Win
    elif (board[6] == board[7] and board[7] == board[8] and board[6] != ' '):
        Game = Win

    # Vertical Winning Condition
    elif (board[0] == board[3] and board[3] == board[6] and board[0] != ' '):
        Game = Win
    elif (board[1] == board[4] and board[4] == board[7] and board[1] != ' '):
        Game = Win
    elif (board[2] == board[5] and board[5] == board[8] and board[2] != ' '):
        Game = Win

    # Diagonal Winning Condition
    elif (board[0] == board[4] and board[4] == board[8] and board[4] != ' '):
        Game = Win
    elif (board[2] == board[4] and board[4] == board[6] and board[4] != ' '):
        Game = Win

    # Match Tie or Draw Condition
    elif (board[0] != ' ' and
          board[1] != ' ' and
          board[2] != ' ' and
          board[3] != ' ' and
          board[4] != ' ' and
          board[5] != ' ' and
          board[6] != ' ' and
          board[7] != ' ' and
          board[8] != ' '):
        Game = Draw
    else:
        Game = Running


print("---- Tic-Tac-Toe ----\n\n")
print("Computer [X] --- User [O]\n\n\n")

while (Game == Running):
    DrawBoard()
    if (player % 2 != 0):
        print("Computer's chance")
        Mark = 'X'
        choice = random.randint(0, 8)
    else:
        print("User's chance")
        Mark = 'O'
        choice = int(input("Enter the position [0-8]: "))

    if (CheckPosition(choice)):
        board[choice] = Mark
        player += 1
        CheckWin()

DrawBoard()
if (Game == Draw):
    print("Game is tied!🏅🏆")
elif (Game == Win):
    player -= 1

if (player % 2 != 0):
    print("Computer Wins!🏆")
else:
    print("User Wins!🏆")

	'''

#11. Write a program that implements Constraints Satisfaction Problem.

'''
from itertools import permutations

# Define the variables representing Australian states
variables = ('WA', 'NT', 'SA', 'Q', 'NSW', 'V', 'T')

# Define the domains as colors for each state
domains = {v: ['red', 'green', 'blue'] for v in variables}

# Define a function to check if neighbors have different values
def const_different(variables, values):
    return values[0] != values[1]  # Expect the value of the neighbors to be different

# Define constraints between neighbors (states) to have different colors (values)
constraints = [
    (('WA', 'NT'), const_different),
    (('WA', 'SA'), const_different),
    (('SA', 'NT'), const_different),
    (('SA', 'Q'), const_different),
    (('NT', 'Q'), const_different),
    (('SA', 'NSW'), const_different),
    (('Q', 'NSW'), const_different),
    (('SA', 'V'), const_different),
    (('NSW', 'V'), const_different),
]

# Define a function to check if the assignment satisfies all constraints
def is_valid(assignment):
    for (var1, var2), constraint in constraints:
        if var1 not in assignment or var2 not in assignment:
            continue  # Skip constraints involving unassigned variables
        if (assignment[var1], assignment[var2]) not in permutations(domains[var1], 2):
            continue  # Skip if the constraint does not involve the assigned values
        if not constraint((var1, var2), (assignment[var1], assignment[var2])):
            return False
    return True


# Define a function to backtrack search for a solution
def backtrack_search(assignment):
    if len(assignment) == len(variables):
        return assignment
    var = next(var for var in variables if var not in assignment)
    for value in domains[var]:
        new_assignment = assignment.copy()
        new_assignment[var] = value
        if is_valid(new_assignment):
            result = backtrack_search(new_assignment)
            if result is not None:
                return result
    return None

# Solve the CSP problem using backtrack search algorithm
solution_backtrack = backtrack_search({})

# Print the solution
print("Solution using backtrack search:")
print(solution_backtrack)
'''

#12.Write a program to find local maxima using hill climb search algorithm
'''

# This dictionary holds all the nodes with their successors and their corresponding heuristic value
adjList = {
    'A': [('B', 10), ('J', 8), ('F', 7)],
    'B': [('D', 4), ('C', 2)],
    'C': [('H', 0)],
    'E': [('I', 6)],
    'F': [('E', 5), ('G', 3)],
    'I': [('K', 0)],
    'J': [('K', 0)],
}

# root node
initial_node = str(input("Input initial node: ")).capitalize()
# holds heuristic value of root node
initial_value = eval(input(f"Input {initial_node}'s heuristic value: "))

# Function to sort the selected list in ascending order based on heuristic value


def sortList(new_list):
    new_list.sort(key=lambda x: x[1])
    return new_list


# Function to find shortest path using heuristic value
def hillClimbing_search(node, value):
    new_list = list()
    if node in adjList.keys():
        new_list = adjList[node]
        new_list = sortList(new_list)
        if (value > new_list[0][1]):
            value = new_list[0][1]
            node = new_list[0][0]
            hillClimbing_search(node, value)
        if (value < new_list[0][1]):
            print(
                f"\nLocal maxima at node: '{node}'\nHeuristic value: {value}")
    else:
        print(f"\nLocal maxima at node: '{node}'\nHeuristic value: {value}")


if __name__ == "__main__":
    hillClimbing_search(initial_node, initial_value) 
'''