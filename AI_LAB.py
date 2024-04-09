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
import sys

def print_board(board):
    for row in board:
        print(" | ".join(row))
        print("-" * 5)

def check_winner(board):
    # Check rows
    for row in board:
        if row[0] == row[1] == row[2] != " ":
            return row[0]

    # Check columns
    for col in range(3):
        if board[0][col] == board[1][col] == board[2][col] != " ":
            return board[0][col]

    # Check diagonals
    if board[0][0] == board[1][1] == board[2][2] != " ":
        return board[0][0]
    if board[0][2] == board[1][1] == board[2][0] != " ":
        return board[0][2]

    return None

def is_full(board):
    for row in board:
        for cell in row:
            if cell == " ":
                return False
    return True

def minimax(board, depth, maximizing_player):
    winner = check_winner(board)
    if winner:
        return 1 if winner == 'X' else -1 if winner == 'O' else 0

    if is_full(board):
        return 0

    if maximizing_player:
        max_eval = -sys.maxsize
        for i in range(3):
            for j in range(3):
                if board[i][j] == " ":
                    board[i][j] = 'X'
                    eval = minimax(board, depth+1, False)
                    board[i][j] = " "
                    max_eval = max(eval, max_eval)
        return max_eval
    else:
        min_eval = sys.maxsize
        for i in range(3):
            for j in range(3):
                if board[i][j] == " ":
                    board[i][j] = 'O'
                    eval = minimax(board, depth+1, True)
                    board[i][j] = " "
                    min_eval = min(eval, min_eval)
        return min_eval

def find_best_move(board):
    best_eval = -sys.maxsize
    best_move = None
    for i in range(3):
        for j in range(3):
            if board[i][j] == " ":
                board[i][j] = 'X'
                eval = minimax(board, 0, False)
                board[i][j] = " "
                if eval > best_eval:
                    best_eval = eval
                    best_move = (i, j)
    return best_move

def main():
    board = [[" " for _ in range(3)] for _ in range(3)]
    print("Welcome to Tic Tac Toe")
    print_board(board)

    while True:
        player_move = input("Enter your move (row column): ").split()
        row, col = map(int, player_move)
        if board[row][col] != " ":
            print("Invalid move. Cell already occupied.")
            continue
        board[row][col] = 'O'
        print_board(board)
        winner = check_winner(board)
        if winner:
            print(f"Player '{winner}' wins!")
            break
        if is_full(board):
            print("It's a draw!")
            break

        print("Computer is thinking...")
        computer_move = find_best_move(board)
        board[computer_move[0]][computer_move[1]] = 'X'
        print(f"Computer moves to ({computer_move[0]}, {computer_move[1]})")
        print_board(board)
        winner = check_winner(board)
        if winner:
            print(f"Computer '{winner}' wins!")
            break
        if is_full(board):
            print("It's a draw!")
            break

if __name__ == "__main__":
    main()
'''

#11. Write a program that implements Constraints Satisfaction Problem.
'''
puzzle = [[5, 3, 0, 0, 7, 0, 0, 0, 0], 
		[6, 0, 0, 1, 9, 5, 0, 0, 0], 
		[0, 9, 8, 0, 0, 0, 0, 6, 0], 
		[8, 0, 0, 0, 6, 0, 0, 0, 3], 
		[4, 0, 0, 8, 0, 3, 0, 0, 1], 
		[7, 0, 0, 0, 2, 0, 0, 0, 6], 
		[0, 6, 0, 0, 0, 0, 2, 8, 0], 
		[0, 0, 0, 4, 1, 9, 0, 0, 5], 
		[0, 0, 0, 0, 8, 0, 0, 0, 0] 
		] 

def print_sudoku(puzzle): 
	for i in range(9): 
		if i % 3 == 0 and i != 0: 
			print("- - - - - - - - - - - ") 
		for j in range(9): 
			if j % 3 == 0 and j != 0: 
				print(" | ", end="") 
			print(puzzle[i][j], end=" ") 
		print() 

print_sudoku(puzzle) 

class CSP: 
	def __init__(self, variables, Domains,constraints): 
		self.variables = variables 
		self.domains = Domains 
		self.constraints = constraints 
		self.solution = None

	def solve(self): 
		assignment = {} 
		self.solution = self.backtrack(assignment) 
		return self.solution 

	def backtrack(self, assignment): 
		if len(assignment) == len(self.variables): 
			return assignment 

		var = self.select_unassigned_variable(assignment) 
		for value in self.order_domain_values(var, assignment): 
			if self.is_consistent(var, value, assignment): 
				assignment[var] = value 
				result = self.backtrack(assignment) 
				if result is not None: 
					return result 
				del assignment[var] 
		return None

	def select_unassigned_variable(self, assignment): 
		unassigned_vars = [var for var in self.variables if var not in assignment] 
		return min(unassigned_vars, key=lambda var: len(self.domains[var])) 

	def order_domain_values(self, var, assignment): 
		return self.domains[var] 

	def is_consistent(self, var, value, assignment): 
		for constraint_var in self.constraints[var]: 
			if constraint_var in assignment and assignment[constraint_var] == value: 
				return False
		return True
	
	
# Variables 
variables = [(i, j) for i in range(9) for j in range(9)] 
# Domains 
Domains = {var: set(range(1, 10)) if puzzle[var[0]][var[1]] == 0
						else {puzzle[var[0]][var[1]]} for var in variables} 

# Add contraint 
def add_constraint(var): 
	constraints[var] = [] 
	for i in range(9): 
		if i != var[0]: 
			constraints[var].append((i, var[1])) 
		if i != var[1]: 
			constraints[var].append((var[0], i)) 
	sub_i, sub_j = var[0] // 3, var[1] // 3
	for i in range(sub_i * 3, (sub_i + 1) * 3): 
		for j in range(sub_j * 3, (sub_j + 1) * 3): 
			if (i, j) != var: 
				constraints[var].append((i, j)) 
# constraints		 
constraints = {} 
for i in range(9): 
	for j in range(9): 
		add_constraint((i, j)) 
		
# Solution 
print('*'*7,'Solution','*'*7) 
csp = CSP(variables, Domains, constraints) 
sol = csp.solve() 

solution = [[0 for i in range(9)] for i in range(9)] 
for i,j in sol: 
	solution[i][j]=sol[i,j] 
	
print_sudoku(solution)

'''