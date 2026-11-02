import math
import random
import itertools
from queue import PriorityQueue

############################################################
# CIS 5210: Homework 3
############################################################

student_name = ""

############################################################
# Section 1: Tile Puzzle
############################################################


def create_tile_puzzle(rows, cols):
    tiles = list(range(1, rows * cols)) + [0]
    board = [tiles[r * cols:(r + 1) * cols] for r in range(rows)]
    return TilePuzzle(board)


class TilePuzzle(object):

    def __init__(self, board):
        self.board = board
        self.rows = len(board)
        self.cols = len(board[0])
        for r in range(self.rows):
            for c in range(self.cols):
                if board[r][c] == 0:
                    self.empty_r = r
                    self.empty_c = c

    def get_board(self):
        return self.board

    def perform_move(self, direction):
        r, c = self.empty_r, self.empty_c
        if direction == "up":
            nr, nc = r - 1, c
        elif direction == "down":
            nr, nc = r + 1, c
        elif direction == "left":
            nr, nc = r, c - 1
        elif direction == "right":
            nr, nc = r, c + 1
        else:
            return False
        if 0 <= nr < self.rows and 0 <= nc < self.cols:
            tile = self.board[nr][nc]
            self.board[nr][nc] = self.board[r][c]
            self.board[r][c] = tile
            self.empty_r, self.empty_c = nr, nc
            return True
        return False

    def scramble(self, num_moves):
        directions = ["up", "down", "left", "right"]
        for _ in range(num_moves):
            self.perform_move(random.choice(directions))

    def is_solved(self):
        n = self.rows * self.cols
        flat = [self.board[r][c]
                for r in range(self.rows)
                for c in range(self.cols)]
        return flat == list(range(1, n)) + [0]

    def copy(self):
        return TilePuzzle([row[:] for row in self.board])

    def successors(self):
        for direction in ["up", "down", "left", "right"]:
            new_puzzle = self.copy()
            if new_puzzle.perform_move(direction):
                yield direction, new_puzzle

    def iddfs_helper(self, limit, moves):
        if self.is_solved():
            yield moves
            return
        if limit == 0:
            return
        for direction, new_puzzle in self.successors():
            yield from new_puzzle.iddfs_helper(
                limit - 1, moves + [direction]
            )

    def find_solutions_iddfs(self):
        for limit in itertools.count():
            solutions = list(self.iddfs_helper(limit, []))
            if solutions:
                yield from solutions
                return

    def find_solution_a_star(self):
        def heuristic(board):
            cols = len(board[0])
            total = 0
            for r, row in enumerate(board):
                for c, v in enumerate(row):
                    if v != 0:
                        gr = (v - 1) // cols
                        gc = (v - 1) % cols
                        total += abs(r - gr) + abs(c - gc)
            return total

        pq = PriorityQueue()
        counter = itertools.count()
        h = heuristic(self.board)
        pq.put((h, next(counter), self, []))
        visited = set()

        while not pq.empty():
            f, _, puzzle, moves = pq.get()
            state = tuple(tuple(row) for row in puzzle.get_board())
            if state in visited:
                continue
            visited.add(state)
            if puzzle.is_solved():
                return moves
            g = len(moves)
            for direction, new_puzzle in puzzle.successors():
                new_state = tuple(
                    tuple(row) for row in new_puzzle.get_board()
                )
                if new_state not in visited:
                    new_g = g + 1
                    new_h = heuristic(new_puzzle.get_board())
                    pq.put((new_g + new_h, next(counter),
                            new_puzzle, moves + [direction]))

        return None


############################################################
# Section 2: Grid Navigation
############################################################


def find_path(start, goal, scene):
    if scene[start[0]][start[1]] or scene[goal[0]][goal[1]]:
        return None

    rows = len(scene)
    cols = len(scene[0])

    def heuristic(pos):
        return math.sqrt(
            (pos[0] - goal[0]) ** 2 + (pos[1] - goal[1]) ** 2
        )

    directions = [
        (-1, 0, 1.0), (1, 0, 1.0), (0, -1, 1.0), (0, 1, 1.0),
        (-1, -1, math.sqrt(2)), (-1, 1, math.sqrt(2)),
        (1, -1, math.sqrt(2)), (1, 1, math.sqrt(2))
    ]

    pq = PriorityQueue()
    counter = itertools.count()
    pq.put((heuristic(start), next(counter), 0.0, start, [start]))
    visited = set()

    while not pq.empty():
        f, _, g, pos, path = pq.get()
        if pos in visited:
            continue
        visited.add(pos)
        if pos == goal:
            return path
        for dr, dc, cost in directions:
            nr, nc = pos[0] + dr, pos[1] + dc
            if 0 <= nr < rows and 0 <= nc < cols and not scene[nr][nc]:
                new_pos = (nr, nc)
                if new_pos not in visited:
                    new_g = g + cost
                    new_h = heuristic(new_pos)
                    pq.put((new_g + new_h, next(counter),
                            new_g, new_pos, path + [new_pos]))

    return None


############################################################
# Section 3: Linear Disk Movement, Revisited
############################################################


def solve_distinct_disks(length, n):
    start = tuple(i if i < n else -1 for i in range(length))
    goal_list = [-1] * length
    for i in range(n):
        goal_list[length - 1 - i] = i
    goal = tuple(goal_list)

    if start == goal:
        return []

    def heuristic(state):
        total = 0
        for pos in range(length):
            if state[pos] != -1:
                disk = state[pos]
                target = length - 1 - disk
                total += (abs(pos - target) + 1) // 2
        return total

    pq = PriorityQueue()
    counter = itertools.count()
    h = heuristic(start)
    pq.put((h, next(counter), 0, start, []))
    visited = set()

    while not pq.empty():
        f, _, g, state, moves = pq.get()
        if state in visited:
            continue
        visited.add(state)
        if state == goal:
            return moves
        for i in range(length):
            if state[i] == -1:
                continue
            targets = []
            if i + 1 < length and state[i + 1] == -1:
                targets.append(i + 1)
            if (i + 2 < length and state[i + 1] != -1
                    and state[i + 2] == -1):
                targets.append(i + 2)
            if i - 1 >= 0 and state[i - 1] == -1:
                targets.append(i - 1)
            if (i - 2 >= 0 and state[i - 1] != -1
                    and state[i - 2] == -1):
                targets.append(i - 2)
            for j in targets:
                lst = list(state)
                lst[j] = lst[i]
                lst[i] = -1
                new_state = tuple(lst)
                if new_state not in visited:
                    new_g = g + 1
                    new_h = heuristic(new_state)
                    pq.put((new_g + new_h, next(counter),
                            new_g, new_state, moves + [(i, j)]))

    return None


############################################################
# Section 4: Feedback
############################################################

# Just an approximation is fine.
feedback_question_1 = """
About 10 hours.
"""

feedback_question_2 = """
Grid navigation was the most challenging part of this assignment for me.
I had trouble getting the costs right for diagonal moves and making sure
the heuristic was still admissible with those varying step sizes.
It took a few tries to get the priority queue updates working correctly.
"""

feedback_question_3 = """
The linear disk movement problem was the most interesting section because
figuring out the heuristic felt more open ended than the other problems.
It was satisfying once the search was running efficiently with a good
admissible formulation. I think a little more guidance on heuristic
design for that problem would have been helpful in the spec.
"""

############################################################
# Section 5: Acknowledgements & reflections
############################################################

# Did you collaborate with anyone or use generative AI?
acknowledgements = """
I used Claude to help me when I was stuck with this homework
not by asking for the direct answer but ask me questions to
prompt me to develop and write the answer on my own. Using an AI
tool sped up my thinking process but I practiced tracing
through edge cases to make sure I understood what each function does.
"""
