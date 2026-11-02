import math
import random
from collections import deque

############################################################
# CIS 5210: Homework 2
############################################################

student_name = ""

############################################################
# Section 1: N-Queens
############################################################


def num_placements_all(n):
    return math.comb(n * n, n)


def num_placements_one_per_row(n):
    return n ** n


def n_queens_valid(board):
    for i in range(len(board)):
        for j in range(i + 1, len(board)):
            if board[i] == board[j]:
                return False
            if abs(board[i] - board[j]) == abs(i - j):
                return False
    return True


def n_queens_helper(n, board):
    if len(board) == n:
        yield board
        return
    for col in range(n):
        new_board = board + [col]
        if n_queens_valid(new_board):
            yield from n_queens_helper(n, new_board)


def n_queens_solutions(n):
    return list(n_queens_helper(n, []))


############################################################
# Section 2: Lights Out
############################################################


class LightsOutPuzzle(object):

    def __init__(self, board):
        self.board = board
        self.rows = len(board)
        self.cols = len(board[0])

    def get_board(self):
        return self.board

    def perform_move(self, row, col):
        for r, c in [(row, col), (row - 1, col), (row + 1, col),
                     (row, col - 1), (row, col + 1)]:
            if 0 <= r < self.rows and 0 <= c < self.cols:
                self.board[r][c] = not self.board[r][c]

    def scramble(self):
        for row in range(self.rows):
            for col in range(self.cols):
                if random.random() < 0.5:
                    self.perform_move(row, col)

    def is_solved(self):
        return all(
            not self.board[r][c]
            for r in range(self.rows)
            for c in range(self.cols)
        )

    def copy(self):
        return LightsOutPuzzle([row[:] for row in self.board])

    def successors(self):
        for row in range(self.rows):
            for col in range(self.cols):
                new_puzzle = self.copy()
                new_puzzle.perform_move(row, col)
                yield (row, col), new_puzzle

    def find_solution(self):
        def to_tuple(board):
            return tuple(tuple(row) for row in board)

        start = to_tuple(self.board)
        if self.is_solved():
            return []

        queue = deque([self])
        visited = {start: (None, None)}

        while queue:
            puzzle = queue.popleft()
            cur = to_tuple(puzzle.get_board())
            for move, new_puzzle in puzzle.successors():
                state = to_tuple(new_puzzle.get_board())
                if state not in visited:
                    visited[state] = (cur, move)
                    if new_puzzle.is_solved():
                        path = []
                        s = state
                        while visited[s][1] is not None:
                            path.append(visited[s][1])
                            s = visited[s][0]
                        return list(reversed(path))
                    queue.append(new_puzzle)

        return None


def create_puzzle(rows, cols):
    return LightsOutPuzzle([[False] * cols for _ in range(rows)])


############################################################
# Section 3: Linear Disk Movement
############################################################


def solve_identical_disks(length, n):
    start = tuple(i < n for i in range(length))
    goal = tuple(i >= length - n for i in range(length))

    if start == goal:
        return []

    queue = deque([start])
    visited = {start: (None, None)}

    while queue:
        state = queue.popleft()
        for i in range(length):
            if not state[i]:
                continue
            targets = []
            if i + 1 < length and not state[i + 1]:
                targets.append(i + 1)
            if i + 2 < length and state[i + 1] and not state[i + 2]:
                targets.append(i + 2)
            if i - 1 >= 0 and not state[i - 1]:
                targets.append(i - 1)
            if i - 2 >= 0 and state[i - 1] and not state[i - 2]:
                targets.append(i - 2)
            for j in targets:
                lst = list(state)
                lst[i], lst[j] = False, True
                new_state = tuple(lst)
                if new_state not in visited:
                    visited[new_state] = (state, (i, j))
                    if new_state == goal:
                        path = []
                        cur = new_state
                        while visited[cur][1] is not None:
                            path.append(visited[cur][1])
                            cur = visited[cur][0]
                        return list(reversed(path))
                    queue.append(new_state)

    return None


def solve_distinct_disks(length, n):
    start = tuple(i if i < n else -1 for i in range(length))
    goal_list = [-1] * length
    for i in range(n):
        goal_list[length - 1 - i] = i
    goal = tuple(goal_list)

    if start == goal:
        return []

    queue = deque([start])
    visited = {start: (None, None)}

    while queue:
        state = queue.popleft()
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
                    visited[new_state] = (state, (i, j))
                    if new_state == goal:
                        path = []
                        cur = new_state
                        while visited[cur][1] is not None:
                            path.append(visited[cur][1])
                            cur = visited[cur][0]
                        return list(reversed(path))
                    queue.append(new_state)

    return None


############################################################
# Section 4: Feedback
############################################################

# Just an approximation is fine.
feedback_question_1 = """
About 3 hours.
"""

feedback_question_2 = """
Definitely the Lights Out find_solution and the distinct disks.
For find_solution I kept messing up the visited set logic and my
BFS was revisiting states for a while before I figured out the
tuple trick. The distinct disks was also really confusing at first
because I wasn't sure how to define the goal state, I had to
read the problem a few times.
"""

feedback_question_3 = """
N-Queens was honestly pretty enjoyable, it felt satisfying when the DFS
worked and I got all 92 solutions for n=8. I kind of wish the
linear disk section had said a bit more upfront about how to
represent the state because I spent way too long on that before
actually writing the BFS.
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
