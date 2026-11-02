############################################################
# Imports
############################################################

import copy
import random

############################################################
# CIS 5210: Homework 4
############################################################

student_name = ""

############################################################
# Section 1: Dominoes Game
############################################################


def create_dominoes_game(rows, cols):
    board = [[False] * cols for _ in range(rows)]
    return DominoesGame(board)


class DominoesGame(object):

    # Required
    def __init__(self, board):
        self.board = board
        self.rows = len(board)
        self.cols = len(board[0]) if board else 0

    def get_board(self):
        return self.board

    def reset(self):
        self.board = [[False] * self.cols for _ in range(self.rows)]

    def is_legal_move(self, row, col, vertical):
        if vertical:
            if row + 1 >= self.rows:
                return False
            return not self.board[row][col] and not self.board[row + 1][col]
        else:
            if col + 1 >= self.cols:
                return False
            return not self.board[row][col] and not self.board[row][col + 1]

    def legal_moves(self, vertical):
        for r in range(self.rows):
            for c in range(self.cols):
                if self.is_legal_move(r, c, vertical):
                    yield (r, c)

    def perform_move(self, row, col, vertical):
        self.board[row][col] = True
        if vertical:
            self.board[row + 1][col] = True
        else:
            self.board[row][col + 1] = True

    def game_over(self, vertical):
        return not any(True for _ in self.legal_moves(vertical))

    def copy(self):
        return DominoesGame(copy.deepcopy(self.board))

    def successors(self, vertical):
        for move in self.legal_moves(vertical):
            new_game = self.copy()
            new_game.perform_move(move[0], move[1], vertical)
            yield (move, new_game)

    def get_random_move(self, vertical):
        return random.choice(list(self.legal_moves(vertical)))

    # Required
    def get_best_move(self, vertical, limit):
        best_move, best_value, leaves = self._alpha_beta(
            vertical, limit, float('-inf'), float('inf')
        )
        return (best_move, best_value, leaves)

    def _evaluate(self, vertical):
        my_moves = sum(1 for _ in self.legal_moves(vertical))
        opp_moves = sum(1 for _ in self.legal_moves(not vertical))
        return my_moves - opp_moves

    def _alpha_beta(self, vertical, limit, alpha, beta):
        if self.game_over(vertical) or limit == 0:
            return (None, self._evaluate(vertical), 1)

        best_move = None
        total_leaves = 0

        for move, new_game in self.successors(vertical):
            _, value, leaves = new_game._alpha_beta(
                not vertical, limit - 1, -beta, -alpha
            )
            value = -value
            total_leaves += leaves

            if best_move is None or value > alpha:
                alpha = value
                best_move = move

            if alpha >= beta:
                break

        return (best_move, alpha, total_leaves)


############################################################
# Section 3: Feedback
############################################################


# Just an approximation is fine.
feedback_question_1 = """
Around 5 hours.
"""

feedback_question_2 = """
The trickiest part was getting the alpha-beta pruning to match the
textbook's Figure 5.7 exactly, especially negamax-style sign flipping
versus maintaining separate max/min branches. I kept getting off-by-one
errors in the leaf count until I realized terminal states (game over or
depth limit) both count as leaves regardless of which condition triggers
first. The evaluation function framing "my moves minus opponent moves"
from the perspective of the current player also required careful
thought about whose turn it is at each recursive level.
"""

feedback_question_3 = """
I liked that the game is simple enough to reason about by hand, which
made it possible to verify small test cases on paper and catch subtle
bugs in the search. If I could change one thing, I would have the spec
clarify earlier that the negamax formulation is acceptable (or even
preferred) over the separate maximizer/minimizer version that
realization saved a lot of code complexity.
"""

############################################################
# Section 4: Acknowledgements & reflections
############################################################

# Did you collaborate with anyone or use generative AI?
acknowledgements = """
I used Claude to help me when I was stuck with this homework
not by asking for the direct answer but ask me questions to
prompt me to develop and write the answer on my own. Using an AI
tool sped up my thinking process but I practiced tracing
through edge cases to make sure I understood what each function does.
"""
