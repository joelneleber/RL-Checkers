
# File Management
import glob
import os

# Variable assistance
import random
import string

from typing import Dict, List

# Board manipulation
import numpy as np
import copy

from PolicyEnum import Policy

# Constants for pieces
BLACK = -1
WHITE = 1

MOVE_LIMIT = 500

debug = False

# Array values to string for printing
piece_string = {
    2: "WK",
    1: "W ",
    0: "  ",
    -1: "B ",
    -2: "BK",
}


class BoardItem:
    """
    Used for tracking the moves made in the game
    Each move creates a new BoardItem
    """
    board = None
    last_move = None
    next = None

    def __init__(self, b, lm):
        """
        Initialize a new BoardItem
        :param b: board array (8x8 numpy int)
        :param lm: last move ((int,int),(int,int))
        """
        self.board = b
        self.last_move = lm


class BoardStack:
    stack_head = None

    def push_board(self, b, last_move):
        """
        Add a move to the stack
        :param b: board array (8x8 numpy int)
        :param last_move: last move ((int,int),(int,int))
        :return: void
        """
        if not self.stack_head:
            self.stack_head = BoardItem(b, last_move)

        new_head = BoardItem(b, last_move)
        new_head.next = self.stack_head
        self.stack_head = new_head

    def pop_board(self):
        """
        Get the most recent item on the stack
        :return: The most recent BoardItem
        """
        if not self.stack_head:
            return None

        to_return = self.stack_head
        self.stack_head = to_return.next
        return to_return

    def size(self):
        """
        Get the size of the stack
        :return: int: The size of the stack
        """
        if not self.stack_head:
            return 0

        count = 0
        cur = self.stack_head
        while cur.next is not None:
            count += 1
            cur = cur.next
        return count

    def __str__(self):
        """
        Useful for debugging
        :return: The string equivalent of the whole stack
        """
        if not self.stack_head:
            return "Empty"

        count = "["
        cur = self.stack_head
        while cur.next is not None:
            count += str(cur.last_move) + ", " + str(cur.board) + ", "
            cur = cur.next
        return count + "]"


def view_coordinates_board_coordinates(input_coordinates: str) -> (int, int):
    """
    Converts user entered coordinates (comma separated) into two ints

    :param input_coordinates: user input in the following form: 0a,1b or a0,1b
    :return: a array coordinate equivalent
    """
    input_coordinates = input_coordinates.upper().strip().replace(" ", "")
    if input_coordinates[0] in string.ascii_uppercase:
        x = string.ascii_uppercase.find(input_coordinates[0])
        y = int(input_coordinates[1])
    else:
        x = string.ascii_uppercase.find(input_coordinates[1])
        y = int(input_coordinates[0])
    return x, y


def verify_location(new_y: int, new_x: int) -> bool:
    """
    Check if a location is valid

    :param new_y: new y value
    :param new_x: new x value
    :return: if the new location is valid
    """
    return 0 <= new_y <= 7 and 0 <= new_x <= 7


def add_move(dic, key, value):
    """
    Updates a dictionary key to store an array as its value and appends new values if the key already exists

    :param dic: an existing dictionary
    :param key: the key to be added to the dictionary
    :param value: the value to associate with the key
    :return: void
    """
    if key in dic:
        dic[key].append(value)
    else:
        dic[key] = [value]


class Board:
    """
    A checker board
    """
    board_array = np.zeros((8, 8), dtype=int)
    white_move: bool = True
    game_over: bool = False
    white_wins: bool = False
    draw: bool = False
    logging: bool = False
    stop_probability: float = 0
    white_policy: int = Policy.LOOKAHEAD
    black_policy: int = Policy.RANDOM
    to_file = None
    last_move = ((0, 0), (0, 0))
    board_stack = None
    black_depth = 1
    white_depth = 1
    total_moves = 0

    # Setup the board in the default configuration
    def __init__(self, outfile=None):
        """
        Initialize a new checkers board, will set the pieces correctly and set local vars
        :param outfile: the location where logging will take place
        """
        for i in range(8):
            for j in range(8):
                if i % 2 == 0:
                    if j % 2 == 0:
                        if i < 3:
                            self.board_array[i][j] = BLACK
                        elif i > 4:
                            self.board_array[i][j] = WHITE
                        else:
                            self.board_array[i][j] = 0
                else:
                    if j % 2 != 0:
                        if i < 3:
                            self.board_array[i][j] = BLACK
                        elif i > 4:
                            self.board_array[i][j] = WHITE
                        else:
                            self.board_array[i][j] = 0

        if outfile:
            self.logging = True
            self.to_file = outfile
            self.board_stack = BoardStack()

    def print_board(self):
        """
        Prints the board with associated letters and numbers for user coordinates

        :return: void
        """
        if self.game_over:
            print("       Game Over!")
        else:
            print("       " + ("White's" if self.white_move else "Black's") + " move")
        print("   ", end="")
        for _ in range(25):
            print("_", end="")
        print()
        for i in range(8):
            print(string.ascii_uppercase[i] + "  ", end="")
            for j in range(8):
                print("|" + piece_string[self.board_array[i][j]], end="")
            print("|")
        print("   ", end="")
        for _ in range(25):
            print("▔", end="")
        print()
        print("    ", end="")
        for i in range(8):
            print(str(i) + "  ", end="")
        print()

    def log(self):
        """
        Update the local to_file to contain the following: white_move, last_board, last_move, current_board, reward
        :return: void
        """
        cb = copy.deepcopy(self.board_array)
        d_board = self.board_stack.pop_board()
        while self.board_stack.size():
            next_board = self.board_stack.pop_board()

            self.to_file.write(str(self.white_move) + "\t")  # Write the current color
            self.to_file.write(np.array2string(d_board.board, threshold=100, max_line_width=np.inf,
                                               separator=',').replace('\n', '') + "\t")  # Write the last_board
            self.board_array = d_board.board
            last_reward = self.reward()
            self.board_array = next_board.board
            self.to_file.write(str(next_board.last_move) + "\t")  # Write the last move
            self.to_file.write(np.array2string(next_board.board, threshold=100, max_line_width=np.inf,
                                               separator=',').replace('\n', '') + "\t")  # Write the new board
            self.to_file.write(str(last_reward - self.reward()) + "\n")  # Write the reward change from this move
            d_board = next_board

        self.board_array = cb

    def get_single_moves(self, y: int, x: int):
        """
        Calculates all of the captures that a single piece can make
        This function is used when the player captures a piece and needs to see if they can move that piece again

        :param y: y coordinate of the piece
        :param x: x coordinate of the piece
        :return: A dictionary of the current location (x,y)->[(new_location_x, new_location_y),...]
        """
        to_return = dict()
        to_return[(y, x)] = list()

        current_color = WHITE if self.white_move else BLACK
        y_change = -current_color
        # Non-King
        if abs(self.board_array[y][x]) == 1:
            if verify_location(y + y_change * 2, x + 2) and \
                    (self.board_array[y + y_change][x + 1] == current_color * -1 or
                     self.board_array[y + y_change][x + 1] == current_color * -2) and \
                    self.board_array[y + y_change * 2][x + 2] == 0:
                to_return[(y, x)].append((y + y_change * 2, x + 2))
            if verify_location(y + y_change * 2, x - 2) and \
                    (self.board_array[y + y_change][x - 1] == current_color * -1 or
                     self.board_array[y + y_change][x - 1] == current_color * -2) and \
                    self.board_array[y + y_change * 2][x - 2] == 0:
                to_return[(y, x)].append((y + y_change * 2, x - 2))

        else:  # King
            if verify_location(y + 2, x + 2) and \
                    (self.board_array[y + 1][x + 1] == current_color * -1 or
                     self.board_array[y + 1][x + 1] == current_color * -2) and \
                    self.board_array[y + 2][x + 2] == 0:
                to_return[(y, x)].append((y + 2, x + 2))
            if verify_location(y + 2, x - 2) and \
                    (self.board_array[y + 1][x - 1] == current_color * -1 or
                     self.board_array[y + 1][x - 1] == current_color * -2) and \
                    self.board_array[y + 2][x - 2] == 0:
                to_return[(y, x)].append((y + 2, x - 2))
            if verify_location(y - 2, x + 2) and \
                    (self.board_array[y - 1][x + 1] == current_color * -1 or
                     self.board_array[y - 1][x + 1] == current_color * -2) and \
                    self.board_array[y - 2][x + 2] == 0:
                to_return[(y, x)].append((y - 2, x + 2))
            if verify_location(y - 2, x - 2) and \
                    (self.board_array[y - 1][x - 1] == current_color * -1 or
                     self.board_array[y - 1][x - 1] == current_color * -2) and \
                    self.board_array[y - 2][x - 2] == 0:
                to_return[(y, x)].append((y - 2, x - 2))
        if debug:
            self.print_board()
            print("potential next moves:", to_return)
        return to_return

    def get_possible_moves(self):
        """
        This will iterate through all of their current pieces and list actions that each piece can take.
        :return: A dictionary of the locations to destinations (start_x,start_y)->[(new_location_x, new_location_y),...]
        """
        current_color = WHITE if self.white_move else BLACK

        x, y, ky, kx = list(), list(), list(), list()

        # x and y values of the current board pieces
        non_king_moves = np.where(self.board_array == current_color)
        if len(non_king_moves) == 2:
            y = non_king_moves[0]
            x = non_king_moves[1]

        king_moves = np.where(self.board_array == current_color * 2)
        if len(non_king_moves) == 2:
            ky = king_moves[0]
            kx = king_moves[1]
        possible_moves = dict()
        # Moving Non-king
        for i in range(len(y)):
            y_change = -1 if self.white_move else 1

            # Without capture
            if verify_location(y[i] + y_change, x[i] + 1) and self.board_array[y[i] + y_change][x[i] + 1] == 0:
                add_move(possible_moves, (y[i], x[i]), (y[i] + y_change, x[i] + 1))
            if verify_location(y[i] + y_change, x[i] - 1) and self.board_array[y[i] + y_change][x[i] - 1] == 0:
                add_move(possible_moves, (y[i], x[i]), (y[i] + y_change, x[i] - 1))

            # With capture
            if verify_location(y[i] + y_change * 2, x[i] + 2) and \
                    (self.board_array[y[i] + y_change][x[i] + 1] == current_color * -1 or
                     self.board_array[y[i] + y_change][x[i] + 1] == current_color * -2) and \
                    self.board_array[y[i] + y_change * 2][x[i] + 2] == 0:
                add_move(possible_moves, (y[i], x[i]), (y[i] + y_change * 2, x[i] + 2))
            if verify_location(y[i] + y_change * 2, x[i] - 2) and \
                    (self.board_array[y[i] + y_change][x[i] - 1] == current_color * -1 or
                     self.board_array[y[i] + y_change][x[i] - 1] == current_color * -2) and \
                    self.board_array[y[i] + y_change * 2][x[i] - 2] == 0:
                add_move(possible_moves, (y[i], x[i]), (y[i] + y_change * 2, x[i] - 2))

        # Moving king
        for i in range(len(ky)):
            # Without capture
            if verify_location(ky[i] + 1, kx[i] + 1) and self.board_array[ky[i] + 1][kx[i] + 1] == 0:
                add_move(possible_moves, (ky[i], kx[i]), (ky[i] + 1, kx[i] + 1))
            if verify_location(ky[i] + 1, kx[i] - 1) and self.board_array[ky[i] + 1][kx[i] - 1] == 0:
                add_move(possible_moves, (ky[i], kx[i]), (ky[i] + 1, kx[i] - 1))
            if verify_location(ky[i] - 1, kx[i] + 1) and self.board_array[ky[i] - 1][kx[i] + 1] == 0:
                add_move(possible_moves, (ky[i], kx[i]), (ky[i] - 1, kx[i] + 1))
            if verify_location(ky[i] - 1, kx[i] - 1) and self.board_array[ky[i] - 1][kx[i] - 1] == 0:
                add_move(possible_moves, (ky[i], kx[i]), (ky[i] - 1, kx[i] - 1))

            # With capture
            if verify_location(ky[i] + 2, kx[i] + 2) and \
                    (self.board_array[ky[i] + 1][kx[i] + 1] == current_color * -1 or
                     self.board_array[ky[i] + 1][kx[i] + 1] == current_color * -2) and \
                    self.board_array[ky[i] + 2][kx[i] + 2] == 0:
                add_move(possible_moves, (ky[i], kx[i]), (ky[i] + 2, kx[i] + 2))
            if verify_location(ky[i] + 2, kx[i] - 2) and \
                    (self.board_array[ky[i] + 1][kx[i] - 1] == current_color * -1 or
                     self.board_array[ky[i] + 1][kx[i] - 1] == current_color * -2) and \
                    self.board_array[ky[i] + 2][kx[i] - 2] == 0:
                add_move(possible_moves, (ky[i], kx[i]), (ky[i] + 2, kx[i] - 2))
            if verify_location(ky[i] - 2, kx[i] + 2) and \
                    (self.board_array[ky[i] - 1][kx[i] + 1] == current_color * -1 or
                     self.board_array[ky[i] - 1][kx[i] + 1] == current_color * -2) and \
                    self.board_array[ky[i] - 2][kx[i] + 2] == 0:
                add_move(possible_moves, (ky[i], kx[i]), (ky[i] - 2, kx[i] + 2))
            if verify_location(ky[i] - 2, kx[i] - 2) and \
                    (self.board_array[ky[i] - 1][kx[i] - 1] == current_color * -1 or
                     self.board_array[ky[i] - 1][kx[i] - 1] == current_color * -2) and \
                    self.board_array[ky[i] - 2][kx[i] - 2] == 0:
                add_move(possible_moves, (ky[i], kx[i]), (ky[i] - 2, kx[i] - 2))

        # No pieces left, game over
        if len(possible_moves) == 0:
            self.endgame()

        return possible_moves

    def move(self, x: int, y: int, next_x: int, next_y: int, is_secondary: bool, auto: bool) -> bool:
        """
        Move a piece from one position to another

        :param x: current x location
        :param y: current y location
        :param next_x: next x location
        :param next_y: next y location
        :param is_secondary: is this a subsequent move
        :param auto: is this game being played automatically or by a user
        :return:
        """
        # Greatly slows down program but makes board logging possible, add current_board to the stack
        if self.logging:
            self.board_stack.push_board(copy.deepcopy(self.board_array), ((x, y), (next_x, next_y)))

        if is_secondary:
            possible_moves = self.get_single_moves(x, y)
        else:
            possible_moves = self.get_possible_moves()

        if self.game_over:
            return False

        # Cause a draw if there is no moves left for this player to make
        if len(possible_moves) == 0:
            self.game_over = True

        if (x, y) in possible_moves and (next_x, next_y) in possible_moves[(x, y)]:
            self.last_move = ((x, y), (next_x, next_y))
            current_value = self.board_array[x][y]
            self.board_array[x][y] = 0

            # Making kings when you get to the other side
            if current_value == WHITE and self.white_move and next_x == 0:
                current_value *= 2
            elif current_value == BLACK and not self.white_move and next_x == 7:
                current_value *= 2

            self.board_array[next_x][next_y] = current_value

            # Capture
            if abs(x - next_x) > 1:
                if debug:
                    print("Capture")
                self.board_array[int((x + next_x) / 2)][int((y + next_y) / 2)] = 0

                # only check secondary if move was made using random/onestep
                check_secondary = \
                    (self.white_move and (self.white_policy == Policy.RANDOM or self.white_policy == Policy.LOOKAHEAD)) \
                    or (not self.white_move and self.black_policy == Policy.RANDOM or self.black_policy == Policy.LOOKAHEAD)

                if check_secondary:
                    next_moves = self.get_single_moves(next_x, next_y)
                    self.play(auto, next_moves, True)
                # if self.logging and not is_secondary:
                #     self.board_stack.push_board(copy.deepcopy(self.board_array), None)
            if self.logging and not is_secondary:
                self.board_stack.push_board(copy.deepcopy(self.board_array), None)

            if self.total_moves > MOVE_LIMIT:
                self.game_over = True
            self.total_moves += 1
            return True
        return False

    def play(self, auto: bool, actions, secondary: bool):
        """
        This is the brain of the game, it will follow a policy or allow the user to input the moves

        :param auto: is this game being played automatically or by a user
        :param actions: a dictionary of moves from starting location to ending location
        :param secondary: is this a subsequent move
        :return:
        """
        # if self.logging:
        #     self.board_stack.push_board(copy.deepcopy(self.board_array), ((0, 0), (0, 0)))
        if auto:
            # Tie if there are no moves left
            if len(actions.keys()) == 0:
                self.endgame()
                return False

            if self.white_move:
                if self.white_policy == Policy.RANDOM:
                    self.random_policy(actions, secondary)
                elif self.white_policy == Policy.LOOKAHEAD:
                    self.lookahead_policy(actions, secondary, self.white_depth)
                elif self.white_policy == Policy.ROLLOUTS:
                    self.rollouts_policy(actions, secondary)

            else:
                if self.black_policy == Policy.RANDOM:
                    self.random_policy(actions, secondary)
                elif self.black_policy == Policy.LOOKAHEAD:
                    self.lookahead_policy(actions, secondary, self.black_depth)
                elif self.black_policy == Policy.ROLLOUTS:
                    self.rollouts_policy(actions, secondary)

        else:
            self.print_board()
            print(f"Moves: {actions}")
            if secondary:
                if len(actions) != 1 or len(actions[random.choice(list(actions.keys()))]) != 0:
                    inp = input("Capture! Enter next move (a0,b1) or 'P' to pass: ")
                    if "P" in inp.upper():
                        if debug:
                            print("Passing")
                    else:
                        current_piece, attempted_move = inp.split(",")
                        current_piece = view_coordinates_board_coordinates(current_piece)
                        attempted_move = view_coordinates_board_coordinates(attempted_move)
                        self.move(current_piece[0], current_piece[1], attempted_move[0], attempted_move[1], True, False)
            else:
                board_coordinates_old, board_coordinates_new = input("Enter move (a0,b1): ").split(",")
                board_coordinates_old = view_coordinates_board_coordinates(board_coordinates_old)
                board_coordinates_new = view_coordinates_board_coordinates(board_coordinates_new)

                # If input is not valid, try again
                if board_coordinates_old not in actions.keys() or \
                        board_coordinates_new not in actions[board_coordinates_old]:
                    print("Invalid move, try again.")
                    self.play(False, actions, False)

                self.move(board_coordinates_old[0], board_coordinates_old[1],
                          board_coordinates_new[0], board_coordinates_new[1], secondary,
                          False)

        # If this is not a secondary move, flip whose turn it is (this happens after all secondary moves are made)
        if not secondary:
            self.white_move = not self.white_move  # Flip whose turn it is

            # Log all the moves that were made this turn
            if self.logging:
                if self.last_move != ((0, 0), (0, 0)):
                    self.log()
                if debug:
                    print(self.board_stack.size())
                    # if self.board_stack.size() > 1:
                    #     print(self.board_stack, "\n\n")
        return True

    def endgame(self):
        """
        Prints out who won the game

        :return: void
        """
        if self.game_over:
            return
        if np.any(self.board_array < 0) and not np.any(self.board_array > 0):
            print("Black wins!")
        elif np.any(self.board_array > 0) and not np.any(self.board_array < 0):
            print("White wins!")
            self.white_wins = True
        else:
            print("Draw!")
            self.draw = True

        self.game_over = True

    def reward(self) -> int:
        """
        Prints the current reward of the pieces
        :return: a sum of all the values in the board_array
        """
        return sum(sum(self.board_array))

    def random_policy(self, actions, secondary):
        """
        Takes a random move from a random piece

        Policy number: 0

        :param actions: a dictionary of moves from starting location to ending location
        :param secondary: is this a subsequent move
        :return: void
        """
        random_piece = random.choice(list(actions.keys()))
        if len(actions[random_piece]) != 0:
            piece_move = random.choice(actions[random_piece])
            if secondary:
                if random.random() > self.stop_probability:
                    self.move(random_piece[0], random_piece[1], piece_move[0], piece_move[1], True, True)
            else:
                self.move(random_piece[0], random_piece[1], piece_move[0], piece_move[1], secondary, True)

    def lookahead_policy(self, actions, secondary, depth):
        """
        Calculates the reward from each piece and each action then takes the greatest rewards move.

        :param depth: the number of lookahead steps
        :param actions: a dictionary of moves from starting location to ending location
        :param secondary: is this a subsequent move
        :return: void
        """
        rewards = dict()
        c_board = copy.deepcopy(self.board_array)

        for piece in actions.keys():
            for move in actions[piece]:
                temp = Board()
                temp.board_array = copy.deepcopy(self.board_array)
                temp.white_move = self.white_move
                temp.white_depth = self.white_depth
                temp.black_depth = self.black_depth
                # Make each move
                temp.move(piece[0], piece[1], move[0], move[1], secondary, True)
                # Flip the turn
                temp.white_move = not temp.white_move
                temp.print_board()

                if depth != 1:
                    temp.white_policy = self.white_policy
                    temp.black_policy = self.black_policy
                    if temp.white_move:
                        temp.black_depth = depth - 1
                    else:
                        temp.white_depth = depth - 1

                    # Opponents moves
                    t_moves = temp.get_possible_moves()
                    t_board = copy.deepcopy(temp.board_array)
                    for p in t_moves.keys():
                        for m in t_moves[p]:
                            ntemp = Board()
                            ntemp.board_array = copy.deepcopy(temp.board_array)
                            ntemp.white_move = temp.white_move
                            ntemp.move(p[0], p[1], m[0], m[1], False, True)
                            temp.play(True, ntemp.get_possible_moves(), False)
                    temp.board_array = t_board

                add_move(rewards, temp.reward(), (piece, move))
        self.board_array = c_board

        if len(rewards.keys()) == 0:
            return

        if self.white_move:
            best = max(rewards.keys())
            next_move = random.choice(rewards[best])
            self.move(next_move[0][0], next_move[0][1], next_move[1][0], next_move[1][1],
                      secondary, True)
        else:
            best = min(rewards.keys())
            next_move = random.choice(rewards[best])
            self.move(next_move[0][0], next_move[0][1], next_move[1][0], next_move[1][1],
                      secondary, True)
    def policy_gradient():
        """
        Attempting to implement Algorithm 4.1 in Lecture 10 notes.

        Inputs: 
        A stochastic dynamic program
        A feature function phi
        Monte-Carlo sample size M (If time allows we will try multiple)
        Step sizes \alpha_0,... 
        Gradient bound \delta

        Output: Parameter \theta \in R^K, used to construct the randomized policy

        """ 

        #initialize \theta^0 arbitrarily
        theta = 0
        while 



    def rollouts_policy(self, actions, secondary):
        """
        Uses one step lookahead + simulates remainder of game using heuristic policy to choose action.
        The opposing side's moves are simulated using a random policy.
        The heuristic policy is to move the piece closest to home side in a random direction.

        :param actions: a dictionary of moves from starting location to ending location
        :param secondary: is this a subsequent move
        :return: void
        """
        # M = number of times to simulate rollout
        M = 1
        # create dict of first move to total rewards where total = one_step_lookahead + rollout_sim
        # then take action with max total reward
        first_move_to_total_reward: Dict[((int, int), (int, int)), float] = dict()
        # first move to board used to simulate entire games and find rollout return
        first_move_to_board: Dict[((int, int), (int, int)), Board] = dict()

        c_board = copy.deepcopy(self.board_array)
        # TODO generalize turn bool updates so policy can be used by white or black
        # first calculate one_step_lookahead rewards
        for (y1, x1), moves_list in actions.items():
            for (y2, x2) in moves_list:
                b = Board()
                b.board_array = copy.deepcopy(self.board_array)
                b.white_move = self.white_move
                b.black_policy = Policy.RANDOM
                b.white_policy = Policy.ROLLOUTS

                last_reward = b.reward()
                b.move(y1, x1, y2, x2, secondary, True)
                b.white_move = False
                first_move_to_board[((y1, x1), (y2, x2))] = b
                first_move_to_total_reward[((y1, x1), (y2, x2))] = (last_reward - b.reward())

        # now loop over each board and simulate to endgame, add rewards to total
        for ((y1, x1), (y2, x2)), b in first_move_to_board.items():
            print('---------------------------simulating next board')
            total_sim_rewards: float = 0
            # M = number of times to sim rollout
            for _ in range(M):
                while not b.game_over:
                    print(f'before black moves randomly')
                    b.print_board()
                    sim_actions = b.get_possible_moves()
                    last_reward = b.reward()
                    # black makes random move
                    b.random_policy(sim_actions, False)
                    b.white_move = True

                    total_sim_rewards += (last_reward - b.reward())
                    last_reward = b.reward()

                    # white moves piece closest to home side randomly
                    # find closest home piece
                    for y in reversed(range(8)):  # white home is bottom of board
                        moved: bool = False
                        for x in range(8):
                            piece = b.board_array[y][x]
                            if piece == WHITE:  # TODO change to work for current color, not only white
                                # TODO inefficient because calculating all moves when we only care about one piece
                                # default value -1 if (y,x) not in possible moves
                                single_moves = b.get_possible_moves().get((y, x), -1)
                                if single_moves == -1:
                                    continue
                                # if piece can move, move it
                                # single moves always has (starty, startx)->[], need to check if the list is empty
                                if len(single_moves) != 0:
                                    (heuristic_y2, heuristic_x2) = random.choice(single_moves)
                                    print(f'white moving home piece')
                                    b.print_board()
                                    if b.move(y, x, heuristic_y2, heuristic_x2, False, True):
                                        print('white moved home piece')
                                    else:
                                        print('white failed to move home piece')
                                    b.white_move = False
                                    moved = True
                                    total_sim_rewards += (last_reward - b.reward())
                                    last_reward = b.reward()
                                    break
                        if moved:  # make just one heuristic move and go back to black move
                            break
            # divide by num simulations to get avg reward
            avg_sim_reward: float = total_sim_rewards / M
            # add sim reward to one step lookahead reward
            first_move_to_total_reward[((y1, x1), (y2, x2))] += avg_sim_reward

        # make optimal rollout move on current board
        self.board_array = c_board
        # find moves with max total reward
        print(f'board before real move')
        self.print_board()
        if self.white_move:
            # find key(first move) with highest value(reward)
            best = max(first_move_to_total_reward, key=first_move_to_total_reward.get)
        else:
            best = min(first_move_to_total_reward, key=first_move_to_total_reward.get)

        self.move(best[0][0], best[0][1], best[1][0], best[1][1], secondary, True)


    #phi = lambda ks, ko, rs, ro, vs, vo:[2*ks, -2*ko, rs, ro, -3*vrs, 3*vro, -5*vks, 5*vko] #defining our feature function for policy gradient
    #ks = 0  # of kings that we (s for self) have
    #ko = 0  # of kings the opponent has
    #rs = 12 # of regular pieces we have
    #ro = 12 # of regular pieces the opponent has
    #vrs = 0  # of our pieces that are vulnerable (opponent could take on their next turn)
    #vro = 0  # of the opponent's pieces that are vulnerable (we could take on this turn)
    #vks = 0  # of our pieces that are vulnerable (opponent could take on their next turn)
    #vko = 0  # of the opponent's pieces that are vulnerable (we could take on this turn)

    def phi(self, s, a):
        '''
        Our phi is now complex enough
        that we can't just make it a lambda...

        s: current state (the board (self.board))
        a: an action to take--what will the effects of this action be on the state?

        Calculate the phi of the state s of the board AFTER taking this action!
        '''
        #counts the number of each type of pieces on the board 
        #(i.e. how many of our regular pieces, how how many of our kings, etc.)
        #count_arr = np.bincount(self.board)
        #ks = count_arr[2]
        #ko = count_arr[-2]
        #rs = count_arr[1]
        #ro = count_arr[-2]

        #a: an action to take--what will the effects of this action be on the state?
        #action is a tuple containing a piece's x and y and a move's x and y

        #For now let's assume we will only apply policy gradient to white
        temp = Board()
        temp.board_array = copy.deepcopy(self.board_array)
        temp.white_move = self.white_move

        # a[0] = x
        # a[1] = y
        # a[2] = new_x
        # a[3] = new_y

         
        # Needs to not be a secondary move.
        temp.move(a[0], a[1], a[2], a[3], False, True)
 
        white_moves = get_possible_moves()
        temp.white_move = False
        black_moves = get_possible_moves()
        temp.white_move = True
        
        vrs = 0
        vro = 0
        vks = 0
        vko = 0
        
        for w in white_moves:
            if abs(x - next_x) > 1:
            #need to do this


        # Capture
       

        phi_result [2*np.bincount(s)[2], #ks
                        -2*np.bincount(s)[-2], #ko
                            1 * np.bincount(s)[1], #rs
                            -2 * np.bincount(s)[-1], #ro
                            -3*vrs,
                            3*vro, 
                            -5*vks, 
                            5*vko]
        


    def pi_theta_policy_gradient(self, actions, current_action, theta, secondary):

    """
    A softmax policy that takes in weights "theta" from policy gradient

    Policy number: 10

    :param actions: a dictionary of moves from starting location to ending location
    :param current_action: the action being passed into pi_theta
    :param secondary: is this a subsequent move
    :param theta: weights theta for policy gradient

    :return: the "probability" of this action being the best action to take
    """
    rewards = dict()

  
    e_cur_a = exp((phi(s,a).T) @ theta)

    # Try to make one move for all your pieces (whites)
    # And save the reward(s)
    for piece in actions.keys():
        for move in actions[piece]:
            temp = Board()
            temp.board_array = copy.deepcopy(self.board_array)
            temp.white_move = self.white_move
            temp.move(piece[0], piece[1], move[0], move[1], secondary, True)
            add_move(rewards, temp.reward(), (piece, move))
    self.board_array = c_board

    # This seems to be the case that handles if no white moves are possible
    if len(rewards.keys()) == 0:
        return

    # Calculates white reward?
    if self.white_move:
        best = max(rewards.keys())
        next_move = random.choice(rewards[best])
        self.move(next_move[0][0], next_move[0][1], next_move[1][0], next_move[1][1],
                    secondary, True)

    # Calculates black reward?
    else:
        best = min(rewards.keys())
        next_move = random.choice(rewards[best])
        self.move(next_move[0][0], next_move[0][1], next_move[1][0], next_move[1][1],
                    secondary, True)



def get_next_csv_number() -> int:
    """
    Return the next csv file number; use this to get the next file number to ensure that nothing is overwritten
    :return:
    """
    # ---- Finding the current file name ----
    current_file_path = os.path.dirname(__file__)

    # Getting all the last games in specified folder
    last_file_name = glob.glob(current_file_path + "/game_records/*.tsv")
    next_file_name = 0

    if last_file_name:
        for index, item in enumerate(last_file_name):
            file_number = int(item[item.rfind("/") + 1: -4])
            if next_file_name <= file_number:
                next_file_name = file_number + 1
    else:
        next_file_name = 0

    return next_file_name


if __name__ == "__main__":
    draws = 0
    black_wins = 0
    white_wins = 0
    for _ in range(1):
        with open("./game_records/" + str(get_next_csv_number()) + ".tsv", "w") as to_file:
            current_board = Board()
            current_board.white_move = True

            current_board.black_policy = Policy.RANDOM
            current_board.white_policy = Policy.ROLLOUTS
            current_board.black_depth = 1
            current_board.white_depth = 2
            still_playing: bool = True

            while still_playing:
                #  White
                moves = current_board.get_possible_moves()
                still_playing = current_board.play(True, moves, False)

                if not still_playing:
                    break

                #  Black
                moves = current_board.get_possible_moves()
                still_playing = current_board.play(True, current_board.get_possible_moves(), False)

            if current_board.draw:
                draws += 1
            elif current_board.white_wins:
                white_wins += 1
            else:
                black_wins += 1
            # current_board.print_board()

    print("Draws:", draws)
    print("White wins:", white_wins)
    print("Black wins:", black_wins)
