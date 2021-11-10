import copy

import numpy as np
import string
import random

# Constants for pieces
BLACK = -1
WHITE = 1

debug = False

# Array values to string for printing
piece_string = {
    2: "WK",
    1: "W ",
    0: "  ",
    -1: "B ",
    -2: "BK",
}


def view_coords_board_coords(input_coords: str) -> (int, int):
    """
    Converts user entered coordinates (comma seperated) into two ints

    :param input_coords: user input in the following form: 0a,1b or a0,1b
    :return: a array coordinate equivalent
    """
    input_coords = input_coords.upper().strip().replace(" ", "")
    if input_coords[0] in string.ascii_uppercase:
        x = string.ascii_uppercase.find(input_coords[0])
        y = int(input_coords[1])
    else:
        x = string.ascii_uppercase.find(input_coords[1])
        y = int(input_coords[0])
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
    board_array = np.zeros((8, 8), dtype=int)
    white_move: bool = True
    game_over: bool = False
    white_wins: bool = False
    draw: bool = False
    stop_probability: float = 0

    # Setup the board in the default configuration
    def __init__(self):
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

    def get_single_moves(self, y: int, x: int) -> dict[(int, int):list[(int, int)]]:
        """
        Calculates all of the moves(captures) that a single piece can make
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
        if abs(self.board_array[x][y]) == 1:
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
            print("potential next moves:", to_return)
        return to_return

    def get_possible_moves(self) -> dict[(int, int):list[(int, int)]]:
        """
        This will iterate through all of their current pieces and list actions that each piece can take.
        :return: A dictionary of the locations to destinations (start_x,start_y)->[(new_location_x, new_location_y),...]
        """
        current_color = WHITE if self.white_move else BLACK

        x, y, ky, kx = list(), list(), list(), list()

        # x and y values of the current board pieces
        nonKingMoves = np.where(self.board_array == current_color)
        if len(nonKingMoves) == 2:
            y = nonKingMoves[0]
            x = nonKingMoves[1]

        kingMoves = np.where(self.board_array == current_color * 2)
        if len(nonKingMoves) == 2:
            ky = kingMoves[0]
            kx = kingMoves[1]
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

        if len(possible_moves) == 0:
            self.endGame()

        return possible_moves

    def move(self, x: int, y: int, next_x: int, next_y: int, is_secondary: bool, auto: bool) -> bool:
        """

        :param x: current x location
        :param y: current y location
        :param next_x: next x location
        :param next_y: next y location
        :param is_secondary: is this a subsequent move
        :param auto: is this game being played automatically or by a user
        :return:
        """
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

                next_moves = self.get_single_moves(next_x, next_y)
                self.play(auto, next_moves, True)

            return True
        return False

    def play(self, auto: bool, actions: dict[(int, int):list[(int, int)]], secondary: bool):
        """
        This is the brain of the game, it will follow a policy or allow the user to input the moves

        :param auto: is this game being played automatically or by a user
        :param actions: a dictionary of moves from starting location to ending location
        :param secondary: is this a subsequent move
        :return:
        """
        if auto:
            if len(actions.keys()) == 0:
                self.endGame()
                return False
            if self.white_move:
                self.one_step_lookahead_policy(actions, secondary)
            else:
                self.random_policy(actions, secondary)
        else:
            if secondary:
                if len(actions) != 1 or len(actions[random.choice(list(actions.keys()))]) != 0:
                    inp = input("Capture! Enter next move (a0,b1) or 'P' to pass: ")
                    if "P" in inp.upper():
                        if debug:
                            print("Passing")
                    else:
                        current_piece, attempted_move = inp.split(",")
                        current_piece = view_coords_board_coords(current_piece)
                        attempted_move = view_coords_board_coords(attempted_move)
                        self.move(current_piece[0], current_piece[1], attempted_move[0], attempted_move[1], True, False)
            else:
                board_coords_old, board_coords_new = input("Enter move (a0,b1): ").split(",")
                board_coords_old = view_coords_board_coords(board_coords_old)
                board_coords_new = view_coords_board_coords(board_coords_new)

                # If input is not valid, try again
                if board_coords_old not in actions.keys() or board_coords_new not in actions[board_coords_old]:
                    print("Invalid move, try again.")
                    self.play(False, actions, False)

                self.move(board_coords_old[0], board_coords_old[1], board_coords_new[0], board_coords_new[1], secondary,
                          False)

        # If this is not a secondary move, flip whose turn it is (this happens after all seconary moves are made)
        if not secondary:
            self.white_move = not self.white_move  # Flip whose turn it is
        return True

    def endGame(self):
        """
        Prints out who won the game

        :return: void
        """
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

    def random_policy(self, actions: dict[(int, int):list[(int, int)]], secondary):
        """
        Takes a random move from a random piece

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

    def one_step_lookahead_policy(self, actions: dict[(int, int):list[(int, int)]], secondary):
        """
        Calculates the reward from each piece and each action then takes the greatest rewards move.

        :param actions: a dictionary of moves from starting location to ending location
        :param secondary: is this a subsequent move
        :return: void
        """
        rewards: dict[int:list[((int, int), (int, int))]] = dict()
        c_board = copy.deepcopy(self.board_array)
        for piece in actions.keys():
            for move in actions[piece]:
                temp = Board()
                temp.board_array = copy.deepcopy(self.board_array)
                temp.move(piece[0], piece[1], move[0], move[1], False, True)
                add_move(rewards, temp.reward(), (piece, move))
        self.board_array = c_board

        if len(rewards.keys()) == 0:
            return
        if self.white_move:
            best = max(rewards.keys())
            self.move(rewards[best][0][0][0], rewards[best][0][0][1], rewards[best][0][1][0], rewards[best][0][1][1],
                      secondary, True)
        else:
            best = min(rewards.keys())
            self.move(rewards[best][0][0][0], rewards[best][0][0][1], rewards[best][0][1][0], rewards[best][0][1][1],
                      secondary, True)


if __name__ == "__main__":
    draws = 0
    black_wins = 0
    white_wins = 0
    for _ in range(1000):
        current_board = Board()
        still_playing: bool = True
        while still_playing:
            #  White
            # current_board.print_board()
            moves = current_board.get_possible_moves()
            # print("Moves:", moves)
            still_playing = current_board.play(True, moves, False)

            if not still_playing:
                break
            #  Black
            # current_board.print_board()
            moves = current_board.get_possible_moves()
            # print("Moves:", moves)
            still_playing = current_board.play(True, current_board.get_possible_moves(), False)

        if current_board.draw:
            draws += 1
        elif current_board.white_wins:
            white_wins += 1
        else:
            black_wins += 1
        current_board.print_board()

    print("Draws:", draws)
    print("White wins:", white_wins)
    print("Black wins:", black_wins)
