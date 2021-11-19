# File Management
import glob
import os

# Variable assistance
import random
import string

# Board manipulation
import numpy as np
import copy

# Constants for pieces
BLACK = -1
WHITE = 1

debug = False
logging = False

# Array values to string for printing
piece_string = {
    2: "WK",
    1: "W ",
    0: "  ",
    -1: "B ",
    -2: "BK",
}


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
    board_array = np.zeros((8, 8), dtype=int)
    last_board_array = np.zeros((8,8), dtype=int)
    white_move: bool = True
    game_over: bool = False
    white_wins: bool = False
    draw: bool = False
    stop_probability: float = 0
    white_policy: int = 1
    black_policy: int = 0
    to_file = None
    last_move = ((0, 0), (0, 0))

    # Setup the board in the default configuration
    def __init__(self, outfile = None):
        if logging:
            self.to_file = outfile
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

    def log(self):
        to_file.write(np.array2string(current_board.last_board_array, threshold=100, max_line_width=np.inf,
                                      separator=',').replace('\n', '') + "\t")  # Write the last_board
        last_reward = current_board.reward()
        to_file.write(str(current_board.last_move) + "\t")  # Write the last move
        to_file.write(np.array2string(current_board.board_array, threshold=100, max_line_width=np.inf,
                                      separator=',').replace('\n', '') + "\t")  # Write the new board
        to_file.write(str(last_reward - current_board.reward()) + "\n")  # Write the reward change from this move

    def get_single_moves(self, y: int, x: int):
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
        # Greatly slows down program but makes single statement logging possible
        if logging:
            self.last_board_array = copy.deepcopy(self.board_array)

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
                next_moves = self.get_single_moves(next_x, next_y)
                if logging:
                    self.log()
                self.play(auto, next_moves, True)
            elif logging:
                    self.log()

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
        if auto:
            # Tie if there are no moves left
            if len(actions.keys()) == 0:
                self.endgame()
                return False

            if self.white_move:
                if self.white_policy == 0:
                    self.random_policy(actions, secondary)
                elif self.white_policy == 1:
                    self.one_step_lookahead_policy(actions, secondary)
            else:
                if self.black_policy == 0:
                    self.random_policy(actions, secondary)
                elif self.black_policy == 1:
                    self.one_step_lookahead_policy(actions, secondary)
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

    def one_step_lookahead_policy(self, actions, secondary):
        """
        Calculates the reward from each piece and each action then takes the greatest rewards move.

        Policy number: 1

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
                temp.move(piece[0], piece[1], move[0], move[1], secondary, True)
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


def get_next_csv_number() -> int:
    # ---- Finding the last file name ----
    current_file_path = os.path.dirname(__file__)

    # Getting the names of all isolated shots
    last_file_name = glob.glob(current_file_path + "/game_records/*.tsv")
    next_file_name = 0
    # Finding the number of the last isolated shot
    isolated_shots = []
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
    y = np.array([[-1, 0, 2, 0, 0, 0,-1, 0], [ 0, 0, 0,-1, 0,-1, 0,-1], [ 0, 0, 0, 0,-1, 0,-1, 0], [ 0,-1, 0,-1, 0, 0, 0,-1], [ 0, 0, 0, 0, 1, 0, 1, 0], [ 0, 1, 0, 1, 0, 1, 0, 0], [ 1, 0, 0, 0, 1, 0, 1, 0], [ 0, 1, 0, 1, 0, 0, 0, 1]])
    x = np.array([[-1, 0, 2, 0, 2, 0,-1, 0], [ 0, 0, 0, 0, 0,-1, 0,-1], [ 0, 0, 0, 0,-1, 0,-1, 0], [ 0,-1, 0, 0, 0, 0, 0,-1], [ 0, 0, 0, 0, 0, 0, 1, 0], [ 0, 1, 0, 1, 0, 1, 0, 0], [ 1, 0, 0, 0, 1, 0, 1, 0], [ 0, 1, 0, 1, 0, 0, 0, 1]])
    for _ in range(10):
        current_board = Board()
        current_board.board_array = x
        current_board.print_board()
        current_board.board_array = y
        current_board.print_board()
        break
        current_board.black_policy = 1
        current_board.white_policy = 1
        still_playing: bool = True
        with open("./game_records/" + str(get_next_csv_number()) + ".tsv", "w") as to_file:
            while still_playing:
                #  White
                to_file.write(np.array2string(current_board.board_array, threshold=100, max_line_width=np.inf,
                                              separator=',').replace('\n', '') + "\t")  # Write the current board
                old_reward = current_board.reward()

                #  Playing
                moves = current_board.get_possible_moves()
                still_playing = current_board.play(True, moves, False)
                #  Recording
                to_file.write(str(current_board.last_move) + "\t")  # Write the last move
                to_file.write(np.array2string(current_board.board_array, threshold=100, max_line_width=np.inf,
                                              separator=',').replace('\n', '') + "\t")  # Write the new board
                to_file.write(str(old_reward - current_board.reward()) + "\n")  # Write the reward change from this move

                if not still_playing:
                    break

                #  Black
                old_reward = current_board.reward()
                to_file.write(np.array2string(current_board.board_array, threshold=100, max_line_width=np.inf,
                                              separator=',').replace('\n', '') + "\t")  # Write the current board

                #  Playing
                moves = current_board.get_possible_moves()
                still_playing = current_board.play(True, current_board.get_possible_moves(), False)

                #  Recording
                to_file.write(str(current_board.last_move)+ "\t")  # Write the last move
                to_file.write(np.array2string(current_board.board_array, threshold=100, max_line_width=np.inf,
                                              separator=',').replace('\n', '') + "\t")  # Write the new board
                to_file.write(str(old_reward - current_board.reward()) + "\n")  # Write the reward change from this move

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
