
import numpy as np
import copy

delta = 0.009 #once we get code running, we'll let it go for a while, see what it converges to, and set delta accordingly
alpha = #step sizes
theta =

i = 0
#start with 100 trajectories
M = 100
#pi_theta from lecture 10
#It is an approximation of the greedy policy with respect to q_bar
#It's always going to take the greedy action wit probability 1

def pi_theta_policy_gradient(self, actions, secondary):
    """
    Takes a random move from a random piece

    Policy number: 0

    :param actions: a dictionary of moves from starting location to ending location
    :param secondary: is this a subsequent move
    :return: the move to take
    """
    rewards = dict()
    c_board = copy.deepcopy(self.board_array)

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

def softmax_policy(self, actions, secondary, theta):
    """
    A softmax policy that takes in weights "theta" from policy gradient
    

    Policy number: 0

    :param actions: a dictionary of moves from starting location to ending location
    :param secondary: is this a subsequent move
    :return: the move to take
    """
    rewards = dict()
    c_board = copy.deepcopy(self.board_array)

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

    
def follow_pi_theta(self, actions, secondary, theta):
    """
    Follows policy pi_theta (a softmax policy)
    from the current state to the end of the game
    and returns the reward z
    """

    c_board = copy.deepcopy(self.board_array)

    c_board.white_move = False

    c_board.black_policy = 0
    c_board.white_policy = 5

    still_playing: bool = True

    while still_playing:
        #  Get white moves
        moves = c_board.get_possible_moves()
        # get count of possible white moves
        max_prob = 0
        max_prob_move = ((0, 0), (0, 0))

        for key, value in moves.items():
            for m in value:
                cur_prob = c_board.pi_theta_policy_gradient(moves, m, theta)
                if cur_prob > max_prob:
                    max_prob = cur_prob
                    max_prob_move = m
        c_board.move(max_prob_move[0][0], max_prob_move[0][1])
        # check if still playing
        if np.any(c_board.board_array < 0) and not np.any(c_board.board_array > 0):
            #Black win
            still_playing = False
        elif np.any(c_board.board_array > 0) and not np.any(c_board.board_array < 0):
            #White win
            c_board.white_wins = True
            still_playing = False
        else:
            #Draw
            c_board.draw = True
            still_playing = False

        # Black's turn
        c_board.white_move = False
        moves = c_board.get_possible_moves()
        # random policy-- make a random move
        random_piece = random.choice(list(moves.keys()))
        if len(moves[random_piece]) != 0:
            piece_move = random.choice(moves[random_piece])
            if secondary:
                if random.random() > c_board.stop_probability:
                    c_board.move(random_piece[0], random_piece[1], piece_move[0], piece_move[1], True, True)
            else:
                c_board.move(random_piece[0], random_piece[1], piece_move[0], piece_move[1], secondary, True)


        
        # check if still playing
        if np.any(c_board.board_array < 0) and not np.any(c_board.board_array > 0):
            #Black win
            still_playing = False
        elif np.any(c_board.board_array > 0) and not np.any(c_board.board_array < 0):
            #White win
            c_board.white_wins = True
            still_playing = False
        else:
            #Draw
            c_board.draw = True
            still_playing = False 

    if c_board.draw:
        return 0
    elif c_board.white_wins:
        return 1
    else:
        return -1
    # current_board.print_board()
    

phi = lambda ks, ko, rs, ro, vs, vo:[2*ks, -2*ko, rs, ro, -3*vrs, 3*vro, -5*vks, 5*vko] #defining our feature function

#These are the variables used in the feature function
ks = 0  # of kings that we (s for self) have
ko = 0  # of kings the opponent has
rs = 12 # of regular pieces we have
ro = 12 # of regular pieces the opponent has
vrs = 0  # of our pieces that are vulnerable (opponent could take on their next turn)
vro = 0  # of the opponent's pieces that are vulnerable (we could take on this turn)
vks = 0  # of our pieces that are vulnerable (opponent could take on their next turn)
vko = 0  # of the opponent's pieces that are vulnerable (we could take on this turn)
#check to see if we can make a king --> ADD A NEW FEATURE FOR THIS!!!!

def calculate_phi_vars(board): #take the current state (board) and calculate and set our vars
    count_arr = np.bincount(board) #counts the number of each type of pieces on the board (i.e. how many of our regular pieces, how how many of our kings, etc.)

    ks = count_arr[2]
    ko = count_arr[-2]
    rs = count_arr[1]
    ro = count_arr[-2]

    #maybe use get_single_moves() or get_possible_moves()
    #we're trying to see if we can make captures or make pieces kings (or if our opponent can)
        #check 426 for info on captures and 420-423 for kings

    #line 428 in code may also be helpful: take abs of x or y from old x location to new x location
    vrs =
    vro =
    vks =
    vko =


#once we get code running, we'll let it go for a while, see what it converges to, and set delta accordingly
j = np.zeros(M)
#note that d[i] = d[0]
while np.linalg.norm(d[i]) >= delta: 
    #Sample M trajectories, following pi_theta_i
    for trajectories in j:
        follow_pi_theta
    #compute the gradient
    grad_P = phi(s, a) - pi_theta(s,b) * phi(s,b)
    #calculate d[i]
    d[i] = -1/M * ()
    #calculate theta[i]
    theta[i+1] = theta[i] + (alpha[i] * d[i])
    #increase i by 1
    i = i + 1