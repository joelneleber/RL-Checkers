
delta = 0.009 #once we get code running, we'll let it go for a while, see what it converges to, and set delta accordingly
alpha = #step sizes
theta =

i = 0

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

def calculate_phi_vars(board): #take the current state (board) and calculate and set our vars
count_arr = np.bincount(board)

ks = count_arr[2]
ko = count_arr[-2]
rs = count_arr[1]
ro = count_arr[-2]

vrs =
vro =
vks =
vko =

#once we get code running, we'll let it go for a while, see what it converges to, and set delta accordingly
while >= delta:  #to pick our
    grad_P = phi(s, a) - pi_theta(s,b) * phi(s,b)
    d[i] = -1/M * ()
    theta[i+1] = theta[i] + alpha[i] * d[i]