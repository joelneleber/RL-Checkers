import glob
import os
import numpy as np
from ast import literal_eval

gamma = 0.9
NUM_FILES = 100

current_file_path = os.path.dirname(__file__)
last_file_name = glob.glob(current_file_path + "\\game_records\\*.tsv")


def count_lines():
    lines = 0
    for j in range(NUM_FILES):
        with open(str(last_file_name[j]), "r") as to_files:
            for _ in to_files.readlines():
                lines += 1
    return lines


num_lines = count_lines()


Data = {'state': np.full((num_lines, 8, 8), np.array([0, 0, 0, 0, 0, 0, 0, 0])),
        'stateprime': np.full((num_lines, 8, 8), np.array([0, 0, 0, 0, 0, 0, 0, 0])),
        'reward': np.full(num_lines, -1000),
        }


def print_data():
    for j in range(num_lines):
        print('============================================================')
        print('State ---- >', Data['state'][j])
        print('StatePrime ---- >', Data['stateprime'][j])
        print('Reward ---- >', Data['reward'][j])


phi = lambda s: [1, s, s ** 2, s ** 3, s ** 4, s ** 5, s ** 6, s ** 7]
Phi = np.array([phi(s) for s in range(8)])

K = np.size(Phi, 0)
F = np.zeros((K, K))
H = np.zeros((K, K))
h = np.full((K, K), 1)

L = 0

prev = 0
for i in range(NUM_FILES):
    with open(str(last_file_name[i]), "r") as to_file:
        for line in to_file.readlines():
            cols = line.split('\t')
            Data['state'][L] = literal_eval(cols[3])
            Data['stateprime'][L] = literal_eval(cols[1])
            Data['reward'][L] = int(cols[4])
            L += 1

for t in range(num_lines):
    state = Data['state'][t]
    statePrime = Data['stateprime'][t]
    F += np.matmul(Phi * state, Phi * state.T)
    H += np.matmul(Phi * state, Phi * statePrime.T)
    h += np.dot((Data['reward'][t]), Phi * state)


print((F - gamma * H) / h)


def print_other():
    print('=-==============F IS HERE=====================')
    print(F)
    print('=-==============H IS HERE=====================')
    print(H)
    print('=-===============little h is here====================')
    print(h)


#print_other()
