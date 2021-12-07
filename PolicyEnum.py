from enum import Enum, unique


@unique
class Policy(Enum):
    RANDOM = 0
    ONE_STEP_LOOKAHEAD = 1
    ROLLOUTS = 2
    LSTD = 3
    POLICY_GRADIENT = 4
    MULTI_STEP_LOOKAHEAD = 5

