# This file includes parameter settings


################################
# Parameters for problem setting
################################
DEBUG = 'yes'
ENCODER_NAME = 'HGT'
NUM_WEAPONS = 40
NUM_TARGETS = 40
MIN_TARGET_VALUE = 1
MAX_TARGET_VALUE = 10
MAX_TIME = 5
PREPARATION_TIME = [1, 2, 1, 2, 1,   1, 2, 1, 2, 1,   1, 2, 1, 2, 1,   1, 2, 1, 2, 1,   1, 2, 1, 2, 1,   1, 2, 1, 2, 1,   1, 2, 1, 2, 1,   1, 2, 1, 2, 1]#,   1, 2, 1, 2, 1,  1, 2, 1, 2, 1]
# MAX_TIME_WINDOW = [10, 10, 10, 10, 10, 50, 50, 50, 50, 50]
# EMERGING_TIME = [0, 2, 1, 3, 0,   0, 2, 1, 3, 0]
AMM = [4, 4, 4, 4, 4,   4, 4, 4, 4, 4, 4, 4, 4, 4, 4,   4, 4, 4, 4, 4, 4, 4, 4, 4, 4,   4, 4, 4, 4, 4, 4, 4, 4, 4, 4,   4, 4, 4, 4, 4, 4, 4, 4, 4, 4,   4, 4, 4, 4, 4]#   10-10-10, 40, 30, 40, 30, 40, 4, 3, 4, 3, 4, 4, 3, 4, 3, 4]

#AMM = [2, 2, 2, 2, 2,   2, 2, 2, 2, 2] #   (10-5-10)
#AMM = [4, 3, 4, 3, 4,   4, 6, 8, 6, 8]#,   8, 6, 8, 6, 8,   8, 6, 8, 6, 8]
LOW_PROB = 0.2
HIGH_PROB = 0.9
###############################
# Train: Hyper-Parameters

###############################`
TOTAL_EPISODE = 5             # flexible
BUFFER_SIZE = 1000
UPDATE_PERIOD = 10
TOTAL_EPOCH = 300
#TOTAL_EPOCH = TOTAL_EPISODE/UPDATE_PERIOD
MINI_BATCH = MAX_TIME*NUM_WEAPONS
EVALUATION_PERIOD = UPDATE_PERIOD
NUM_EVALUATION = 200
TRAIN_BATCH = 50
VAL_BATCH = 1
VAL_PARA = 1
NUM_PAR = 5
SYNC_TARGET = 60
ACTOR_LEARNING_RATE = 1e-3
ACTOR_WEIGHT_DECAY = 1e-3
CRITIC_LEARNING_RATE = 1e-4
CRITIC_WEIGHT_DECAY = 1e-6
START_PUCT = 4.0
EPS_MIN = 1.414
EPS_TARGET = 1000
#EPS_DECAY = (START_PUCT -EPS_MIN)/EPS_TARGET

##############################
# TRANSFORMER_MODEL-Parameters
##############################
INPUT_DIM_W = NUM_WEAPONS + 1      # 3 is not feature but used in the simulation
INPUT_DIM_T = NUM_TARGETS + 1    # +1 Probability
EMBEDDING_DIM = 256               # ORIGINAL 256
HIDDEN_DIM = 256
KEY_DIM = 16                 # Length of q, k, v of EACH attention head
SQRT_KEY_DIM = int(KEY_DIM ** 0.5)
HEAD_NUM = 16
ENCODER_LAYER_NUM = 4      # 6 by 6 =2,
FF_HIDDEN_DIM = 512         # ORIGINAL 512
#############################
# MCTS parameters
############################
NUM_SIM = 100
n_sim = max(1, int(NUM_SIM/(NUM_WEAPONS)))
DEVICE = 'mps'

alpha = 5
n_step = 10
TEST_MODE ='yes'
CHUNK ='True'



































































































































































