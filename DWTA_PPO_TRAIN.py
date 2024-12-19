import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import time
import numpy as np
from utilities import Get_Logger
import matplotlib.pyplot as plt
from Dynamic_HYPER_PARAMETER import *
from TORCH_OBJECTS import *
import DWTA_BHGT as MCTS_Model
import Dynamic_Sampling_BHGT as Sampler
import DWTA_Evaluation
import pandas as pd
import ast

SAVE_FOLDER_NAME = f'W{NUM_WEAPONS}T{NUM_TARGETS}' \
                   f'{TOTAL_EPISODE}EPISODE_{TOTAL_EPISODE//UPDATE_PERIOD}Epoch_{MINI_BATCH}BatchSize{BUFFER_SIZE}Buffer_SIZE{UPDATE_PERIOD}Update_Period'
logger, result_folder_path = Get_Logger(SAVE_FOLDER_NAME)

# create actor/critic
actor = MCTS_Model.ACTOR().to(DEVICE)
old_actor = MCTS_Model.ACTOR().to(DEVICE)
critic = MCTS_Model.Critic().to(DEVICE)

# target_model.load_state_dict(mcts_model.state_dict())
actor.optimizer = optim.Adam(actor.parameters(), lr=ACTOR_LEARNING_RATE, weight_decay=ACTOR_WEIGHT_DECAY)
critic.optimizer = optim.Adam(critic.parameters(), lr=ACTOR_LEARNING_RATE, weight_decay=ACTOR_WEIGHT_DECAY)
actor.lr_stepper = lr_scheduler.MultiStepLR(actor.optimizer, milestones=[250], gamma=0.1)
critic.lr_stepper = lr_scheduler.MultiStepLR(critic.optimizer, milestones=[250], gamma=0.1)
timer_start = time.time()

EVAL_RESULT = []
Epoch_Train_Reward = []
Epoch_Train_Obj = []
CLIP_RANGE = 0.2
PPO_EPOCHS = 4

# test files
file_name = './TEST_INSTANCE/5M_5N.xlsx'
df = pd.read_excel(file_name)

# Training loop
for epoch in range(TOTAL_EPOCH):

    for episode in range(1, TOTAL_EPISODE + 1):
        Sampler.self_play(old_actor=old_actor, actor=actor, critic=critic, episode=episode, temp=None, epoch=epoch)

    obj_list_1 = list()
    obj_list_2 = list()
    for i in range(1):
        # print("i ==", i)

        V = ast.literal_eval(df.loc[i]['V'])
        P = ast.literal_eval(df.loc[i]['P'])
        TW = ast.literal_eval(df.loc[i]['TW'])
        TW = np.array(TW)
        prob_np = np.array(P)
        obj_1 = DWTA_Evaluation.evaluation_pure(model=actor, value=V, prob=prob_np, TW=TW, episode=None)
        obj_list_1.append(obj_1)
        obj_2 =0
        obj_list_2 = [0]
        # obj_2 = DWTA_Evaluation.evaluation(actor=actor,critic=critic, value=V, prob=prob_np, TW=TW, episode=None)
        # obj_list_2.append(obj_2)

    time_str = time.strftime("%H:%M:%S", time.gmtime(time.time() - timer_start))
    log_str = 'Epoch :{:03d}--Mean Left Over-pure:{:5f}---Mean Left Over-mcts:{:5f}'.format(epoch, sum(obj_list_1)/len(obj_list_1), sum(obj_list_2)/len(obj_list_2))
    logger.info(log_str)

# RESULT PLOT
#if isinstance(obj_list_1, torch.Tensor):
# obj_list_1_ = obj_list_1.cpu().numpy()
#
# plt.figure(figsize=(10, 5))
# plt.plot(obj_list_1_, label='Objective 1')
# plt.title('Evaluation Results')
# plt.xlabel('Iteration')
# plt.ylabel('Objective Value')
# plt.legend()
# plt.grid(True)
#
# # Save the plot
# plt.savefig(f'{result_folder_path}/eval_result.jpg')
# plt.close()