import pandas as pd
from Dynamic_HYPER_PARAMETER import *
from TORCH_OBJECTS import *
from Dynamic_Instance_generation import input_generation
from DWTA_Simulator import Environment
import DWTA_BHGT as MCTS_Model
import time
from utilities import Get_Logger
from utilities import Average_Meter
import os
from datetime import datetime
import ast
import numpy as np
import json

########################################
# EVALUATION
########################################

SAVE_FOLDER_NAME = "REINFORCE-10-10-10"
print(SAVE_FOLDER_NAME)

logger, result_folder_path = Get_Logger(SAVE_FOLDER_NAME)
print(result_folder_path)
actor = MCTS_Model.ACTOR().to(DEVICE)
# critic = MCTS_Model.Critic().to(DEVICE)
actor_path = 'TRAIN/W5T55EPISODE_0Epoch_25BatchSize1000Buffer_SIZE10Update_Period/CheckPoint_epoch00030/ACTOR_state_dic.pt'
actor.load_state_dict(torch.load(actor_path, map_location=torch.device(DEVICE), weights_only=False))
actor.eval()

file_name='./TEST_INSTANCE/50M_50N_10T.xlsx'
df = pd.read_excel(file_name)

obj = list()
start_time = time.time()
for i in range(100):
    print("index----", i)

    V = ast.literal_eval(df.loc[i]['V'])
    df['P'] = df['P'].apply(lambda x: np.array(json.loads(x)) if isinstance(x, str) else x)
    P = df['P'][i]
    # P = ast.literal_eval(df.loc[i]['P'])
    TW = ast.literal_eval(df.loc[i]['TW'])
    TW = np.array(TW)

    # data generation
    assignment_encoding, weapon_to_target_prob = input_generation(NUM_WEAPON=NUM_WEAPONS, NUM_TARGET=NUM_TARGETS, value=V, prob=P, TW=TW, max_time=MAX_TIME, batch_size=VAL_BATCH)
    assignment_encoding = assignment_encoding[:, None, :, :].expand(assignment_encoding.size(0), VAL_PARA,  NUM_TARGETS * NUM_WEAPONS + 1, 9)
    weapon_to_target_prob = weapon_to_target_prob[:, None, :, :].expand(assignment_encoding.size(0), VAL_PARA, NUM_WEAPONS, NUM_TARGETS)
    env_e = Environment(assignment_encoding=assignment_encoding, weapon_to_target_prob=weapon_to_target_prob, max_time=MAX_TIME)
    current_state = env_e.assignment_encoding

    # print("env.----pure", env_e.target_emerging_time)

    for time_clock in range(MAX_TIME):

        print("time", time_clock)

        for index in range(NUM_WEAPONS):

            policy,  _ = actor(assignment_embedding = env_e.assignment_encoding.detach().clone(), prob=weapon_to_target_prob.clone(), mask=env_e.mask.clone())
            # action_index = torch.multinomial(policy.view(-1, NUM_WEAPONS * NUM_TARGETS + 1), 1).view(env_e.assignment_encoding.size(0), env_e.assignment_encoding.size(1))
            # action_index = torch.multinomial(policy.view(-1, NUM_WEAPONS * NUM_TARGETS + 1), 1).view(VAL_BATCH, VAL_PARA)
            action_index = policy.view(-1, NUM_WEAPONS * NUM_TARGETS + 1).argmax(dim=1).view(assignment_encoding.size(0), assignment_encoding.size(1))

            # if time_clock <3:
            #     print("time_clock", time_clock)
            #     print("action_index", action_index)
            #     print("machine_index", action_index//NUM_TARGETS)
            #     print("target_index", action_index % NUM_TARGETS)
            #     print("mask", env_e.mask)
            #     a = input()
            # print("action_index", action_index)
            env_e.update_internal_variables(selected_action=action_index)
            # a =input()

        env_e.time_update()


    obj_value =  (env_e.current_target_value[:, :, 0:NUM_TARGETS]).sum(2)
    obj_value_ = obj_value.squeeze()
    obj_values = torch.min(obj_value_)
    obj.append(obj_values)

    logger.info('---------------------------------------------------')
    logger.info('value = {}'.format(env_e.current_target_value[:, :, 0:NUM_TARGETS].sum(2)))
    logger.info('average = {}'.format(sum(obj) / len(obj)))
    logger.info('---------------------------------------------------')
    logger.info('---------------------------------------------------')

end_time = time.time()

logger.info('---------------------------------------------------')
logger.info('average = {}'.format(sum(obj) / len(obj)))
logger.info('time = {}'.format((end_time - start_time) / 100))
logger.info('---------------------------------------------------')
logger.info('---------------------------------------------------')

obj_cpu = [tensor.cpu().numpy() if hasattr(tensor, 'cpu') else tensor for tensor in obj]

# Convert the list of NumPy arrays to a DataFrame
df = pd.DataFrame(obj_cpu, columns=['obj'])

# Save the DataFrame to a CSV file
csv_file_path = 'INF/REINFORCE-10-10-10/RE-10-10-10.csv'
df.to_csv(csv_file_path, index=False)
print(f"DataFrame saved to {csv_file_path}")


