import pandas as pd
from Dynamic_Instance_generation import *
from DWTA_Simulator import Environment
import DWTA_BHGT
import gc
import copy
import DWTA_BHGT as MCTS_Model
from utilities import Average_Meter, Get_Logger
import ast
import numpy as np
import json
import time

SAVE_FOLDER_NAME = "GreedySearch_10-5-10"
logger, result_folder_path = Get_Logger(SAVE_FOLDER_NAME)
file_name= './TEST_INSTANCE/10M_5N_10T.xlsx'
df = pd.read_excel(file_name)


########################################
# EVALUATION
########################################
start_time = time.time()
obj = list()
for i in range(1):

    print("test inference index----", i)
    V = ast.literal_eval(df.loc[i]['V'])
    df['P'] = df['P'].apply(lambda x: np.array(json.loads(x)) if isinstance(x, str) else x)
    P = df['P'][i]
    TW = ast.literal_eval(df.loc[i]['TW'])
    TW = np.array(TW)
    prob_np = np.array(P)

    # data generation
    assignment_encoding, weapon_to_target_prob = input_generation(NUM_WEAPON=NUM_WEAPONS, NUM_TARGET=NUM_TARGETS, value=V, prob=prob_np, TW=TW, max_time=MAX_TIME, batch_size=1)
    batch_size = assignment_encoding.size(0)
    node_size = assignment_encoding.size(1)
    feature_size = assignment_encoding.size(2)
    assignment_encoding = assignment_encoding[:, None, :, :].expand(batch_size, 1, node_size, feature_size)
    weapon_to_target_prob = weapon_to_target_prob[:, None, :, :].expand(batch_size, 1, NUM_WEAPONS, NUM_TARGETS)
    env_e = Environment(assignment_encoding=assignment_encoding, weapon_to_target_prob=weapon_to_target_prob, max_time=MAX_TIME)
    current_state = env_e.assignment_encoding

    print(env_e.current_target_value)
    for time_clock in range(MAX_TIME):

        print("time_clock", time_clock)

        for index in range(NUM_WEAPONS):

            #print("env_e.available_action", env_e.available_actions)
            action_list = torch.nonzero(env_e.available_actions, as_tuple=True)[-1]
            #print(action_list)


            if len(action_list)>0:

                reward = -10
                best_index= - 1

                for action_index in action_list:
                    # print("action_index", action_index)
                    greedy_env = copy.deepcopy(env_e)
                    before = greedy_env.current_target_value[:, :, 0:NUM_TARGETS].clone().sum(2)
                    action_index_reshape = torch.tensor([[action_index.item()]]).to(DEVICE)
                    # a = input()
                    greedy_env.update_internal_variables(selected_action=action_index_reshape)
                    # print(greedy_env.current_target_value[:NUM_TARGETS].sum())
                    immediate_reward = (before - greedy_env.current_target_value[:,:, 0:NUM_TARGETS].sum())/before
                    # print("immediate_reward", immediate_reward)
                    if immediate_reward > reward:
                        best_index = action_index_reshape
                        reward = immediate_reward

                    del greedy_env
            else:
                best_index = torch.tensor([[NUM_WEAPONS*NUM_TARGETS]]).to(DEVICE)


            action_index = best_index
            # print(action_index)
            # a = input()
            env_e.update_internal_variables(selected_action=action_index)

        env_e.time_update()
    """ GET OBJECTIVE"""


    #obj_value, _ = env_e.reward_calculation()
    # print(env_e.current_target_value)
    # a = input()

    obj.append(env_e.current_target_value[:, :, 0:NUM_TARGETS].sum())

    logger.info('---------------------------------------------------')
    logger.info('value = {}'.format(env_e.current_target_value[:, :, 0:NUM_TARGETS].sum()))
    logger.info('average = {}'.format(sum(obj) / len(obj)))
    logger.info('---------------------------------------------------')
    logger.info('---------------------------------------------------')

end_time = time.time()

logger.info('---------------------------------------------------')
logger.info('average = {}'.format(sum(obj)/len(obj)))
logger.info('time = {}'.format((end_time-start_time)/100))
logger.info('---------------------------------------------------')
logger.info('---------------------------------------------------')


obj_cpu = [tensor.cpu().numpy() if hasattr(tensor, 'cpu') else tensor for tensor in obj]

# Convert the list of NumPy arrays to a DataFrame
df = pd.DataFrame(obj_cpu, columns=['obj'])

# Save the DataFrame to a CSV file
csv_file_path = 'Greedy/GreedySearch_10-5-10/greedy-10-5-10-.csv'
df.to_csv(csv_file_path, index=False)
print(f"DataFrame saved to {csv_file_path}")