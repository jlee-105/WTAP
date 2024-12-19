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

SAVE_FOLDER_NAME = "GreedySearch_10-10-5"
logger, result_folder_path = Get_Logger(SAVE_FOLDER_NAME)
file_name= './TEST_INSTANCE/5M_5N.xlsx'
df = pd.read_excel(file_name)

########################################
# EVALUATION
########################################

obj = list()
for i in range(10):

    print("test inference index----", i)
    V = ast.literal_eval(df.loc[i]['V'])
    df['P'] = df['P'].apply(lambda x: np.array(json.loads(x)) if isinstance(x, str) else x)
    P = df['P'][i]
    TW = ast.literal_eval(df.loc[i]['TW'])
    TW = np.array(TW)
    prob_np = np.array(P)

    # data generation
    assignment_encoding, weapon_to_target_prob = input_generation(NUM_WEAPON=NUM_WEAPONS, NUM_TARGET=NUM_TARGETS, value=V, prob=prob_np, TW=TW, max_time=MAX_TIME, batch_size=1)
    env_e = Environment(assignment_encoding=assignment_encoding, weapon_to_target_prob=weapon_to_target_prob, max_time=MAX_TIME)
    current_state = env_e.assignment_encoding

    for time_clock in range(MAX_TIME):

        print("time_clock", time_clock)

        for index in range(NUM_WEAPONS):

            if len(env_e.available_actions)>0:

                reward = -10
                best_index= - 1

                for action_index in env_e.available_actions:
                    # print("action_index", action_index)
                    greedy_env = copy.deepcopy(env_e)

                    before = greedy_env.current_target_value[:NUM_TARGETS].clone().sum()

                    # a = input()
                    greedy_env.update_internal_variables(selected_action=action_index)
                    # print(greedy_env.current_target_value[:NUM_TARGETS].sum())
                    immediate_reward = (before - greedy_env.current_target_value[:NUM_TARGETS].sum())/before
                    # print("immediate_reward", immediate_reward)
                    if immediate_reward > reward:
                        best_index = action_index
                        reward = immediate_reward

                    del greedy_env
            else:
                best_index = NUM_WEAPONS*NUM_TARGETS

            action_index = best_index
            env_e.update_internal_variables(selected_action=action_index)

        env_e.time_update()
    """ GET OBJECTIVE"""
    obj_value, _ = env_e.reward_calculation()
    obj.append(env_e.current_target_value[:NUM_TARGETS].sum())

    logger.info('---------------------------------------------------')
    logger.info('value = {}'.format(env_e.current_target_value[:NUM_TARGETS].sum()))
    logger.info('average = {}'.format(sum(obj) / len(obj)))
    logger.info('---------------------------------------------------')
    logger.info('---------------------------------------------------')

logger.info('---------------------------------------------------')
logger.info('average = {}'.format(sum(obj)/len(obj)))
logger.info('---------------------------------------------------')
logger.info('---------------------------------------------------')
