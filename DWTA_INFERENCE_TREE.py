import pandas as pd

from Dynamic_Instance_generation import *
from DWTA_Simulator import Environment
import MCTS_WTAP
import gc
import copy
import DWTA_BHGT as MCTS_Model
from utilities import Average_Meter, Get_Logger
import ast
import numpy as np
import json
import time
from BEAM_WITH_SIMULATION import *

SAVE_FOLDER_NAME = "REINFORCE-OLD_TRAIN-WITHOUT-MCTS-10-10-5-CP2000"
print(SAVE_FOLDER_NAME)

logger, result_folder_path = Get_Logger(SAVE_FOLDER_NAME)
print(result_folder_path)
actor = MCTS_Model.ACTOR().to(DEVICE)
# critic = MCTS_Model.Critic().to(DEVICE)
actor_path = 'Training/20250110_1521__W5T55EPISODE_0Epoch_25BatchSize10000Buffer_SIZE10Update_Period/CheckPoint_epoch00002/ACTOR_state_dic.pt'
actor.load_state_dict(torch.load(actor_path, map_location=torch.device(DEVICE), weights_only=False))
actor.eval()
# critic.eval()
#

# logger.info('==============================================================================')
# logger.info('==============================================================================')
# log_str = '  <<< MODEL: {:s} >>>'.format(model_path)
# logger.info(log_str)

# data file name
file_name='./TEST_INSTANCE/5M_5N.xlsx'
df = pd.read_excel(file_name)

########################################
# EVALUATION
########################################





obj_container = list()
for i in range(1):
    start = time.time()
    print("index----", i)
    i = i+2

    V = ast.literal_eval(df.loc[i]['V'])
    df['P'] = df['P'].apply(lambda x: np.array(json.loads(x)) if isinstance(x, str) else x)
    P= df['P'][i]
    #P = ast.literal_eval(df.loc[i]['P'])
    TW = ast.literal_eval(df.loc[i]['TW'])
    TW = np.array(TW)

    #prob_np = np.array(P)

    # initial data generation
    assignment_encoding, weapon_to_target_prob = input_generation(NUM_WEAPON=NUM_WEAPONS, NUM_TARGET=NUM_TARGETS, value=V, prob=P, TW=TW, max_time=MAX_TIME, batch_size=1)
    assignment_encoding = assignment_encoding[:, None, :, :].expand(1, VAL_PARA, NUM_TARGETS * NUM_WEAPONS + 1, 9)
    assignment_encoding = assignment_encoding.repeat(alpha, 1, 1, 1)
    weapon_to_target_prob = weapon_to_target_prob[:, None, :, :].expand(1, VAL_PARA, NUM_WEAPONS, NUM_TARGETS)
    weapon_to_target_prob = weapon_to_target_prob.repeat(alpha, 1, 1, 1)
    env_e = Environment(assignment_encoding=assignment_encoding, weapon_to_target_prob=weapon_to_target_prob, max_time=MAX_TIME)

    beam_actor = copy.deepcopy(actor)
    for time_clock in range(MAX_TIME):

        print("inference master time clock", time_clock)

        for index in range(NUM_WEAPONS):

            #check all possible expansions
            possible_actions = env_e.mask.clone()

            if (possible_actions < NUM_WEAPONS*NUM_TARGETS).any():


                beam_env = copy.deepcopy(env_e)
                # print("beem_env____copied", beam_env)
                # print("beem_env____copied----- current--target_value", beam_env.current_target_value)
                # a = input()
                # beam_actor = copy.deepcopy(actor)
                beam_search = Beam_Search(actor=beam_actor, env=beam_env, available_actions=possible_actions)
                # dimension adjust
                beam_search.reset()
                node_index = beam_search.expand_actions()
                selected_batch_index, selected_group_index = beam_search.do_beam_simulation(node_index = node_index, time=time_clock, w_index=index)
                selected_action = selected_group_index.unsqueeze(dim=1)
                # print("selected_batch_index----------------------------", selected_batch_index)
                # a = input()
                new_env = batch_dimension_resize(env=beam_env, batch_index=selected_batch_index, group_index=selected_group_index)
                # print("new----env---target_value", new_env.current_target_value)
                env_e = copy.deepcopy(new_env)

                del beam_env
            else:
                selected_action = torch.tensor([NUM_WEAPONS*NUM_TARGETS]).to(DEVICE)
                selected_action = selected_action[None, :].expand(alpha, selected_action.size(0))

            env_e.update_internal_variables(selected_action=selected_action)
            # print("new--_value", env_e.current_target_value)





            # print("selected_group_index", selected_group_index)
            # print("selected_group_index.shape", selected_group_index.shape)
            # selected_action = selected_group_index.unsqueeze(dim=1)
            #
            # print("env-n_fire", env_e.n_fires)
            #
            # env_e.update_internal_variables(selected_action=selected_action)
            # print("env-n_fire--after", env_e.n_fires)
            # a = input()

        env_e.time_update()

    # print("env-n_fires", env_e.n_fires)
    # print(env_e.current_target_value)

    obj_value =  (env_e.current_target_value[:, :, 0:NUM_TARGETS]).sum(2)

    # obj_value_ = obj_value.squeeze()
    # obj_values = torch.min(obj_value_)

    print(obj_value)


    """ GET OBJECTIVE"""
    # obj_value, _ = env_e.reward_calculation()
    # obj_container.append(obj_values)
    # logger.info('---------------------------------------------------')
    # logger.info('average = {}'.format(obj_values))
    # logger.info('average = {}'.format(sum(obj_container) / len(obj_container)))
    # logger.info('---------------------------------------------------')
    # logger.info('---------------------------------------------------')







