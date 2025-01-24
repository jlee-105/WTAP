import copy

import torch

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

########################################
# EVALUATION
########################################

def evaluation(actor, critic, value, prob, TW, episode):

    actor.eval()
    critic.eval()

    # data generation
    assignment_encoding, weapon_to_target_prob = input_generation(NUM_WEAPON=NUM_WEAPONS, NUM_TARGET=NUM_TARGETS, value=value, prob=prob, TW=TW, max_time=MAX_TIME, batch_size=VAL_BATCH)
    assignment_encoding = assignment_encoding[:, None, :, :].expand(assignment_encoding.size(0), VAL_PARA, NUM_TARGETS * NUM_WEAPONS + 1, 9)
    weapon_to_target_prob = weapon_to_target_prob[:, None, :, :].expand(assignment_encoding.size(0), VAL_PARA, NUM_WEAPONS, NUM_TARGETS)
    env_e = Environment(assignment_encoding=assignment_encoding, weapon_to_target_prob=weapon_to_target_prob, max_time=MAX_TIME)
    current_state = env_e.assignment_encoding


    # print("env.", current_state)
    # a = input()

    # print("env.", env_e.target_emerging_time)

    for time_clock in range(MAX_TIME):

        for index in range(NUM_WEAPONS):

            mcts = MCTS_WTAP.MCTS_SIM(n_sim=NUM_SIM, c_puct=4.0)
            # copy current Env
            copy_env = copy.deepcopy(env_e)
            # get simulation result
            root_result = mcts.simulation(env=copy_env, actor=actor, critic=critic, mcts_state=env_e.assignment_encoding.clone())

            # print(root_result.children)

            # get state value
            state_value = [child.state_value[0][0] for child in root_result.children.values()]
            # print(state_value)


            #visit_count = [child.visit_count for child in root_result.children.values()]
            # get immediate reward
            immediate_reward = [child.immediate_reward for child in root_result.children.values()]

            # tankers for updated state value, q_value, reward
            mcts_updated_state_values = [torch.tensor([[0.0]]).to(DEVICE) for _ in range(NUM_WEAPONS * NUM_TARGETS + 1)]
            mcts_updated_q_values = [torch.tensor([[0.0]]).to(DEVICE) for _ in range(NUM_WEAPONS * NUM_TARGETS + 1)]
            mcts_updated_r_values = [torch.tensor([[0.0]]).to(DEVICE) for _ in range(NUM_WEAPONS * NUM_TARGETS + 1)]

            avail_actions = env_e.available_actions[0][0]
            true_indices = avail_actions.nonzero(as_tuple=True)[0]
            # print(true_indices)

            if len(true_indices) > 0:
                skip_added_action = torch.cat((true_indices.detach().clone(), torch.tensor([NUM_WEAPONS * NUM_TARGETS]).to(DEVICE)), dim=0)
            else:
                skip_added_action = torch.tensor([NUM_WEAPONS * NUM_TARGETS]).to(DEVICE)

            # print("skip_added_action", skip_added_action)

            for idx, tensor, reward in zip(skip_added_action, state_value, immediate_reward):

                #print('index', idx)

                if type(tensor) == torch.Tensor:
                    mcts_updated_state_values[idx] = tensor
                    mcts_updated_r_values[idx] = reward
                else:

                    mcts_updated_state_values[idx] = torch.tensor(tensor).to(DEVICE)
                    mcts_updated_r_values[idx] = torch.tensor(tensor).to(DEVICE)

                # get q-value
                mcts_updated_q_values[idx] = mcts_updated_r_values[idx] + mcts_updated_state_values[idx]
                #print(mcts_updated_q_values[idx])

            #mcts_updated_child_state_values = torch.stack(mcts_updated_state_values)

            mcts_updated_q_values = torch.stack(mcts_updated_q_values)
            mcts_action_index = torch.argmax(mcts_updated_q_values)

            mcts_action_index_reshape = mcts_action_index.reshape(1,1)


            # if all other options are zero then Skip
            if torch.all(mcts_updated_q_values[:-1] == 0):
                mcts_action_index_reshape = torch.tensor([[NUM_WEAPONS * NUM_TARGETS]]).to(DEVICE)
            else:
                if torch.all(mcts_updated_q_values <= 0):
                    new_tensor = torch.where(mcts_updated_q_values == 0, mcts_updated_q_values - 5.0, mcts_updated_q_values)
                    mcts_action_index_reshape = torch.argmax(new_tensor)

            action_index = mcts_action_index_reshape
            env_e.update_internal_variables(selected_action=action_index)


            # if action_index < NUM_WEAPONS * NUM_TARGETS:
            #     env_e.update_internal_variables(selected_action=action_index)
        env_e.time_update()
    """ GET OBJECTIVE"""
    obj_value, _ = env_e.reward_calculation()
    # enemy_value_left = env_e.target_value.sum()
    print(env_e.current_target_value[0:NUM_TARGETS].sum())

    return env_e.current_target_value[:, :, 0:NUM_TARGETS].sum()

def evaluation_pure(model, value, prob, TW, episode):

    TRAIN_BATCH = 1
    NUM_PAR = 1

    model.eval()

    # data generation
    assignment_encoding, weapon_to_target_prob = input_generation(NUM_WEAPON=NUM_WEAPONS, NUM_TARGET=NUM_TARGETS, value=value, prob=prob, TW=TW, max_time=MAX_TIME, batch_size=VAL_BATCH)
    assignment_encoding = assignment_encoding[:, None, :, :].expand(assignment_encoding.size(0), VAL_PARA,  NUM_TARGETS * NUM_WEAPONS + 1, 9)
    weapon_to_target_prob = weapon_to_target_prob[:, None, :, :].expand(assignment_encoding.size(0), VAL_PARA, NUM_WEAPONS, NUM_TARGETS)
    env_e = Environment(assignment_encoding=assignment_encoding, weapon_to_target_prob=weapon_to_target_prob, max_time=MAX_TIME)
    current_state = env_e.assignment_encoding

    # print("env.----pure", env_e.target_emerging_time)

    for time_clock in range(MAX_TIME):

        for index in range(NUM_WEAPONS):

            policy,  _ = model(assignment_embedding = env_e.assignment_encoding.detach().clone(), prob=weapon_to_target_prob.clone(), mask=env_e.mask.clone())
            # action_index = torch.multinomial(policy.view(-1, NUM_WEAPONS * NUM_TARGETS + 1), 1).view(env_e.assignment_encoding.size(0), env_e.assignment_encoding.size(1))
            action_index = torch.multinomial(policy.view(-1, NUM_WEAPONS * NUM_TARGETS + 1), 1).view(VAL_BATCH, VAL_PARA)
            env_e.update_internal_variables(selected_action=action_index)
            #print("action_index", action_index)

        env_e.time_update()


    #print((env_e.current_target_value[:, :, 0:NUM_TARGETS]))
    obj_value =  (env_e.current_target_value[:, :, 0:NUM_TARGETS]).sum(2)
    #print("ob", obj_value)
    obj_value_ = obj_value.squeeze()
    obj_values = torch.min(obj_value_)
    # print("obj_values", obj_values)
    # a = input()

    return obj_values
