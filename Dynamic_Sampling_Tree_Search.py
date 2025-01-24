import pandas as pd

#from DWTA_INFERENCE import beam_actor
from Dynamic_Instance_generation import *
from DWTA_Simulator import Environment
import time
import MCTS_WTAP
import gc
import copy
import DWTA_BHGT as MCTS_Model
from utilities import Average_Meter, Get_Logger
import ast
import numpy as np
import json
import time
import os
from datetime import datetime
from BEAM_WITH_SIMULATION import *

SAVE_FOLDER_NAME = "TreeSearch"
print(SAVE_FOLDER_NAME)

logger, result_folder_path = Get_Logger(SAVE_FOLDER_NAME)
print(result_folder_path)


def self_play(actor, old_actor, episode, temp, epoch):

    actor.train()

    # distance_min = Average_Meter()
    # distance_mean = Average_Meter()
    total_reward = 0
    obj_function_value = 0

    # data generation
    assignment_encoding, weapon_to_target_prob = input_generation(NUM_WEAPON=NUM_WEAPONS, NUM_TARGET=NUM_TARGETS,  value=None, prob=None, TW=None, max_time=MAX_TIME, batch_size=1)
    # shape=[batch, num_assignment+1, 9], shape=[batch, num_w, num_t]
    assignment_encoding = assignment_encoding[:, None, :, :].expand(1, VAL_PARA, NUM_TARGETS * NUM_WEAPONS + 1, 9)
    assignment_encoding = assignment_encoding.repeat(alpha, 1, 1, 1)
    weapon_to_target_prob = weapon_to_target_prob[:, None, :, :].expand(1, VAL_PARA, NUM_WEAPONS, NUM_TARGETS)
    weapon_to_target_prob = weapon_to_target_prob.repeat(alpha, 1, 1, 1)
    env_e = Environment(assignment_encoding=assignment_encoding, weapon_to_target_prob=weapon_to_target_prob, max_time=MAX_TIME)

    # This is the part regular -----------------------------------------------------------------
    group_prob_list = torch.empty((alpha, VAL_PARA, 0)).to(DEVICE)
    beam_actor = MCTS_Model.ACTOR().to(DEVICE)
    beam_actor.load_state_dict(actor.state_dict())


    for time_clock in range(MAX_TIME):

        # print("inference master time clock", time_clock)

        for index in range(NUM_WEAPONS):

            # print("n--------------------------------weapon", index)

            # check all possible expansions
            possible_actions = env_e.mask.clone()

            if (possible_actions < NUM_WEAPONS * NUM_TARGETS).any():

                beam_env = copy.deepcopy(env_e)
                # print("beem_env____copied----- current--target_value", beam_env.current_target_value)
                # a = input()
                # beam_actor = copy.deepcopy(actor)
                beam_search = Beam_Search(actor=beam_actor, env=beam_env, available_actions=possible_actions)
                # dimension adjust
                beam_search.reset()
                node_index = beam_search.expand_actions()
                selected_batch_index, selected_group_index = beam_search.do_beam_simulation(node_index=node_index, time=time_clock, w_index=index)
                selected_action = selected_group_index.unsqueeze(dim=1)
                # print("selected_batch_index----------------------------", selected_batch_index)
                # a = input()
                new_env = batch_dimension_resize(env=beam_env, batch_index=selected_batch_index, group_index=selected_group_index)
                # print("new----env---target_value", new_env.current_target_value)
                env_e = copy.deepcopy(new_env)

                del beam_env

            else:
                selected_action = torch.tensor([NUM_WEAPONS * NUM_TARGETS]).to(DEVICE)
                selected_action = selected_action[None, :].expand(alpha, selected_action.size(0))

            policy, value, embedding = actor(assignment_embedding=env_e.assignment_encoding.detach().clone(), prob=env_e.weapon_to_target_prob, mask=env_e.mask)


            # -------------------------------------------------------------------------------
            selected_probs = policy.gather(2, selected_action.unsqueeze(2))
            group_prob_list = torch.cat((group_prob_list, selected_probs), dim=2)

            # sampling
            parent_mask = env_e.mask.detach().clone()
            parent_state = env_e.assignment_encoding.detach().clone()
            before_value = env_e.current_target_value.detach().clone()[:, :, 0:NUM_TARGETS]
            before_value = torch.sum(before_value, dim=-1)


            env_e.update_internal_variables(selected_action=selected_action)

            current_value = env_e.current_target_value.detach().clone()[:, :, 0:NUM_TARGETS]
            current_value = torch.sum(current_value, dim=-1)
            reward = (before_value - current_value) / env_e.original_target_value[:, :, 0:NUM_TARGETS].sum(dim=-1)


            if env_e.n_fires < MAX_TIME * NUM_WEAPONS:
                env_e.all_weapon_NOT_done = torch.ones(alpha, VAL_PARA).to(DEVICE)
            else:
                env_e.all_weapon_NOT_done = torch.zeros(alpha, VAL_PARA).to(DEVICE)

            actor.replay_memory.push(
                (
                    env_e.assignment_encoding.detach().clone(),  # next state
                    parent_state.detach().clone(),  # current state
                    selected_action.clone(),  # action
                    reward.clone(),  # reward
                    parent_mask.detach().clone(),  # current mask
                    env_e.mask.clone(),  # next mask
                    env_e.all_weapon_NOT_done.clone()  # done
                )
            )

        env_e.time_update()

    # print("env-n_fires", env_e.n_fires)
    # print(env_e.current_target_value)

    obj_value = (env_e.current_target_value[:, :, 0:NUM_TARGETS]).sum(2)
    # print(obj_value)

    # print("I AM DONE----------------------")
    # a = input()

    # Update Part
    storage = actor.replay_memory.memory

    for time_index in range(MAX_TIME * NUM_WEAPONS):

        batch = storage[time_index]
        next_state_batch = batch[0].to(device=DEVICE)
        parent_state_batch = batch[1].to(device=DEVICE)
        action_batch = batch[2].to(device=DEVICE)
        reward_batch = batch[3].unsqueeze(-1).to(device=DEVICE)
        parent_state_mask_batch = batch[4].to(device=DEVICE)
        next_state_mask_batch = batch[5].to(device=DEVICE)
        done_batch = batch[6].unsqueeze(-1).to(device=DEVICE)

        # old_actor.load_state_dict(actor.state_dict())
        # # Freeze the parameters of old_actor
        # for param in old_actor.parameters():
        #     param.requires_grad = False

        # for _ in range(PPO_EPOCHS):
        #     with torch.no_grad():
        #         old_probs, _, _ = old_actor(assignment_embedding=parent_state_batch, prob=None,
        #                                     mask=parent_state_mask_batch)
        #         old_log_probs = torch.log(old_probs.gather(2, action_batch.unsqueeze(-1)))

        # Calculate probabilities with the current actor for gradient updates
        new_probs, current_state_value, _ = actor(assignment_embedding=parent_state_batch.clone(), prob=None,
                                                  mask=parent_state_mask_batch.clone())

        new_log_probs = torch.log(new_probs.gather(2, action_batch.unsqueeze(-1)))
        """Need to find elite_action"""

        actor_loss = -new_log_probs.mean()

        # print("actor_loss", actor_loss)


        # critic part ----------------------------------------------------------------------------------------------
        small_number = 1e-10
        zero_batches = torch.all(next_state_mask_batch == 0, dim=(2))
        next_state_mask_batch[zero_batches] += small_number
        _, next_state_value, _ = actor(assignment_embedding=next_state_batch, prob=None, mask=next_state_mask_batch)

        expected_values = reward_batch + next_state_value.reshape(next_state_value.size(0), next_state_value.size(1), -1) * (done_batch.float())
        advantages = expected_values.squeeze().detach() - current_state_value
        squared_advantages = advantages.pow(2)
        sum_over_parallel = squared_advantages.sum(dim=1)
        critic_loss = sum_over_parallel.mean()

        loss = actor_loss + critic_loss

        actor.optimizer.zero_grad()
        loss.backward()
        actor.optimizer.step()

    actor.replay_memory.refresh()
    actor.lr_stepper.step()



    if epoch >= 1 and episode == 5:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        log_str = 'Epoch :{:03d}--actor_loss:{:5f}---value_loss:{:5f}'.format(epoch, actor_loss, critic_loss)
        logger.info(log_str)
        checkpoint_folder_path = '{}/CheckPoint_epoch{:05d}'.format(result_folder_path, epoch)
        os.mkdir(checkpoint_folder_path)


        model_save_path = '{}/ACTOR_state_dic.pt'.format(checkpoint_folder_path)
        torch.save(actor.state_dict(), model_save_path)
        model_save_path = '{}/Critic_state_dic.pt'.format(checkpoint_folder_path)
        # torch.save(critic.state_dict(), model_save_path)

    return total_reward, obj_function_value

