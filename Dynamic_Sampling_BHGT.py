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

SAVE_FOLDER_NAME = f'W{NUM_WEAPONS}T{NUM_TARGETS}' \
                   f'{TOTAL_EPISODE}EPISODE_{TOTAL_EPISODE//UPDATE_PERIOD}Epoch_{MINI_BATCH}BatchSize{BUFFER_SIZE}Buffer_SIZE{UPDATE_PERIOD}Update_Period'
logger, result_folder_path = Get_Logger(SAVE_FOLDER_NAME)

timer_start = time.time()

def self_play(old_actor, actor, critic, episode, temp, epoch):

    actor.train()
    critic.train()
    # distance_min = Average_Meter()
    # distance_mean = Average_Meter()
    total_reward = 0
    obj_function_value = 0

    # data generation
    assignment_encoding, weapon_to_target_prob = input_generation(NUM_WEAPON=NUM_WEAPONS, NUM_TARGET=NUM_TARGETS,  value=None, prob=None, TW=None, max_time=MAX_TIME, batch_size=TRAIN_BATCH)
    # shape=[batch, num_assignment+1, 9], shape=[batch, num_w, num_t]

    # expand the dimension
    assignment_encoding = assignment_encoding[:, None, :, :].expand(TRAIN_BATCH, NUM_PAR, NUM_TARGETS*NUM_WEAPONS+1, 9)
    weapon_to_target_prob = weapon_to_target_prob[:, None, :, :].expand(TRAIN_BATCH, NUM_PAR, NUM_WEAPONS, NUM_TARGETS)
    # [batch, parallel, n_operation, n_feature]

    # environment generation
    env = Environment(assignment_encoding=assignment_encoding, weapon_to_target_prob=weapon_to_target_prob, max_time=MAX_TIME)
    # action list

    # This is the part regular -----------------------------------------------------------------
    group_prob_list = torch.empty((TRAIN_BATCH, NUM_PAR, 0)).to(DEVICE)



    for time_clock in range(MAX_TIME):

        for index in range(NUM_WEAPONS):

            policy,  embedding = actor(assignment_embedding=env.assignment_encoding.detach().clone(), prob=env.weapon_to_target_prob, mask=env.mask)
            action_index = torch.multinomial(policy.view(-1, NUM_WEAPONS*NUM_TARGETS+1), 1).view(TRAIN_BATCH, NUM_PAR)

            # sampling
            parent_mask = env.mask.detach().clone()
            parent_state = env.assignment_encoding.detach().clone()
            # reward
            before_value = env.current_target_value.detach().clone()[:, :, 0:NUM_TARGETS]
            # dim [train_batch, parallel, n_targets]
            before_value = torch.sum(before_value, dim=-1)
            # dim [train_batch, parallel]

            env.update_internal_variables(selected_action=action_index)
            current_value = env.current_target_value.detach().clone()[:, :, 0:NUM_TARGETS]
            current_value = torch.sum(current_value, dim=-1)
            reward = (before_value - current_value)/env.original_target_value[:, :, 0:NUM_TARGETS].sum(dim=-1)

            # print("reward", reward.shape)
            # a = input()

            # -------------------------------------------------------------------------------
            selected_probs = policy.gather(2, action_index.unsqueeze(2))
            group_prob_list = torch.cat((group_prob_list, selected_probs), dim=2)

            if env.n_fires < MAX_TIME * NUM_WEAPONS:
                env.all_weapon_NOT_done = torch.ones(TRAIN_BATCH, NUM_PAR).to(DEVICE)
            else:
                env.all_weapon_NOT_done = torch.zeros(TRAIN_BATCH, NUM_PAR).to(DEVICE)

            actor.replay_memory.push(
                (
                    env.assignment_encoding.detach().clone(),  # next state
                    parent_state.detach().clone(),  # current state
                    action_index.clone(),  # action
                    reward.clone(),  # reward
                    parent_mask.detach().clone(),  # current mask
                    env.mask.clone(),  # next mask
                    env.all_weapon_NOT_done.clone()  # done
                )
            )
        env.time_update()

    left_value = env.current_target_value[:, :, 0:NUM_TARGETS]
    obj = torch.sum(left_value, dim=2)#/env.original_target_value[:, :, 0:NUM_TARGETS].sum(2)
    # print("obj", obj)
    # a = input()

    group_log_prob = group_prob_list.log().sum(dim=2)
    mean = obj.mean(dim=1, keepdim=True)
    # std = obj.std(dim=1, keepdim=True)
    group_advantage = (mean-obj)
    loss = -group_advantage*group_log_prob
    actor_loss = loss.mean()

    actor.optimizer.zero_grad()
    actor_loss.backward()
    actor.optimizer.step()

    # Update Part
    storage = actor.replay_memory.sample(batch_size=MAX_TIME*NUM_WEAPONS)
    # storage = actor.replay_memory.shuffle


    for time_index in range(MAX_TIME*NUM_WEAPONS):
    #
        batch = storage[time_index]
        next_state_batch = batch[0].to(device=DEVICE)
        parent_state_batch = batch[1].to(device=DEVICE)
        action_batch = batch[2].to(device=DEVICE)
        reward_batch = batch[3].unsqueeze(-1).to(device=DEVICE)
        parent_state_mask_batch = batch[4].to(device=DEVICE)
        next_state_mask_batch = batch[5].to(device=DEVICE)
        done_batch = batch[6].unsqueeze(-1).to(device=DEVICE)
    #
    #
    #     old_actor.load_state_dict(actor.state_dict())
    #     # Freeze the parameters of old_actor
    #     for param in old_actor.parameters():
    #         param.requires_grad = False
    #
    #     for _ in range(4):
    #
    #         with torch.no_grad():
    #             old_probs, _  = old_actor(assignment_embedding=parent_state_batch, prob=None, mask=parent_state_mask_batch)
    #             old_log_probs = torch.log(old_probs.gather(2, action_batch.unsqueeze(-1)))
    #
    #         # Calculate probabilities with the current actor for gradient updates
    #         new_probs, _ = actor(assignment_embedding= parent_state_batch.clone().detach(), prob=None, mask=parent_state_mask_batch.clone().detach())
    #         new_log_probs = torch.log(new_probs.gather(2, action_batch.unsqueeze(-1)))
    #         """Need to find elite_action"""
    #
    #         ratios = torch.exp(new_log_probs - old_log_probs).squeeze(-1)
    #         # Compute clipped objective
    #         surr1 = ratios * group_advantage
    #         surr2 = torch.clamp(ratios, 1 - 0.15, 1 + 0.15) * group_advantage
    #         min_values = torch.min(surr1, surr2)
    #         sum_parallel = min_values.sum(dim=1)
    #
    #         loss = -sum_parallel.mean()
    #         actor_loss = loss
    #
    #         actor.optimizer.zero_grad()
    #         actor_loss.backward()
    #         actor.optimizer.step()


        small_number = 1e-10
        zero_batches = torch.all(next_state_mask_batch == 0, dim=(2))
        next_state_mask_batch[zero_batches] += small_number
        current_state_value = critic(state=parent_state_batch, mask=parent_state_mask_batch)
        next_state_value = critic(state=next_state_batch, mask=next_state_mask_batch)

        advantages = current_state_value - (reward_batch.squeeze().detach().clone() + next_state_value.detach().clone()*done_batch.squeeze().detach().clone())
        squared_advantages = advantages.pow(2)
        sum_over_parallel = squared_advantages.sum(dim=1)
        critic_loss = sum_over_parallel.mean()

        critic.optimizer.zero_grad()
        critic_loss.backward()
        critic.optimizer.step()

        # ratios = torch.exp(new_log_probs - old_log_probs).squeeze(-1)
            # # Compute clipped objective
            # surr1 = ratios * advantages
            # surr2 = torch.clamp(ratios, 1 - 0.1, 1 + 0.1) * advantages
            # min_values = torch.min(surr1, surr2)
            # sum_parallel = min_values.sum(dim=1)
            #
            # loss_ = -sum_parallel.mean()
            # actor_loss = loss_



        # current_state_value = critic(state=parent_state_batch.detach().clone(), mask=parent_state_mask_batch.detach().clone())
        # test_advantages = (current_state_value - obj.detach().clone())
        #
        # squared_advantages = test_advantages.pow(2)
        # sum_over_parallel = squared_advantages.sum(dim=1)
        # critic_loss = sum_over_parallel.mean()
        #



    actor.replay_memory.refresh()
    actor.lr_stepper.step()


    if epoch >=1 and episode==5:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_str = 'Epoch :{:03d}--actor_loss:{:5f}---value_loss:{:5f}'.format(epoch,actor_loss, critic_loss)
        logger.info(log_str)
        checkpoint_folder_path = '{}/CheckPoint_epoch{:05d}'.format(result_folder_path, epoch)
        os.mkdir(checkpoint_folder_path)
        model_save_path = '{}/ACTOR_state_dic.pt'.format(checkpoint_folder_path)
        torch.save(actor.state_dict(), model_save_path)
        model_save_path = '{}/Critic_state_dic.pt'.format(checkpoint_folder_path)
        torch.save(critic.state_dict(), model_save_path)

    return total_reward, obj_function_value
