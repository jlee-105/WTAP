import copy

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

def calculate_cumulative_rewards(rewards, gamma=1.0):
    """
    Calculate cumulative rewards for a multi-dimensional reward structure.
    rewards shape: [batch, parallel, num_state_transitions]
    """
    # Reverse the rewards along the state transitions dimension to calculate cumulatively from the end
    reversed_rewards = torch.flip(rewards, dims=[-1])

    # Prepare to collect cumulative rewards for each batch and parallel path
    batch_size, num_parallel, num_transitions = rewards.shape
    cumulative_rewards = torch.zeros_like(rewards)

    # Calculate cumulatively for each batch and each parallel sequence
    for batch in range(batch_size):
        for parallel in range(num_parallel):
            cumulative_reward = torch.zeros_like(rewards[batch, parallel, 0])
            for t in range(num_transitions):
                cumulative_reward = reversed_rewards[batch, parallel, t] + gamma * cumulative_reward
                cumulative_rewards[batch, parallel, t] = cumulative_reward

    # Flip the cumulative_rewards back to original transitions order
    cumulative_rewards = torch.flip(cumulative_rewards, dims=[-1])

    return cumulative_rewards


def self_play(old_actor, actor, critic, episode, temp, epoch):

    actor.train()
    distance_min = Average_Meter()
    distance_mean = Average_Meter()
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
        # print("----------------------time", time_clock)

        for index in range(NUM_WEAPONS):

            policy, embedding = actor(assignment_embedding=env.assignment_encoding.detach().clone(), prob=env.weapon_to_target_prob, mask=env.mask)
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
            reward = (before_value - current_value)

            # -------------------------------------------------------------------------------
            selected_probs = policy.gather(2, action_index.unsqueeze(2))
            group_prob_list = torch.cat((group_prob_list, selected_probs), dim=2)

            if env.n_fires < MAX_TIME * NUM_WEAPONS:
                env.all_weapon_NOT_done = torch.ones(TRAIN_BATCH, NUM_PAR).to(DEVICE)
            else:
                env.all_weapon_NOT_done = torch.zeros(TRAIN_BATCH, NUM_PAR).to(DEVICE)

            actor.memory.push(
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
    obj = torch.sum(left_value, dim=2)
    group_log_prob = group_prob_list.log().sum(dim=2)
    mean = obj.mean(dim=1, keepdim=True)
    std = obj.std(dim=1, keepdim=True)
    group_advantage = (mean-obj)

    # Update Part
    storage = actor.memory.memory
    #
    # # reward calculation in backward way
    # rewards = torch.stack([item[3] for item in storage])
    # rewards = rewards.reshape(TRAIN_BATCH, NUM_PAR, -1)
    # cumulated_rewards = calculate_cumulative_rewards(rewards=rewards)
    #
    # state = [item[1] for item in storage]
    # states = torch.stack(state)
    # states = states.reshape(TRAIN_BATCH, NUM_PAR, NUM_WEAPONS*MAX_TIME, NUM_WEAPONS*NUM_TARGETS+1, -1)
    #
    # mask = [item[4] for item in storage]
    # masks = torch.stack(mask)
    # masks = masks.reshape(TRAIN_BATCH, NUM_PAR, NUM_WEAPONS * MAX_TIME, NUM_WEAPONS * NUM_TARGETS + 1, -1)
    # current_state_value = critic(state=states.detach().clone(),  mask=masks.detach().clone())


    # losses = -group_log_prob * group_advantage  # Element-wise multiplication
    # #group_losses = losses.sum(-1)  # Summing over the last two dimensions, result shape [2]
    # loss = losses.mean()
    #
    # actor.optimizer.zero_grad()
    # loss.backward()
    # actor.optimizer.step()
    # actor.memory.refresh()

    # advantage = current_state_value - cumulated_rewards.detach().clone()
    # group_value_loss = advantage.pow(2).sum(-1)
    # value_loss = group_value_loss.mean()
    # critic.optimizer.zero_grad()
    # value_loss.backward()
    # critic.optimizer.step()




    # for time_index in range(MAX_TIME*NUM_WEAPONS):
    #
    #     batch = storage[time_index]
    #     next_state_batch = batch[0].to(device=DEVICE)
    #     parent_state_batch = batch[1].to(device=DEVICE)
    #     action_batch = batch[2].to(device=DEVICE)
    #     reward_batch = batch[3].unsqueeze(-1).to(device=DEVICE)
    #     parent_state_mask_batch = batch[4].to(device=DEVICE)
    #     next_state_mask_batch = batch[5].to(device=DEVICE)
    #     done_batch = batch[6].unsqueeze(-1).to(device=DEVICE)
    #
    #     new_probs, _ = actor(assignment_embedding= parent_state_batch.clone(), prob=None, mask=parent_state_mask_batch.clone())
    #     new_log_probs = torch.log(new_probs.gather(2, action_batch.unsqueeze(-1)))
    #     """Need to find elite_action"""
    #
    #     elite_action = action_batch[torch.arange(TRAIN_BATCH), elite_index]
    #     elite_log_probs = torch.log(new_probs[torch.arange(TRAIN_BATCH), elite_index, elite_action]).unsqueeze(1)
    #     #elite_log_prob_list = torch.cat((elite_log_prob_list, elite_log_probs), dim=1)
    #
    #     # elite_action_par_loss = -1*elite_log_prob_list.sum(dim=-1)
    #     e_loss = -elite_log_probs.mean()
    #     actor.optimizer.zero_grad()
    #     e_loss.backward()
    #     actor.optimizer.step()







    for time_index in range(MAX_TIME*NUM_WEAPONS):

        batch = storage[time_index]
        next_state_batch = batch[0].to(device=DEVICE)
        parent_state_batch = batch[1].to(device=DEVICE)
        action_batch = batch[2].to(device=DEVICE)
        reward_batch = batch[3].unsqueeze(-1).to(device=DEVICE)
        parent_state_mask_batch = batch[4].to(device=DEVICE)
        next_state_mask_batch = batch[5].to(device=DEVICE)
        done_batch = batch[6].unsqueeze(-1).to(device=DEVICE)

        # current_state_value = critic(state=parent_state_batch.detach().clone(), mask=parent_state_mask_batch.detach().clone())
        # small_number = 1e-10
        # zero_batches = torch.all(next_state_mask_batch == 0, dim=(2))
        # next_state_mask_batch[zero_batches] += small_number
        # next_state_value = critic(state=next_state_batch, mask=next_state_mask_batch)
        #
        # # expected value: reward + next state ---> this should be state value
        # expected_values = reward_batch + next_state_value.reshape(next_state_value.size(0), next_state_value.size(1), -1) * (done_batch.float())
        # advantages = expected_values.squeeze().detach() - current_state_value
        # squared_advantages = advantages.pow(2)
        # sum_over_parallel = squared_advantages.sum(dim=1)
        # critic_loss = sum_over_parallel.mean()
        #
        # critic.optimizer.zero_grad()
        # critic_loss.backward()
        # critic.optimizer.step()
        # Load the state_dict to copy the parameters

        old_actor.load_state_dict(actor.state_dict())
        # Freeze the parameters of old_actor
        for param in old_actor.parameters():
            param.requires_grad = False

        for _ in range(PPO_EPOCHS):

            with torch.no_grad():
                old_probs, _ = old_actor(assignment_embedding=parent_state_batch, prob=None, mask=parent_state_mask_batch)
                old_log_probs = torch.log(old_probs.gather(2, action_batch.unsqueeze(-1)))

            # Calculate probabilities with the current actor for gradient updates
            new_probs, _ = actor(assignment_embedding= parent_state_batch.clone(), prob=None, mask=parent_state_mask_batch.clone())

            new_log_probs = torch.log(new_probs.gather(2, action_batch.unsqueeze(-1)))
            """Need to find elite_action"""

            # elite_action = action_batch[torch.arange(TRAIN_BATCH), elite_index]
            # elite_log_probs = torch.log(new_probs[torch.arange(TRAIN_BATCH), elite_index, elite_action])

            # Calculate the ratio of new to old probabilities
            ratios = torch.exp(new_log_probs - old_log_probs).squeeze(-1)
            # Compute clipped objective
            surr1 = ratios * group_advantage
            surr2 = torch.clamp(ratios, 1 - 0.15, 1 + 0.15) * group_advantage
            min_values = torch.min(surr1, surr2)
            sum_parallel = min_values.sum(dim=1)

            # print("sum_parallel",sum_parallel)
            # a = input()
            loss = -sum_parallel.mean()
            actor_loss = loss
            actor.optimizer.zero_grad()
            actor_loss.backward()
            actor.optimizer.step()

        # current_state_value = critic(state=parent_state_batch.detach().clone(), mask=parent_state_mask_batch.detach().clone())
        # small_number = 1e-10
        # zero_batches = torch.all(next_state_mask_batch == 0, dim=(2))
        # next_state_mask_batch[zero_batches] += small_number
        # next_state_value = critic(state=next_state_batch, mask=next_state_mask_batch)
        #
        # # expected value: reward + next state ---> this should be state value
        # expected_values = reward_batch + next_state_value.reshape(next_state_value.size(0), next_state_value.size(1), -1) * (done_batch.float())
        # advantages = expected_values.squeeze().detach() - current_state_value
        # squared_advantages = advantages.pow(2)
        # sum_over_parallel = squared_advantages.sum(dim=1)
        # critic_loss = sum_over_parallel.mean()
        #
        # critic.optimizer.zero_grad()
        # critic_loss.backward()
        # critic.optimizer.step()

    actor.memory.refresh()
    #
    # # recording
    # min_reward, _ = obj.min(dim=1)
    # distance_min.push(min_reward)
    # mean_reward = obj.mean(dim=1, keepdim=True)
    # distance_mean.push(mean_reward)
    #
    # time_str = time.strftime("%H:%M:%S", time.gmtime(time.time() - timer_start))
    # log_str = 'EPOCH:{:03d}-EPISODE:{:03d}--Actor_loss:{:5f}--Critic_loss:{:5f}--Max_reward:{:5f}--Mean_reward:{:5f})'.format(epoch, episode, actor_loss, critic_loss, distance_min.result(), distance_mean.result())
    # logger.info(log_str)
    # logger_start = time.time()
    # actor.lr_stepper.step()
    #

    if epoch >=1 and episode==5:  # NUMBER_OF_UPDATES-1:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        checkpoint_folder_path = '{}/CheckPoint_epoch{:05d}'.format(result_folder_path, epoch)
        os.mkdir(checkpoint_folder_path)
        model_save_path = '{}/ACTOR_state_dic.pt'.format(checkpoint_folder_path)
        torch.save(actor.state_dict(), model_save_path)
        model_save_path = '{}/Critic_state_dic.pt'.format(checkpoint_folder_path)
        torch.save(critic.state_dict(), model_save_path)

    return total_reward, obj_function_value
