from Dynamic_HYPER_PARAMETER import *
import random
import torch
import numpy as np


def input_generation(NUM_WEAPON, NUM_TARGET, value, prob, TW, max_time, batch_size):

    # weapon to target prob container
    weapon_to_target_prob = list()
    # assignment encoding
    assignment_encoding = torch.zeros((batch_size, NUM_WEAPON * NUM_TARGETS+1, 9)).to(DEVICE)


    # t_value containers
    t_value_scaled = torch.zeros((NUM_TARGETS,1)).to(DEVICE)
    t_value = torch.zeros((NUM_TARGETS, 1)).to(DEVICE)
    # target emerge time containers
    time_target_emerging = torch.zeros((NUM_TARGETS, 1)).to(DEVICE)
    # active target time window containers
    time_target_remove = torch.zeros((NUM_TARGETS, 1)).to(DEVICE)
    # number of targets containers
    number_of_fired_on_target = torch.zeros((NUM_TARGETS, 1)).to(DEVICE)

    # Train
    if value is None:

        for batch in range(assignment_encoding.size(0)):

            index = 0

            # target related parameter
            for target_index in range(NUM_TARGETS):
                # target related parameter
                t_value[target_index] = random.randint(MIN_TARGET_VALUE, MAX_TARGET_VALUE)/MAX_TARGET_VALUE
                t_value_scaled[target_index] = t_value.clone()[target_index].unsqueeze(0)
                time_target_emerging[target_index] = random.randint(0, 3)/MAX_TIME
                time_target_remove[target_index] = MAX_TIME/MAX_TIME

            # weapon related parameter
            for w in range(NUM_WEAPON):
                for t in range(NUM_TARGETS):
                    # weapon availability initial set to 1.0
                    weapon_available = torch.tensor([1.0]).to(DEVICE)
                    # weapon waiting time initial set to Preparation Time
                    weapon_waiting_time = torch.tensor([0.0]).to(DEVICE)
                    # number of ammunition
                    remaining_ammunition = torch.tensor([AMM[w]]).to(DEVICE)/max(AMM)
                    # remaining time
                    remaining_time = torch.tensor([max_time]).to(DEVICE)/MAX_TIME

                    temp = np.random.uniform(low=LOW_PROB, high=HIGH_PROB)
                    # p_list.append(temp)
                    probability = torch.tensor([temp]).to(DEVICE)
                    # = probability/torch.max(probability)
                    target_value = torch.tensor([t_value_scaled[t]]).to(DEVICE)
                    time_target_start = time_target_emerging[t]
                    time_target_end = time_target_remove[t]
                    n_of_fire_on_target = number_of_fired_on_target[t]/(NUM_WEAPONS*MAX_TIME)

                    # final encoding
                    assignment_encoding[batch, index] = torch.cat((remaining_ammunition, weapon_available, remaining_time, weapon_waiting_time, n_of_fire_on_target, time_target_start, time_target_end, target_value, probability), dim=0).to(DEVICE)
                    weapon_to_target_prob.append(temp)
                    index = index + 1

            time_window_NO_action = torch.tensor([0]).to(DEVICE)
            time_target_start_NO_ACTION = torch.tensor([0]).to(DEVICE)
            Number_of_fired_NO_ACTION = torch.tensor([max_time]).to(DEVICE)/MAX_TIME
            left_over_time_NO_ACTION = torch.tensor([0]).to(DEVICE)
            weapon_available_NO_ACTION = torch.tensor([0.0]).to(DEVICE)
            wait_to_ready_NO_ACTION = torch.tensor([0.0]).to(DEVICE)
            target_value_NO_ACTION = torch.tensor([0.0]).to(DEVICE)
            probability_NO_ACTION = torch.tensor([0.0]).to(DEVICE)
            active_time_NO_ACTION = torch.tensor([0.0]).to(DEVICE)
            assignment_encoding[batch, -1] = torch.cat((time_window_NO_action, weapon_available_NO_ACTION, left_over_time_NO_ACTION, wait_to_ready_NO_ACTION, Number_of_fired_NO_ACTION, time_target_start_NO_ACTION,active_time_NO_ACTION,  target_value_NO_ACTION, probability_NO_ACTION), dim=0).to(DEVICE)


        weapon_to_target_prob = torch.FloatTensor(weapon_to_target_prob).to(DEVICE)
        weapon_to_target_prob = weapon_to_target_prob.reshape(assignment_encoding.size(0), NUM_WEAPON, NUM_TARGET)

    # TO DO THIS PART---------------------------------------------
    else:
        for batch in range(VAL_BATCH):

            # target_related value
            for target_index in range(NUM_TARGETS):
                t_value[target_index] = value[target_index]/MAX_TARGET_VALUE
                t_value_scaled[target_index] = t_value[target_index]
                time_target_emerging[target_index] = TW[target_index][0]/MAX_TIME
                time_target_remove[target_index] = TW[target_index][1]/MAX_TIME

            index = 0
            for w in range(NUM_WEAPON):
                for t in range(NUM_TARGETS):
                    weapon_available = torch.tensor([1.0]).to(DEVICE)
                    # weapon waiting time initial set to Preparation Time
                    weapon_waiting_time = torch.tensor([0.0]).to(DEVICE)
                    # number of ammunition
                    remaining_ammunition = torch.tensor([AMM[w]]).to(DEVICE)/max(AMM)
                    # remaining time
                    remaining_time = torch.tensor([max_time]).to(DEVICE)/MAX_TIME

                    # temp = np.random.uniform(low=LOW_PROB, high=HIGH_PROB)
                    # p_list.append(temp)
                    # probability = torch.tensor([temp]).to(DEVICE)
                    # = probability/torch.max(probability)
                    target_value = torch.tensor([t_value_scaled[t]]).to(DEVICE)
                    time_target_start = time_target_emerging[t]
                    time_target_end = time_target_remove[t]
                    n_of_fire_on_target = number_of_fired_on_target[t]/(NUM_WEAPONS*MAX_TIME)
                    probability = torch.tensor([prob[w,t]], dtype=torch.float32).to(DEVICE)

                    assignment_encoding[batch, index] = torch.cat((remaining_ammunition, weapon_available, remaining_time, weapon_waiting_time, n_of_fire_on_target, time_target_start, time_target_end, target_value, probability), dim=0).to(DEVICE)
                    index = index + 1

            weapon_to_target_prob = prob
            weapon_to_target_prob = torch.tensor(weapon_to_target_prob, dtype=torch.float32).to(DEVICE)

            time_window_NO_action = torch.tensor([0]).to(DEVICE)
            time_target_start_NO_ACTION = torch.tensor([0]).to(DEVICE)
            Number_of_fired_NO_ACTION = torch.tensor([0]).to(DEVICE)
            left_over_time_NO_ACTION = torch.tensor([max_time]).to(DEVICE)/MAX_TIME
            weapon_available_NO_ACTION = torch.tensor([0.0]).to(DEVICE)
            wait_to_ready_NO_ACTION = torch.tensor([0.0]).to(DEVICE)
            target_value_NO_ACTION = torch.tensor([0.0]).to(DEVICE)
            probability_NO_ACTION = torch.tensor([0.0]).to(DEVICE)
            active_time_NO_ACTION = torch.tensor([0.0]).to(DEVICE)
            assignment_encoding[batch, -1] = torch.cat((time_window_NO_action, weapon_available_NO_ACTION, left_over_time_NO_ACTION,
                                                wait_to_ready_NO_ACTION, Number_of_fired_NO_ACTION,
                                                time_target_start_NO_ACTION, active_time_NO_ACTION, target_value_NO_ACTION,
                                                probability_NO_ACTION), dim=0).to(DEVICE)

        #assignment_encoding[-1] = torch.cat((left_over_time_NO_ACTION, wait_to_ready_NO_ACTION, weapon_available_NO_ACTION, target_value_NO_ACTION, probability_NO_ACTION), dim=0).to(DEVICE)
            weapon_to_target_prob = weapon_to_target_prob.reshape(1, NUM_WEAPON, NUM_TARGET)

    return assignment_encoding, weapon_to_target_prob
