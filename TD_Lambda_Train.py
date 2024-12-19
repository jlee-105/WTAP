import torch
import torch.optim as optim
import torch.nn as nn
import numpy as np
import os
import random
import time
from matplotlib import pyplot as plt

from utilities import Get_Logger
from Dynamic_HYPER_PARAMETER import *
from TORCH_OBJECTS import *
import DWTA_BHGT as MCTS_Model
import Dynamic_Sampling_BHGT as Sampler
import DWTA_Evaluation
import pickle

# Constants for eligibility traces
GAMMA = 0.99
LAMBDA = 0.9

# Setup
SAVE_FOLDER_NAME = f'W{NUM_WEAPONS}T{NUM_TARGETS}{TOTAL_EPISODE}EPISODE_{TOTAL_EPISODE // UPDATE_PERIOD}Epoch_{MINI_BATCH}BatchSize{BUFFER_SIZE}Buffer_SIZE{UPDATE_PERIOD}Update_Period'
logger, result_folder_path = Get_Logger(SAVE_FOLDER_NAME)
mcts_model = MCTS_Model.ACTOR().to(DEVICE)
mcts_model.optimizer = optim.Adam(mcts_model.parameters(), lr=ACTOR_LEARNING_RATE, weight_decay=ACTOR_WEIGHT_DECAY)
#mcts_model.lr_stepper = lr_scheduler.MultiStepLR(mcts_model.optimizer, milestones=[150], gamma=0.1)
timer_start = time.time()
EVAL_RESULT = []
Epoch_Train_Reward = []
Epoch_Train_Obj = []

# Initialize eligibility traces
eligibility_traces = {param: torch.zeros_like(param, device=DEVICE) for param in mcts_model.parameters()}

# Training Loop
for episode in range(1, TOTAL_EPISODE + 1):
    total_reward, train_obj_value = Sampler.self_play(mcts_model=mcts_model, episode=episode, temp=None)
    Epoch_Train_Reward.append(total_reward)
    Epoch_Train_Obj.append(train_obj_value)

    if episode % UPDATE_PERIOD == 0:
        mcts_model.train()
        samples = mcts_model.memory.sample(len(mcts_model.memory))
        random.shuffle(samples)

        for time_index in range(MAX_TIME):
            batch = list(zip(*samples[time_index * MINI_BATCH:(time_index + 1) * MINI_BATCH]))
            next_state_batch = torch.stack(batch[0]).to(DEVICE)
            parent_state_batch = torch.stack(batch[1]).to(DEVICE)
            action_batch = torch.stack(batch[2]).to(DEVICE)
            reward_batch = torch.stack(batch[3]).unsqueeze(-1).to(DEVICE)
            parent_state_mask_batch = torch.stack(batch[4]).to(DEVICE)
            next_state_mask_batch = torch.stack(batch[5]).to(DEVICE)
            done_batch = torch.stack(batch[6]).unsqueeze(-1).to(DEVICE)

            probs, state_values = mcts_model(assignment_embedding =parent_state_batch, prob=None, mask=parent_state_mask_batch)
            action_log_probs = torch.log(probs.gather(1, action_batch.unsqueeze(1)))

            small_number = 1e-10
            zero_batches = torch.all(next_state_mask_batch == 0, dim=(1))
            next_state_mask_batch[zero_batches] += small_number

            _, next_state_value = mcts_model(assignment_embedding =next_state_batch, mask=next_state_mask_batch, prob=None)
            expected_values = reward_batch + GAMMA*next_state_value.reshape(next_state_value.size(0), -1) * (done_batch.float())
            advantages = expected_values - state_values.squeeze()

            mcts_model.memory.refresh()
            # Calculate losses
            critic_loss = (advantages.pow(2)).mean()
            actor_loss = -(action_log_probs * advantages.detach()).mean()
            loss = actor_loss + 0.5 * critic_loss


            # Reset gradients
            mcts_model.optimizer.zero_grad()

            # Calculate gradients
            loss.backward()

            # Update eligibility traces and parameters
            for param in mcts_model.parameters():
                if param.grad is not None:
                    eligibility_traces[param] = GAMMA * LAMBDA * eligibility_traces[param] + param.grad
                    param.grad = eligibility_traces[param]  # Use the eligibility trace as the gradient

            # Step the optimizer
            mcts_model.optimizer.step()
            # mcts_model.lr_stepper.step()

            p_loss = actor_loss
            v_loss = critic_loss
            e_loss = 0
            total_loss = v_loss + abs(p_loss) + e_loss

            if time_index == MAX_TIME - 1:
                time_str = time.strftime("%H:%M:%S", time.gmtime(time.time() - timer_start))
                log_str_1 = 'EPOCH OLD_TRAIN REWARD:{:03f}--EPOCH OBJECTIVE :{:03F})'.format(
                    sum(Epoch_Train_Reward) / len(Epoch_Train_Reward), sum(Epoch_Train_Obj) / len(Epoch_Train_Obj))
                log_str_2 = 'EPISODE:{:03d}--UPDATE_PERIOD:{:03d}-BATCH:{:03d}-EPOCH:{:03d}(T:{:s}  actor_loss:{:5f} value_loss:{:5f} mcts_loss:{:5f} total_loss:{:5f})'.format(
                    episode, UPDATE_PERIOD, time_index, int(episode / UPDATE_PERIOD), time_str, actor_loss, v_loss,
                    e_loss, total_loss)
                logger.info(log_str_1)
                logger.info(log_str_2)
                logger_start = time.time()
                # mcts_model.lr_stepper.step()
                #  Logger ----------------------------------------------------------------

        if TOTAL_EPISODE // UPDATE_PERIOD > 100:  # NUMBER_OF_UPDATES-1:
            checkpoint_folder_path = '{}/CheckPoint_epoch{:05d}'.format(result_folder_path, episode)
            os.mkdir(checkpoint_folder_path)
            model_save_path = '{}/ACTOR_state_dic.pt'.format(checkpoint_folder_path)
            torch.save(mcts_model.state_dict(), model_save_path)
            optimizer_save_path = '{}/OPTIM_state_dic.pt'.format(checkpoint_folder_path)
            torch.save(mcts_model.optimizer.state_dict(), optimizer_save_path)
            lr_stepper_save_path = '{}/LRSTEP_state_dic.pt'.format(checkpoint_folder_path)
#            torch.save(mcts_model.lr_stepper.state_dict(), lr_stepper_save_path)

        if episode % EVALUATION_PERIOD == 0:  # Assuming you want to train every 5 epochs
            evaluation_original = []
            evaluation_reduced = []
            evaluation_left_over = []
            evaluation_original_p = []
            evaluation_reduced_p = []
            evaluation_left_over_p = []
            use_predefined = 'False'

            import pandas as pd
            import ast

            file_name = './TEST_INSTANCE/5M_5N.xlsx'

            df = pd.read_excel(file_name)
            for i in range(30):
                if use_predefined == 'False':
                    V = ast.literal_eval(df.loc[i]['V'])
                    P = ast.literal_eval(df.loc[i]['P'])
                    TW = ast.literal_eval(df.loc[i]['TW'])
                    TW = np.array(TW)
                    prob_np = np.array(P)
                    # Get Self-play Sample: Same as N_WEAPON
                    # original_value, reduced_pecentage, left_over_value = DWTA_Evaluation.evaluation(mcts_model, value=V, prob=prob_np, TW=TW, episode=episode)
                    original_value_p, reduced_pecentage_p, left_over_value_p = DWTA_Evaluation.evaluation_pure(
                        mcts_model, value=V, prob=prob_np, TW=TW, episode=episode)
                    # print("ahahahah--------------------------------------------")
                    # # print("left_over", left_over_value)
                    # original_value_p, reduced_pecentage_p, left_over_value_p = DWTA_Evaluation.evaluation_pure(mcts_model, value=None, prob=None)
                    # original_value_p=0
                    # reduced_pecentage_p=0
                    # left_over_value_p=0
                else:
                    with open('../../Desktop/TEST_INSTANCE/tensors.pkl', 'rb') as f:
                        print("ahahahah--------------------------------------------")
                        value_, prob_ = pickle.load(f)
                        value = value_[i].to('cpu')
                        prob = prob_[i].to('cpu')

                        original_value, reduced_pecentage, left_over_value = DWTA_Evaluation.evaluation(mcts_model,
                                                                                                        value=value,
                                                                                                        prob=prob,
                                                                                                        eposide=episode)
                        original_value_p, reduced_pecentage_p, left_over_value_p = DWTA_Evaluation.evaluation_pure(
                            mcts_model, value=value, prob=prob, episode=episode)

                # evaluation_original.append(original_value)
                # evaluation_reduced.append(reduced_pecentage)
                # evaluation_left_over.append(left_over_value)

                evaluation_original_p.append(original_value_p)
                evaluation_reduced_p.append(reduced_pecentage_p)
                evaluation_left_over_p.append(left_over_value_p)

            ################### Record #####################################
            # original_mean = sum(evaluation_original) / len(evaluation_original)
            # reduced_mean = sum(evaluation_reduced) / len(evaluation_reduced)
            # left_over_mean = sum(evaluation_left_over) / len(evaluation_left_over)
            original_mean_p = sum(evaluation_original_p) / len(evaluation_original_p)
            reduced_mean_p = sum(evaluation_reduced_p) / len(evaluation_reduced_p)
            left_over_mean_p = sum(evaluation_left_over_p) / len(evaluation_left_over_p)
            logger.info('--------------------------------------------------------------------------')
            # log_str_1 = '  <<< EVAL after Epoch:{:03d} >>> original_value_mean:{:f} left_overmean:{:f}  reduced_percentage:{:f}'.format(episode, original_mean,  left_over_mean, reduced_mean,)
            # logger.info(log_str_1)
            # logger.info('--------------------------------------------------------------------------')
            log_str_2 = '  <<< EVAL after Epoch:{:03d} >>> original_value_mean:{:f} reduced_percentage:{:f}  left_over_mean:{:f}'.format(
                episode, original_mean_p, reduced_mean_p, left_over_mean_p)
            logger.info(log_str_2)
            logger.info('--------------------------------------------------------------------------')
            EVAL_RESULT.append(left_over_mean_p)

        # RESULT PLOT
        from matplotlib import pyplot as plt

        if isinstance(EVAL_RESULT, torch.Tensor):
            EVAL_RESULT = EVAL_RESULT.cpu().numpy()
        plt.plot(EVAL_RESULT)
        plt.grid(True)
        plt.savefig('{}/eval_result.jpg'.format(result_folder_path))

        # Logging and checkpointing
        logger.info(f'Episode: {episode}, Reward: {np.mean(Epoch_Train_Reward)}, Objective: {np.mean(Epoch_Train_Obj)}')

# Plotting results
plt.plot(EVAL_RESULT)
plt.xlabel('Episodes')
plt.ylabel('Evaluation Result')
plt.title('Performance over Time')
plt.grid(True)
plt.show()
