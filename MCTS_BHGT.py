import copy
import math
import numpy as np
from Dynamic_HYPER_PARAMETER import *
import gc
from Dynamic_Instance_generation import *


def ucb_score(parent, child, c_puct):
    """
    UCB score is selecting node until leaf node
    """
    C_puct = c_puct
    if child.visit_count ==0:
        puct_score = C_puct * child.prior * math.sqrt(parent.visit_count)
        #puct_score =0
    else:
        puct_score = child.state_value + C_puct * child.prior * math.sqrt(parent.visit_count) / (child.visit_count + 1)

    return puct_score


class Node:
    def __init__(self, prior, state):
        self.visit_count = 0
        self.prior = prior
        self.state_value = 0
        self.value_sum = 0
        self.immediate_reward = 0
        self.children = {}
        self.state = state
        self.index = None

    def expanded(self):
        return len(self.children) > 0

    # value calculation
    def value(self):
        if self.visit_count == 0:
            self.state_value = torch.tensor(0)
        else:
            self.state_value = self.value_sum / self.visit_count

    # action selection
    def get_q_value(self, env):
        # print("afafafaf", len(env.available_actions))
        # a = input()
        q_values = list([0]) * len(env.available_actions)
        n_visit = list([0]) * len(env.available_actions)
        reward = list([0]) * len(env.available_actions)

        # print("children", self.children)
        # a = input()

        #for idx, child in zip(self.children, self.children.values()):
        for idx, child in enumerate(self.children.values()):

            # print(idx)
            q_values[idx] = child.state_value
            reward[idx] = child.immediate_reward
            n_visit[idx] = child.visit_count

        q_values = np.array(q_values)
        n_visit = np.array(n_visit)
        q_values = torch.FloatTensor(q_values)
        n_visit = torch.FloatTensor(n_visit)
        return q_values, n_visit, reward

    def select_action(self, temperature):

        """
        Select action according to the visit count distribution and the temperature.
        """
        # children has two values: visit_count & value sum
        visit_counts = list([0])*(NUM_WEAPONS*NUM_TARGETS+1)
        # print(self.children.values())
        # a = input()
        for idx, child in zip(self.children, self.children.values()):
            # print(child.visit_count)
            visit_counts[idx] = child.visit_count
        visit_counts = np.array(visit_counts)
        actions = list(range(0,NUM_WEAPONS*NUM_TARGETS+1))

        # if temperature == 0:
        #     action = actions[np.argmax(visit_counts)]
        #     # visit_count_distribution = np.zeros_like(visit_counts)
        #     visit_count_distribution = visit_counts/sum(visit_counts)
        # elif temperature == float("inf"):
        #     action = np.random.choice(actions)
        #     visit_count_distribution = visit_counts ** (1 / temperature)
        #     visit_count_distribution = visit_count_distribution / sum(visit_count_distribution)
        # else:
        #     visit_count_distribution = visit_counts ** (1 / temperature)
        #     visit_count_distribution = visit_count_distribution / sum(visit_count_distribution)
        #     action = np.random.choice(actions, p=visit_count_distribution)
        if temperature == 0:
            # Choose the action with the highest visit count deterministically
            action = actions[np.argmax(visit_counts)]
            visit_count_distribution = visit_counts / sum(visit_counts)
        elif temperature == float("inf"):
            # Choose an action uniformly at random
            action = np.random.choice(actions)
            visit_count_distribution = np.ones_like(visit_counts) / len(visit_counts)
        else:
            # Adjust visit counts by the temperature and sample from the resulting distribution
            visit_count_distribution = visit_counts ** (1 / temperature)
            visit_count_distribution = visit_count_distribution / sum(visit_count_distribution)
            action = np.random.choice(actions, p=visit_count_distribution)

        distribution = torch.FloatTensor(visit_count_distribution)

        return distribution, action

    # generate best action / child
    def select_child(self, c_puct):
        """
        Select the child with the highest UCB score.        """
        best_score = -np.inf
        best_action = -1
        best_child = None
        # unvisited_action = list()
        # unvisited_child = list()

        for action, child in self.children.items():
            score = ucb_score(self, child, c_puct=c_puct)

            if score > best_score:
                best_score = score
                best_action = action
                best_child = child
                # print("best_child", best_child)

        # if unvisited_action:
        #     best_action = random.sample(unvisited_action, 1)[0]
        #     best_child = unvisited_child[unvisited_action.index(best_action)]

        return best_action, best_child

    def expand(self, state, action_probs, value):
        """
        Expand a node and keep track of the prior policy probability given by neural network
        """
        # set

        self.state_value = value
        action_probs = action_probs.reshape(NUM_WEAPONS*NUM_TARGETS+1)
        for action in range(NUM_WEAPONS*NUM_TARGETS+1):
            if action_probs[action] != 0:
                self.children[action] = Node(prior=action_probs[action], state=state)



    def __repr__(self):
        """
        Debugger pretty print node info
        """
        prior = "{0:.4f}".format(self.prior)
        return "State:{},  Count: {}, Value_sum:{}, immediate_rewarde: {}, state_value:{}".format(self.state.__str__(),  self.visit_count, self.value_sum, self.immediate_reward, self.state_value)


class MCTS_SIM:

    def __init__(self, n_sim,  c_puct):
        self.self_play_state = None
        self.self_play_env = None
        self.self_play_model = None
        self.root_state = None
        self.mcts_actor = None
        self.c_env = None
        self.g_data = None
        self.simulation_env = None
        self.simulation_data = None
        self.n_sim = n_sim
        self.mcts_state = None
        self.c_puct = c_puct

    def simulation(self, env, model, mcts_state):

        # get model from main
        self.self_play_model = model
        # get main env
        self.self_play_env = env
        # get main state
        self.self_play_state = mcts_state

        # Root Node Creation with prior =0
        root = Node(prior=0, state = self.self_play_env.assignment_encoding.clone())
        # get probs, value from the main
        action_probs, value = self.self_play_model(assignment_embedding=self.self_play_state,
                                                   prob=self.self_play_env.weapon_to_target_prob,
                                                   mask=self.self_play_env.mask)

        # check initial possible actions and values
        action_probs = action_probs.detach()
        value = value.detach()
        # based on the actions probs, expand
        root.expand(state=root.state.clone(), action_probs=action_probs, value=value)

        # Start simulation -----------------------------------------------------------------------------------

        # copy simulation env
        self.simulation_env = copy.deepcopy(self.self_play_env)
        for trial in range(self.n_sim):

            # making root as node
            node = root
            # augment node in path
            search_paths = [node]

            while node.expanded():
                # action, node selected
                action, node = node.select_child(c_puct=self.c_puct)

                # print(action)
                # print(self.simulation_env.clock)
                # print(self.simulation_env.n_fires)
                # print(self.simulation_env.current_target_value)
                # a = input()

                before_value = self.simulation_env.current_target_value[:NUM_TARGETS].clone().sum()
                # updated all states
                num_actions = len(self.simulation_env.available_actions)

                """"   Simulation env update not"""
                print(num_actions)
                
                self.simulation_env.update_internal_variables(selected_action=action)


                node.state = self.simulation_env.assignment_encoding.clone()
                search_paths.append(node)
                if action < NUM_WEAPONS*NUM_TARGETS:
                    reward = (before_value -self.simulation_env.current_target_value[:NUM_TARGETS].clone().sum())/before_value
                else:
                    if num_actions > 1:
                        reward = torch.tensor(0.1).to(DEVICE)
                    else:
                        reward = torch.tensor(0.0).to(DEVICE)
                # If it is skip action, then reward =0
                # if before_value != self.simulation_env.current_target_value[:NUM_TARGETS].clone().sum():
                #     reward = 1-self.simulation_env.current_target_value[:NUM_TARGETS].clone().sum()/self.simulation_env.original_target_value[:NUM_TARGETS].clone().sum()
                # else:
                #      reward = torch.tensor(0)
                # child node has immediate reward
                node.immediate_reward = reward

                # if all weapons are fired including skip then, time update
                if self.simulation_env.n_fires % NUM_WEAPONS == 0:
                    self.simulation_env.time_update()

            # The leaf node is NOT the terminal then expand it, add values
            if self.simulation_env.n_fires < NUM_WEAPONS*MAX_TIME:
                # print("hehehe we stop")
                # a = input()
                # get action, value
                action_probs, value = self.self_play_model(assignment_embedding=self.simulation_env.assignment_encoding,
                                                           prob=self.simulation_env.weapon_to_target_prob,
                                                           mask=self.simulation_env.mask)
                node.expand(state=self.simulation_env.assignment_encoding, action_probs=action_probs, value=value)

            else:
                value = torch.tensor(0)

            """ TO DO LIST ----- DO'T DO THE """
            self.backpropagate(search_path=search_paths, value_nn=value)

        del self.simulation_env
        gc.collect()


        return root

    def backpropagate(self, search_path, value_nn):

        for i, node in enumerate(reversed(search_path)):
            if i==0:
                # Initialize leaf node: its intrinsic value
                node.value_sum = value_nn  # Assume node.value() returns the intrinsic state value, e.g., V(2) = 30
                node.visit_count = 1
            else:
                child = search_path[len(search_path)-i]
                node.value_sum += child.state_value + child.immediate_reward
                node.visit_count += 1

            node.state_value = node.value_sum / node.visit_count




