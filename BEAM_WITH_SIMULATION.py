from Dynamic_HYPER_PARAMETER import *
import torch


def expansion():

    return 0



class Beam_Search:

    def __init__(self, env, actor, available_actions):

        # assignment encoding
        self.beam_env = env
        # weapon-to-target probability (static)
        self.policy = actor
        # record weapon to target (dynamic)
        self.beam_available_actions = available_actions

        self.assignment_encoding = None
        self.weapon_to_target_prob = None
        self.weapon_to_target_assign = None
        self.current_target_value = None
        self.original_target_value = None
        self.possible_weapons = None
        self.available_actions = None
        self.mask = None
        self.weapon_availability = None
        self.amm_availability = None
        self.weapon_wait_time = None
        self.time_left = None
        self.target_availability = None
        self.target_start_time = None
        self.target_end_time = None
        self.n_target_hit = None
        self.clock = None
        self.n_fires = None

    def reset(self):

        possible_action_index = torch.where(self.beam_available_actions>0)
        batch_idx = possible_action_index[0]
        expanded_group_index = possible_action_index[2]
        unique_elements, counts = torch.unique(batch_idx, return_counts=True)
        max_count = counts.max()

        # reset all internal variables
        # 1) encoding
        self.assignment_encoding = self.beam_env.assignment_encoding.repeat(1, max_count, 1, 1)
        # 2) weapon-to-target probability
        self.weapon_to_target_prob = self.beam_env.weapon_to_target_prob.repeat(1, max_count, 1, 1)
        # 3) weapon-to-target assign
        self.weapon_to_target_assign = self.beam_env.weapon_to_target_assign.repeat(1, max_count, 1)
        # 4) current_target_value
        self.current_target_value = self.beam_env.current_target_value.repeat(1, max_count, 1)
        # 5) current_target_value
        self.original_target_value = self.beam_env.original_target_value.repeat(1, max_count, 1)
        # 6) possible_weapons
        self.possible_weapons = self.beam_env.possible_weapons.repeat(1, max_count, 1)
        # 7) available actions
        self.available_actions = self.beam_env.available_actions.repeat(1, max_count, 1)
        # 8) mask
        self.mask = self.beam_env.mask.repeat(1, max_count, 1)
        # 9) weapon availability
        self.weapon_availability = self.beam_env.weapon_availability.repeat(1, max_count, 1)
        # 10) ammunition availability
        self.amm_availability = self.beam_env.amm_availability.repeat(1, max_count, 1)
        # 11) waiting time for each weapon
        self.weapon_wait_time = self.beam_env.weapon_wait_time.repeat(1, max_count, 1)
        # 12) left time
        self.time_left = self.beam_env.time_left
        # 13) target_availability
        self.target_availability = self.beam_env.target_availability.repeat(1, max_count, 1)
        # 14) target_start_time
        self.target_start_time = self.beam_env.target_start_time.repeat(1, max_count, 1)
        # 15) target_end time
        self.target_end_time = self.beam_env.target_end_time.repeat(1, max_count, 1)
        # 16) target hit
        self.n_target_hit = self.beam_env.n_target_hit.repeat(1, max_count, 1)
        # 17) clock
        self.clock = self.beam_env.clock
        # 18) n_fires
        self.n_fires = self.beam_env.n_fires

    def expand_actions(self):

        possible_action_index = torch.where(self.beam_available_actions > 0)
        print(possible_action_index)

        batch_idx = possible_action_index[0]
        node_index = possible_action_index[2]
        unique_elements, counts = torch.unique(batch_idx, return_counts=True)
        max_count = counts.max()

        selected_actions = torch.full(size=(self.beam_available_actions.size(0), max_count), fill_value=-1, dtype=torch.int64).to(DEVICE)

        group_index = torch.arange(0, max_count, dtype=torch.int64).to(DEVICE)
        print(group_index)
        group_index =  group_index.repeat(self.beam_available_actions.size(0))

        print(batch_idx.type())
        selected_actions[batch_idx, group_index] = node_index
        print(selected_actions.shape)

        self.update_internal_variables(selected_action=selected_actions)
        print("i am here")

        return 0




    def update_internal_variables(self, selected_action):

        batch_size = selected_action.size(0)
        para_size = selected_action.size(1)

        if (selected_action < NUM_WEAPONS * NUM_TARGETS).any():

            selected_action_index = torch.where(selected_action<NUM_WEAPONS*NUM_TARGETS)

            # print(selected_action_index)
            # a = input()

            # batch index, parallel index
            batch_id = selected_action_index[0]
            par_id = selected_action_index[1]

            # weapon, target index
            weapon_index = selected_action[batch_id, par_id] // NUM_TARGETS
            target_index = selected_action[batch_id, par_id] % NUM_TARGETS

            # print("weapon_index", weapon_index)
            # print("target_index", target_index)
            # print("self_a", self.available_actions.shape)
            # a = input()

            self.current_target_value = self.current_target_value.reshape(batch_size, para_size, NUM_WEAPONS, NUM_TARGETS)
            # dim [batch, par, weapon, target]
            reduced_values = self.current_target_value[batch_id, par_id, weapon_index, target_index] * self.weapon_to_target_prob[batch_id, par_id,weapon_index,target_index]
            # shape: dim selected actions
            reduced_values = reduced_values[:, None].expand(reduced_values.size(0), NUM_TARGETS)
            self.current_target_value[batch_id, par_id, :, target_index] = self.current_target_value[batch_id, par_id, :, target_index] - reduced_values.float()
            # reshape of current target value
            self.current_target_value = self.current_target_value.reshape(batch_size, para_size, NUM_WEAPONS*NUM_TARGETS)
            # additional weapon to target update
            self.weapon_to_target_assign[batch_id, par_id, weapon_index] = target_index

            # weapon wait_time -- update
            wait_time = torch.FloatTensor(PREPARATION_TIME[:NUM_WEAPONS]).to(DEVICE)
            wait_time = wait_time[None, None, :].expand(batch_size, para_size, wait_time.size(0))
            self.weapon_wait_time[batch_id, par_id, weapon_index] = wait_time[batch_id, par_id, weapon_index]

            # weapon availability update
            self.weapon_availability[(self.weapon_wait_time >0) | (self.amm_availability<=0)] = 0.0
            self.n_target_hit[batch_id, par_id, target_index] = self.n_target_hit.clone()[batch_id, par_id, target_index]+1

            # ammunition update
            self.amm_availability = self.amm_availability.clone()
            self.amm_availability[batch_id, par_id, weapon_index] -= 1

            # number of fires update
            self.n_fires = self.n_fires + 1

            # possible weapon update
            possible_weapons = torch.arange(0, NUM_WEAPONS).int().to(DEVICE)
            self.possible_weapons = possible_weapons[None, None, :].expand(batch_size, para_size, possible_weapons.size(0))
            possible_weapons_mask = (self.weapon_wait_time <= 0) & (self.amm_availability > 0)
            self.possible_weapons = possible_weapons_mask.long()

            # print("possible_weapon", self.possible_weapons.shape)
            # a = input()

            # update target availability
            target_availability = torch.arange(0, NUM_TARGETS).int().to(DEVICE)
            self.target_availability = target_availability[None, None, :].expand(batch_size, para_size, target_availability.size(0))
            # print("self.target_start_time", self.target_start_time)
            # print("self.target_end_time", self.target_end_time)
            # a = input()

            possible_target_mask = (self.target_start_time<= self.clock) & (self.target_end_time>= self.clock)
            self.target_availability = possible_target_mask.long()

            # print("possible_target", self.target_availability)

            # update available actions
            available_actions = torch.full(size=(batch_size, para_size, NUM_WEAPONS * NUM_TARGETS), fill_value=1.0).to(DEVICE)
            available_actions = available_actions.reshape(available_actions.size(0), available_actions.size(1), NUM_WEAPONS, NUM_TARGETS)

            # print("available_actions", available_actions)
            # a = input()

            if torch.any(self.possible_weapons <= 0):
                unavailable_weapon = torch.where((self.possible_weapons <= 0))
                unavailable_weapon_batch_index = unavailable_weapon[0]
                unavailable_weapon_par_index = unavailable_weapon[1]
                unavailable_weapon_w_index = unavailable_weapon[2]
                available_actions[unavailable_weapon_batch_index, unavailable_weapon_par_index, unavailable_weapon_w_index, :] = 0.0


            if torch.any((self.target_availability <= 0)):
                unavailable_target = torch.where((self.target_availability <= 0))
                unavailable_target_batch_index = unavailable_target[0]
                unavailable_target_par_index = unavailable_target[1]
                unavailable_target_t_index = unavailable_target[2]
                available_actions[unavailable_target_batch_index, unavailable_target_par_index, :, unavailable_target_t_index] = 0.0


            self.available_actions = available_actions.reshape(batch_size, para_size, -1).clone()
            # print(self.available_actions)
            # a = input()

            self.mask[:, :, :-1] = self.available_actions.clone()

            """ factors : additiona,  wait_to_ready, Number_of_fired, time_target_start, active_time, target_value, probability"""
            # State Update
            assignment_encoding_without_no_action = self.assignment_encoding[:, :, :-1 , :].detach().clone()
            assignment_encoding_without_no_action = assignment_encoding_without_no_action.reshape(batch_size, para_size, NUM_WEAPONS, NUM_TARGETS, -1)

            # remaining ammunition update
            remaining_ammunition = self.amm_availability.clone()[:, :, :, None].expand(batch_size, para_size, NUM_WEAPONS, NUM_TARGETS)
            assignment_encoding_without_no_action[batch_id, par_id, weapon_index, :, 0] = remaining_ammunition[batch_id, par_id, weapon_index, :].float()

            # weapon availability update
            assignment_encoding_without_no_action[batch_id, par_id, weapon_index, :, 1] = 0.0

            # number of fired
            n_target_hit = self.n_target_hit.clone()[:, :, None, :].expand(batch_size, para_size, NUM_WEAPONS, NUM_TARGETS)
            assignment_encoding_without_no_action[batch_id, par_id, :, target_index, 4] = n_target_hit[batch_id, par_id, :, target_index]

            # target value
            assignment_encoding_without_no_action[:, :, :, :, -2] = self.current_target_value.clone().reshape(batch_size, para_size, NUM_WEAPONS, NUM_TARGETS)/MAX_TARGET_VALUE

            # reshape
            self.assignment_encoding[:, :, :-1 , :] = assignment_encoding_without_no_action.reshape(batch_size, para_size, NUM_WEAPONS * NUM_TARGETS,-1)

        else:
            self.n_fires = self.n_fires + 1

        self.mask[:, :, -1] =1.0

        # print("self.available_actions", self.available_actions)
        # a = input()

    def time_update(self):

        # move one clock
        self.clock = self.clock + 1
        # weapon_wait_time update
        self.weapon_wait_time[self.weapon_wait_time>0] -= 1
        # update leftover time
        self.time_left = MAX_TIME - self.clock
        # # update self.target_active_window
        # self.target_active_window[self.target_active_window>0] -= 1
        batch_size = self.assignment_encoding.size(0)
        para_size = self.assignment_encoding.size(1)

        # check new target emerge
        # if torch.any(self.target_start_time == self.clock):
        #     new_target_location = torch.where(self.target_start_time == self.clock)
        #     max_time_window = torch.FloatTensor(MAX_TIME_WINDOW).to(DEVICE)
        #     max_time_window = max_time_window[None, None, :].expand(batch_size, para_size, max_time_window.size(0))
        #     self.target_active_window[new_target_location[0], new_target_location[1], new_target_location[2]] = max_time_window[new_target_location[0], new_target_location[1], new_target_location[2]]
        # else:
        #     pass

        # update weapon availability
        self.weapon_availability[(self.weapon_wait_time == 0) & (self.amm_availability>0)] = 1.0
        possible_weapons = torch.arange(0, NUM_WEAPONS).int().to(DEVICE)
        self.possible_weapons = possible_weapons[None, None, :].expand(batch_size, para_size, possible_weapons.size(0))
        possible_weapons_mask = (self.weapon_wait_time <= 0) & (self.amm_availability > 0)
        self.possible_weapons = possible_weapons_mask.long()


        target_availability = torch.arange(0, NUM_TARGETS).int().to(DEVICE)
        self.target_availability = target_availability[None, None, :].expand(batch_size, para_size, target_availability.size(0))
        #if torch.any(self.target_active_window > 0):
        possible_target_mask = (self.target_start_time <= self.clock) & (self.target_end_time >= self.clock)
        self.target_availability = possible_target_mask.long()


        # update available actions and mask based on the possible weapons and targets
        available_actions = torch.full(size=(batch_size, para_size, NUM_WEAPONS * NUM_TARGETS), fill_value=1.0).to(DEVICE)
        available_actions = available_actions.reshape(available_actions.size(0), available_actions.size(1), NUM_WEAPONS, NUM_TARGETS)


        if torch.any((self.possible_weapons <= 0)):
            unavailable_weapon = torch.where((self.possible_weapons <= 0))
            unavailable_weapon_batch_index, unavailable_weapon_par_index, unavailable_weapon_w_index  = unavailable_weapon[0], unavailable_weapon[1], unavailable_weapon[2]
            available_actions[unavailable_weapon_batch_index, unavailable_weapon_par_index, unavailable_weapon_w_index, :] = 0.0

        if torch.any((self.target_availability <= 0)):
            unavailable_target = torch.where((self.target_availability <= 0))
            unavailable_target_batch_index, unavailable_target_par_index, unavailable_target_t_index = unavailable_target[0], unavailable_target[1], unavailable_target[2]
            available_actions[unavailable_target_batch_index, unavailable_target_par_index, :,unavailable_target_t_index] = 0.0

        self.available_actions = available_actions.reshape(batch_size, para_size, -1).clone()
        self.mask[:, :, :-1] = self.available_actions.clone()

        # if self.clock == MAX_TIME:
        self.mask[:, :, -1] = 1.0
        # else:
        #     pass

        # state update ---------------------------------------------
        assignment_encoding_without_no_action = self.assignment_encoding[:, :, :-1, :].clone()
        assignment_encoding_without_no_action = assignment_encoding_without_no_action.reshape(batch_size, para_size, NUM_WEAPONS, NUM_TARGETS, -1)

        # available weapon update
        available_weapon_index = torch.where(self.weapon_availability==1)
        # Weapon availability update:
        assignment_encoding_without_no_action[available_weapon_index[0], available_weapon_index[1], available_weapon_index[2], :, 1] = 1.0

        # weapon wait time update
        updated_wait_time = self.weapon_wait_time.clone()[:, :, :,  None].expand(batch_size, para_size, NUM_WEAPONS, NUM_TARGETS)
        assignment_encoding_without_no_action[:, :, :, :, 3] = updated_wait_time

        # time left
        assignment_encoding_without_no_action[:, :, :, :, 2] = self.time_left

        self.assignment_encoding[:, :, :-1, :] = assignment_encoding_without_no_action.reshape(batch_size, para_size, NUM_WEAPONS * NUM_TARGETS, -1)


        return 0


    def do_beam_simulation(self):

        # set-up for simulation
        start_time = self.n_fires // MAX_TIME
        start_weapon = self.n_fires % MAX_TIME


        for time_clock in range(start_time, MAX_TIME):

            for index in range(start_weapon, NUM_WEAPONS):
                policy, _ = self.policy(assignment_embedding=self.assignment_encoding,  prob=self.weapon_to_target_prob, mask=self.mask)
                # action_index = torch.multinomial(policy.view(-1, NUM_WEAPONS * NUM_TARGETS + 1), 1).view(env_e.assignment_encoding.size(0), env_e.assignment_encoding.size(1))
                action_index = torch.multinomial(policy.view(-1, NUM_WEAPONS * NUM_TARGETS + 1), 1).view(self.assignment_encoding.size(0), self.assignment_encoding.size(1))
                self.update_internal_variables(selected_action=action_index)

            self.time_update()
            start_weapon = 0

        print("env.current_target_value", self.current_target_value)

        obj_value = (self.current_target_value[:, :, 0:NUM_TARGETS]).sum(2)
        print("env.n_fires", self.n_fires)
        # obj_value_ = obj_value.squeeze()
        # obj_values = torch.min(obj_value_)
        print("obj_value", obj_value)
        a = input()


        # select_best_action

        return obj_value