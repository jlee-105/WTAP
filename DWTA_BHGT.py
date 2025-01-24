""" This is python script, part of realization fo Reinforcement Learning
    * Target Problem: WEAPON-ASSIGNMENT
    * Author: Jaejin Lee
    * Date: 08/03/2023
    * This script should be enclosed for internal use and will not be distributed without
      agreement from Jaejin
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from Dynamic_HYPER_PARAMETER import *
from TORCH_OBJECTS import *
import torch
import torch.nn.functional as F
from torch_geometric.nn import HeteroConv, SAGEConv
from torch_geometric.data import HeteroData
import random
# from torch_geometric.nn import HGTConv, to_hetero
import gc


class EmbeddingLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.Wq = nn.Linear(EMBEDDING_DIM, HEAD_NUM * KEY_DIM, bias=False)
        self.Wk = nn.Linear(EMBEDDING_DIM, HEAD_NUM * KEY_DIM, bias=False)
        self.Wv = nn.Linear(EMBEDDING_DIM, HEAD_NUM * KEY_DIM, bias=False)
        self.Multi_head_combine = nn.Linear(HEAD_NUM * KEY_DIM, EMBEDDING_DIM)
        self.Add_n_normalization = InstnaceNormalization()
        self.feed_forward = FeedForwardModule()
        self.drop_layer = nn.Dropout(p=0.1)

    def forward(self, embedding, prob, mask):

        # self-attention b/w same nodes

        q = self.Wq(embedding)
        k = self.Wk(embedding)
        v = self.Wv(embedding)
        # dim: [batch, par, n_assignment, dim_emb]

        # reshape by head_number
        q_reshape = reshape_by_heads(q, head_num=HEAD_NUM)
        k_reshape = reshape_by_heads(k, head_num=HEAD_NUM)
        v_reshape = reshape_by_heads(v, head_num=HEAD_NUM)
        # dim: [batch, par, n_head, n_ass, n_h]

        out_concat = multi_head_attention(q=q_reshape, k=k_reshape, v=v_reshape, mask=mask)
        multi_head_out = self.Multi_head_combine(out_concat)

        out1 = self.Add_n_normalization(q, multi_head_out)
        out2 = self.feed_forward(out1)
        out3 = self.Add_n_normalization(out1, out2)

        return out3


class EmbederLayerBase(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer = EmbeddingLayer()

    def forward(self, assignment_embedding, prob, mask):
        assignment_embedding = self.layer(assignment_embedding, prob, mask)
        return assignment_embedding


class EmbedderBase(nn.Module):
    def __init__(self):
        super().__init__()

        # From raw input to encoding
        self.assignment_encoding = nn.Linear(9, EMBEDDING_DIM)
        # self.encoding_target = nn.Linear(NUM_TARGETS+1, EMBEDDING_DIM)
        self.embedding_layers = nn.ModuleList([EmbederLayerBase() for _ in range(ENCODER_LAYER_NUM)])

    def forward(self, assignment_embedding, prob, mask):
        # assignment_embedding = None
        assignment_embedding = self.assignment_encoding(assignment_embedding)

        for layers in self.embedding_layers:
            assignment_embedding = layers(assignment_embedding, prob, mask)


        return assignment_embedding



class ACTOR(nn.Module):

    def __init__(self):
        super().__init__()
        self.embedder = EmbedderBase()
        # self.gnn_embedder = GNN()        # different input embedder
        self.lstm = LSTM()
        self.mask = None
        self.assignment_embedding = None
        self.current_state = None
        self.replay_memory = ReplayMemory(capacity=BUFFER_SIZE)
        # self.skip_embedding = nn.Parameter(torch.randn(1, EMBEDDING_DIM))

    # def get_current_state(self, probability):
    #
    #     if probability.dim() == 2:
    #         weapon_expanded = self.weapon_embedding.unsqueeze(1).repeat(1, NUM_WEAPONS, 1)
    #         target_expanded = self.target_embedding.unsqueeze(0).repeat(NUM_TARGETS, 1, 1)
    #         self.assignment_embedding = torch.cat([weapon_expanded, target_expanded], dim=2)
    #         prob = probability.reshape(self.assignment_embedding.size(0), self.assignment_embedding.size(1))
    #         prob_reshaped = prob.unsqueeze(-1)
    #         current_state = torch.cat([self.assignment_embedding, prob_reshaped], dim=-1)
    #     else:
    #         weapon_expanded = self.weapon_embedding.unsqueeze(2).repeat(1, 1, NUM_TARGETS, 1)
    #         target_expanded = self.target_embedding.unsqueeze(1).repeat(1, NUM_WEAPONS, 1, 1)
    #         # Concatenate to form assignment_embedding
    #         assignment_embedding = torch.cat([weapon_expanded, target_expanded], dim=-1)
    #         # Reshape probability and add extra dimension
    #         prob_reshaped = probability.unsqueeze(-1)  # Shape: (BATCH_SIZE, NUM_WEAPONS, NUM_TARGETS, 1)
    #         # Concatenate to form current_state
    #         current_state = torch.cat([assignment_embedding, prob_reshaped], dim=-1)
    #
    #     return current_state

    def get_p_v_from_state(self, assignment_embedding, prob, mask):

        self.assignment_embedding = self.embedder(assignment_embedding=assignment_embedding, prob=prob, mask=mask)
        # self.weapon_embedding, self.target_embedding = self.gnn_embedder(weapon_input=weapon_input, target_input=target_input, prob=prob, mask=mask)

        self.current_state = self.assignment_embedding
        return self.assignment_embedding

    def forward(self, assignment_embedding, prob, mask):
        if assignment_embedding.dim() ==2:
            input_to_transformer = assignment_embedding[:, :]
        else:
            input_to_transformer = assignment_embedding


        # print("inopt", input_to_transformer.shape)
        self.assignment_embedding = self.get_p_v_from_state(assignment_embedding=input_to_transformer, prob=prob, mask=mask)

        # if self.assignment_embedding.dim() ==2:
        #     skip_added_embedding = torch.cat((self.assignment_embedding, self.skip_embedding), dim=0)
        # else:
        #     expanded_skip = self.skip_embedding.clone()[None, :, :].expand(self.assignment_embedding.size(0), self.skip_embedding.size(0), self.skip_embedding.size(1))
        #     skip_added_embedding = torch.cat((self.assignment_embedding, expanded_skip), dim=1)
            # print(skip_added_embedding.shape)
            # a = input()

        #self.assignment_embedding = skip_added_embedding.detach()

        value = self.lstm(self.assignment_embedding, mask)
        return value


class LSTM(nn.Module):
    def __init__(self):
        super(LSTM, self).__init__()
        #self.rnn = torch.nn.LSTM(HIDDEN_DIM, HIDDEN_DIM, num_layers=2, bias=True, batch_first=True, dropout=0.1,  bidirectional=False, proj_size=0, device=DEVICE, dtype=None)

        # head for policy
        self.policy_head_1 = nn.Linear(HIDDEN_DIM, 2*HIDDEN_DIM)  # Assuming a scalar output for simplicity
        self.policy_head_2 = nn.Linear(2*HIDDEN_DIM, HIDDEN_DIM)
        self.policy_head_3 = nn.Linear(HIDDEN_DIM, 1)

        """ New try"""
        # self.value_head_1 = nn.Linear(HIDDEN_DIM, 2*HIDDEN_DIM)
        # self.value_head_2 = nn.Linear(2*HIDDEN_DIM, HIDDEN_DIM)
        # self.value_head_3 = nn.Linear(HIDDEN_DIM, 1)
        #self.softmax_2 = nn.Softmax(dim=1)
        self.gelu = nn.GELU()
        self.Leaky = nn.LeakyReLU()
        self.relu = nn.ReLU()
        #self.sigmoid = nn.Sigmoid()

    def forward(self, state, mask):

        shared_result = state
        # dim = n_batch, EMBEDDING_DIM

        mask_c = mask.clone().bool()
        # value_nn = self.value_head_1(shared_result)
        # value_nn = self.relu(value_nn)
        # value_nn = self.value_head_2(value_nn)
        # value_nn = self.relu(value_nn)
        # value_nn = self.value_head_3(value_nn).reshape(state.size(0), state.size(1), NUM_WEAPONS * NUM_TARGETS+1)
        # value_nn = self.relu(value_nn)
        # value_nn = value_nn * mask_c.reshape(state.size(0), state.size(1), NUM_WEAPONS * NUM_TARGETS+1)
        # value_nn = torch.mean(value_nn, dim=-1)


        policy_nn = self.policy_head_1(shared_result)
        policy_nn = self.relu(policy_nn)
        policy_nn = self.policy_head_2(policy_nn)
        policy_nn = self.relu(policy_nn)
        policy_nn = self.policy_head_3(policy_nn).reshape(state.size(0), state.size(1), NUM_WEAPONS * NUM_TARGETS + 1)

        # policy_nn = self.softmax_2(policy_nn)
        policy_nn = policy_nn.masked_fill(~mask_c, float('-inf'))
        policy_nn = F.softmax(policy_nn, dim=-1)
        # a = input()

        return policy_nn,  shared_result


class Critic(nn.Module):
    def __init__(self):
        super(Critic, self).__init__()
        self.value_head_1 = nn.Linear(9, HIDDEN_DIM)
        self.value_head_2 = nn.Linear(HIDDEN_DIM, 2*HIDDEN_DIM)
        self.value_head_3 = nn.Linear(2*HIDDEN_DIM, HIDDEN_DIM)
        self.value_head_4 = nn.Linear(HIDDEN_DIM, 1)
        # self.softmax_2 = nn.Softmax(dim=1)
        self.gelu = nn.GELU()
        self.Leaky = nn.LeakyReLU()
        self.relu = nn.ReLU()

    def forward(self, state, mask):

        # shared_result, (h_n, c_n) = self.rnn(state)
        shared_result = state
        # dim = n_batch, EMBEDDING_DIM
        if shared_result.ndim == 2:
            value_nn = self.value_head_1(shared_result)
            value_nn = self.relu(value_nn)
            value_nn = self.value_head_2(value_nn)
            value_nn = self.relu(value_nn)
            value_nn = self.value_head_3(value_nn)
            value_nn = self.relu(value_nn)
            value_nn = self.value_head_4(value_nn).view(NUM_WEAPONS * NUM_TARGETS + 1)
            # value_nn = torch.mean(value_nn, dim=-1)
            value_nn = self.relu(value_nn)
            mask = mask.view(NUM_WEAPONS * NUM_TARGETS + 1)
            value_nn = value_nn * mask
            value_nn = torch.mean(value_nn, dim=-1)

        # dim = n_batch, EMBEDDING_DIM
        else:
            value_nn = self.value_head_1(shared_result)
            value_nn = self.relu(value_nn)
            value_nn = self.value_head_2(value_nn)
            value_nn = self.relu(value_nn)
            value_nn = self.value_head_3(value_nn)
            value_nn = self.relu(value_nn)
            value_nn = self.value_head_4(value_nn)
            value_nn = self.relu(value_nn)
            value_nn = value_nn.squeeze()
            value_nn = value_nn * mask.squeeze()
            value_nn = torch.mean(value_nn, dim=-1)

        return value_nn


def multi_head_attention(q, k, v, mask):

    # q, k, v: [number head, sequence size, key dim]
    if q.dim()==3:
        n_nodes = q.size(1)   # sequence size
        n_head = q.size(0)  # head size
        key_dim = q.size(2)   # query size
        score = torch.matmul(q, k.transpose(1,2))
        #mask = mask[:-1]

        mask = mask.reshape(NUM_WEAPONS*NUM_TARGETS+1)[None, None, :].expand(n_head, NUM_WEAPONS*NUM_TARGETS+1, NUM_WEAPONS*NUM_TARGETS+1)
        #mask = mask.reshape(NUM_WEAPONS * NUM_TARGETS)[None, None, :].expand(n_head, NUM_WEAPONS * NUM_TARGETS, NUM_WEAPONS * NUM_TARGETS)
        score_scaled = score / np.sqrt(key_dim)
        # print(score_scaled.shape)
        # print(mask.shape)
        # a =input()
        masked_score = score_scaled.masked_fill(mask == 0, float('-inf'))
        weights = nn.Softmax(dim=2)(masked_score)
        out = torch.matmul(weights, v)
        out_concat = out.reshape(n_nodes, n_head * key_dim)
    else:
        n_head = q.size(2)  # sequence size
        n_batch = q.size(0)  # head size
        n_par = q.size(1)
        n_nodes = q.size(3)  # query size
        key_dim = q.size(4)
        # score = torch.matmul(q, k.transpose(3, 4))
        #
        # #mask = mask[:, :-1]
        # # mask = mask.reshape(-1, NUM_WEAPONS * NUM_TARGETS)[:, None, None, :].expand(n_batch, n_head,
        # #                                                                                NUM_WEAPONS * NUM_TARGETS,
        # #                                                                                NUM_WEAPONS * NUM_TARGETS)
        #
        # mask = mask[:, :, None, None, :].expand(n_batch, n_par, n_head, NUM_WEAPONS * NUM_TARGETS+1, NUM_WEAPONS * NUM_TARGETS+1)
        # score_scaled = score / np.sqrt(key_dim)
        # masked_score = score_scaled.masked_fill(mask == 0, float('-inf'))
        #
        # weights = nn.Softmax(dim=4)(masked_score)
        # out = torch.matmul(weights, v)
        # out_concat = out.reshape(n_batch, n_par, n_nodes, n_head * key_dim)


        # ----------------------------------------------------------------------------------------------------------
        k_t = k.transpose(3, 4)
        score_list = []
        for q_chunk in torch.split(q, 300, dim=3):
            # q_chunk => [n_batch, n_par, n_head, chunk_size, key_dim]
            # matmul => [n_batch, n_par, n_head, chunk_size, n_nodes]
            score_chunk = torch.matmul(q_chunk, k_t)
            score_list.append(score_chunk)
        # Concatenate back along dim=3
        score = torch.cat(score_list, dim=3)  # => [n_batch, n_par, n_head, n_nodes, n_nodes]

        # 2) Expand mask to match [n_batch, n_par, n_head, n_nodes, n_nodes]
        mask = mask[:, :, None, None, :].expand(n_batch, n_par, n_head, n_nodes, n_nodes)

        # 3) Scale & mask
        score_scaled = score / np.sqrt(key_dim)
        masked_score = score_scaled.masked_fill(mask == 0, float('-inf'))

        # 4) Softmax over last dim (dim=4)
        weights = nn.Softmax(dim=4)(masked_score)  # => [n_batch, n_par, n_head, n_nodes, n_nodes]

        # 5) Chunked matmul: out = weights x v
        #    v => [n_batch, n_par, n_head, n_nodes, key_dim]
        out_list = []
        for w_chunk in torch.split(weights, 300, dim=3):
            # w_chunk => [n_batch, n_par, n_head, chunk_size, n_nodes]
            # matmul => [n_batch, n_par, n_head, chunk_size, key_dim]
            out_chunk = torch.matmul(w_chunk, v)
            out_list.append(out_chunk)
        out = torch.cat(out_list, dim=3)  # => [n_batch, n_par, n_head, n_nodes, key_dim]

        # 6) Final reshape
        out_concat = out.reshape(n_batch, n_par, n_nodes, n_head * key_dim)


    return out_concat


# Define the Replay Memory to store experiences
class ReplayMemory:
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    # Store a new experience in memory
    def push(self, transition):
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = transition
        self.position = (self.position + 1) % self.capacity

    # Randomly sample a batch of experiences from memory
    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __getitem__(self, index):
        # Allows direct access like memory[index]
        # Check for index validity
        if index < 0 or index >= len(self.memory):
            raise IndexError('Index out of bounds')
        return self.memory[index]

    def refresh(self):
        self.memory = []
        self.position = 0

    def __len__(self):
        return len(self.memory)


# class InstnaceNormalization(nn.Module):
#     def __init__(self):
#         super().__init__()
#
#         self.norm_by_batch  = nn.BatchNorm1d(EMBEDDING_DIM)
#
#
#     def forward(self, x1, x2):
#         x = x1+x2
#         # x shape: [batch, par, n_nodes, feature]
#         b, p, n, f = x.shape
#         # Step 1 & 2: merge (b * p) and move feature dimension to the "channels"
#         # After view: [b*p, n, f]
#         # After permute: [b*p, f, n]  (channel-first for BatchNorm1d)
#         x = x.view(b * p, n, f).permute(0, 2, 1)
#
#         # Step 3: apply normalization over the feature dimension
#         x = self.norm_by_batch(x)  # shape stays [b*p, f, n]
#
#         # Step 4: permute back to [b*p, n, f], then reshape to [b, p, n, f]
#         x = x.permute(0, 2, 1).view(b, p, n, f)
#
#         return x

class InstnaceNormalization(nn.Module):
    def __init__(self):
        super().__init__()

        self.norm_by_batch  = nn.BatchNorm1d(EMBEDDING_DIM)


    def forward(self, x1, x2):
        x = x1+x2
        # x shape: [batch, par, n_nodes, feature]
        b, p, n, f = x.shape
        # Step 1 & 2: merge (b * p) and move feature dimension to the "channels"
        # After view: [b*p, n, f]
        # After permute: [b*p, f, n]  (channel-first for BatchNorm1d)
        x = x.view(b * p, n, f).permute(0, 2, 1)

        # Step 3: apply normalization over the feature dimension
        x = self.norm_by_batch(x)  # shape stays [b*p, f, n]

        # Step 4: permute back to [b*p, n, f], then reshape to [b, p, n, f]
        x = x.permute(0, 2, 1).view(b, p, n, f)

        return x


class FeedForwardModule(nn.Module):
    def __init__(self):
        super().__init__()
        self.W1 = nn.Linear(EMBEDDING_DIM, FF_HIDDEN_DIM)
        self.W2 = nn.Linear(FF_HIDDEN_DIM, EMBEDDING_DIM)

    def forward(self, input1):
        # input.shape = (batch_s, TSP_SIZE, EMBEDDING_DIM)
        return self.W2(F.gelu(self.W1(input1)))


def reshape_by_heads(qkv, head_num):
    if qkv.dim() == 2:
        seq_size = qkv.size(0)
        q_reshaped = qkv.reshape(head_num, seq_size, -1)
    else:
        batch_size = qkv.size(0)
        par_size = qkv.size(1)
        seq_size = qkv.size(2)
        q_reshaped = qkv.reshape(batch_size, par_size, head_num, seq_size, -1)

    return q_reshaped