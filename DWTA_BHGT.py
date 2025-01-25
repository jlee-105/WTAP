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


def chunked_inference_mha_4d(q, k, v, chunk_size_q=32, chunk_size_k=32, mask=None):


    """
    Chunked multi-head attention for inference (no gradients),
    avoiding a giant (B, H, S, S) allocation.

    q, k, v: (B, H, S, D)
    chunk_size_q: how many query positions to process at once
    chunk_size_k: how many key positions to process at once
    mask (optional): broadcastable to (B, H, S, S)

    Returns:
      out: (B, H, S, D)
    """
    print("-------------------------------------------------")
    B, H, S, D = q.shape

    # We'll transpose K once => (B, H, D, S)
    k_t = k.transpose(-1, -2)
    # Prepare final output => (B, H, S, D)
    out = torch.zeros_like(q)

    for i_start in range(0, S, chunk_size_q):

        print("i_start", i_start)

        i_end = min(i_start + chunk_size_q, S)

        # q_chunk => (B, H, chunk_q, D)
        q_chunk = q[:, :, i_start:i_end, :]

        # We'll accumulate partial scores => shape (B, H, chunk_q, S)
        # but we won't store the entire (S, S) for all queries at once
        partial_scores = torch.empty(
            (B, H, i_end - i_start, S),
            device=q.device,
            dtype=q.dtype
        )

        # Fill partial_scores by chunking K as well
        for j_start in range(0, S, chunk_size_k):

            print("j_start", j_start)

            j_end = min(j_start + chunk_size_k, S)
            # k_t_chunk => (B, H, D, chunk_k)
            k_t_chunk = k_t[:, :, :, j_start:j_end]

            # partial => (B, H, chunk_q, chunk_k)
            partial = torch.matmul(q_chunk, k_t_chunk)
            partial_scores[..., j_start:j_end] = partial

        # Optional: scale scores by sqrt(D)
        partial_scores = partial_scores / (D ** 0.5)

        # If there's a mask, apply it
        if mask is not None:
            # Mask shape must broadcast to (B, H, chunk_q, S)
            partial_scores += mask[:, :, i_start:i_end, :]

        # Softmax over last dimension => attn_weights (B, H, chunk_q, S)
        attn_weights = F.softmax(partial_scores, dim=-1)

        # Multiply by V => out_chunk shape (B, H, chunk_q, D)
        out_chunk = torch.zeros(
            (B, H, i_end - i_start, D),
            device=q.device,
            dtype=q.dtype
        )

        for j_start in range(0, S, chunk_size_k):

            print("j_start", j_start)


            j_end = min(j_start + chunk_size_k, S)
            # slice attn_weights => (B, H, chunk_q, j_chunk)
            attn_slice = attn_weights[..., j_start:j_end]
            # slice v => (B, H, j_chunk, D)
            v_slice = v[:, :, j_start:j_end, :]

            out_chunk += torch.matmul(attn_slice, v_slice)

        # Place result into final out
        out[:, :, i_start:i_end, :] = out_chunk

    return out


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
        n_head = q.size(2)   # num of head
        n_batch = q.size(0)  # batch
        n_par = q.size(1)    #
        n_nodes = q.size(3)  # query size
        key_dim = q.size(4)  # key dimension
        mask = mask[:, :, None, None, :].expand(n_batch, n_par, n_head, NUM_WEAPONS * NUM_TARGETS + 1, NUM_WEAPONS * NUM_TARGETS + 1)

        if CHUNK == 'False':
            score = torch.matmul(q, k.transpose(3, 4))
            # mask = mask[:, :, None, None, :].expand(n_batch, n_par, n_head, NUM_WEAPONS * NUM_TARGETS+1, NUM_WEAPONS * NUM_TARGETS+1)
            score_scaled = score / np.sqrt(key_dim)
            masked_score = score_scaled.masked_fill(mask == 0, float('-inf'))
            weights = nn.Softmax(dim=4)(masked_score)
            out = torch.matmul(weights, v)
            out_concat = out.reshape(n_batch, n_par, n_nodes, n_head * key_dim)
        else:

        # ----------------------------------------------------------------------------------------------------------
            n_batch, n_par, n_heads, n_nodes, d_head = q.shape
            B = n_batch * n_par

            q_4d = q.reshape(B, n_heads, n_nodes, d_head)
            k_4d = k.reshape(B, n_heads, n_nodes, d_head)
            v_4d = v.reshape(B, n_heads, n_nodes, d_head)

            chunk_size_q = 32
            chunk_size_k = 32
            mask = mask.reshape(mask.size(0)*mask.size(1), mask.size(2), mask.size(3), mask.size(4))
            out_4d = chunked_inference_mha_4d(q_4d, k_4d, v_4d, chunk_size_q=chunk_size_q, chunk_size_k=chunk_size_k, mask=mask)
            out_concat = out_4d.reshape(n_batch, n_par, n_nodes, n_head*key_dim)

            print("I am here------------------------------------------------------------------")


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