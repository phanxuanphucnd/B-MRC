# -*- coding: utf-8 -*-
# Copyright (c) 2022 by Phuc Phan

import os
import torch
import utils
import timeit
import argparse

from typing import Any
from model import BMRCBertModel
from torch.nn import functional as F
from transformers import BertTokenizer
from unicodedata import normalize as unormalize


def preprocessing(input: str=None):
    input = unormalize('NFKC', input)
    input = input.lower().strip()

    return input


def infer(input: str=None, tokenizer: Any=None, model: Any=None, version: str=None):
    model.eval()

    asp_predict = []
    opi_predict = []
    asp_opi_predict = []
    asp_pol_predict = []

    forward_pair_list = []
    forward_pair_prob = []
    forward_pair_ind_list = []

    backward_pair_list = []
    backward_pair_prob = []
    backward_pair_ind_list = []

    final_asp_list = []
    final_opi_list = []
    final_asp_ind_list = []
    final_opi_ind_list = []

    #TODO: Pre-processing
    input = preprocessing(input)
    word_list = input.split()

    max_forward_opi_query_length = 0
    max_backward_asp_query_length = 0

    #TODO: Forward
    f_asp_query = f"[CLS] What aspects ? [SEP] {input}".split()
    forward_asp_query_seg = [0]*(len(f_asp_query) - len(word_list)) + [1]*len(word_list)
    forward_asp_query = tokenizer.convert_tokens_to_ids(
        [word.lower() if word not in ['[CLS]', '[SEP]'] else word for word in f_asp_query]
    )
    forward_asp_query_mask = [1 for i in range(len(f_asp_query))]

    forward_asp_query = torch.tensor(forward_asp_query).unsqueeze(0).long().cuda()
    forward_asp_query_seg = torch.tensor(forward_asp_query_seg).unsqueeze(0).long().cuda()
    forward_asp_query_mask = torch.tensor(forward_asp_query_mask).unsqueeze(0).float().cuda()

    f_asp_start_scores, f_asp_end_scores = model(
        forward_asp_query,
        forward_asp_query_mask,
        forward_asp_query_seg, 0)

    f_asp_start_scores = F.softmax(f_asp_start_scores[0], dim=1)
    f_asp_end_scores = F.softmax(f_asp_end_scores[0], dim=1)
    f_asp_start_prob, f_asp_start_ind = torch.max(f_asp_start_scores, dim=1)
    f_asp_end_prob, f_asp_end_ind = torch.max(f_asp_end_scores, dim=1)

    f_asp_start_prob_temp = []
    f_asp_end_prob_temp = []
    f_asp_start_index_temp = []
    f_asp_end_index_temp = []

    for i in range(f_asp_start_ind.size(0)):
        if f_asp_start_ind[i].item() == 1:
            f_asp_start_index_temp.append(i)
            f_asp_start_prob_temp.append(f_asp_start_prob[i].item())
        if f_asp_end_ind[i].item() == 1:
            f_asp_end_index_temp.append(i)
            f_asp_end_prob_temp.append(f_asp_end_prob[i].item())

    f_asp_start_index, f_asp_end_index, f_asp_prob = utils.filter_unpaired(
        f_asp_start_prob_temp, f_asp_end_prob_temp, f_asp_start_index_temp, f_asp_end_index_temp)

    for i in range(len(f_asp_start_index)):
        opinion_query = tokenizer.convert_tokens_to_ids(
            [word.lower() if word in ['[CLS]', '[SEP]'] else word
             for word in '[CLS] what opinion given the aspect'.split()]
        )
        for j in range(f_asp_start_index[i], f_asp_end_index[i] + 1):
            opinion_query.append(forward_asp_query[0][j].item())

        opinion_query.append(tokenizer.convert_tokens_to_ids('?'))
        opinion_query.append(tokenizer.convert_tokens_to_ids('[SEP]'))
        opinion_query_seg = [0] * len(opinion_query)
        f_opi_length = len(opinion_query)
        opinion_query = torch.tensor(opinion_query).long().cuda()
        opinion_query = torch.cat([opinion_query, forward_asp_query[0][5:]], -1).unsqueeze(0)
        opinion_query_seg += [1]*forward_asp_query[0][5:].size(0)
        opinion_query_mask = torch.ones(opinion_query.size(1)).float().cuda().unsqueeze(0)
        opinion_query_seg = torch.tensor(opinion_query_seg).long().cuda().unsqueeze(0)

        f_opi_start_scores, f_opi_end_scores = model(opinion_query, opinion_query_mask, opinion_query_seg, 0)

        f_opi_start_scores = F.softmax(f_opi_start_scores[0], dim=1)
        f_opi_end_scores = F.softmax(f_opi_end_scores[0], dim=1)
        f_opi_start_prob, f_opi_start_ind = torch.max(f_opi_start_scores, dim=1)
        f_opi_end_prob, f_opi_end_ind = torch.max(f_opi_end_scores, dim=1)

        f_opi_start_prob_temp = []
        f_opi_end_prob_temp = []
        f_opi_start_index_temp = []
        f_opi_end_index_temp = []
        for k in range(f_opi_start_ind.size(0)):
            if opinion_query_seg[0, k] == 1:
                if f_opi_start_ind[k].item() == 1:
                    f_opi_start_index_temp.append(k)
                    f_opi_start_prob_temp.append(f_opi_start_prob[k].item())
                if f_opi_end_ind[k].item() == 1:
                    f_opi_end_index_temp.append(k)
                    f_opi_end_prob_temp.append(f_opi_end_prob[k].item())

        f_opi_start_index, f_opi_end_index, f_opi_prob = utils.filter_unpaired(
            f_opi_start_prob_temp, f_opi_end_prob_temp, f_opi_start_index_temp, f_opi_end_index_temp)

        for idx in range(len(f_opi_start_index)):
            asp = [forward_asp_query[0][j].item() for j in
                   range(f_asp_start_index[i], f_asp_end_index[i] + 1)]
            opi = [opinion_query[0][j].item() for j in range(f_opi_start_index[idx], f_opi_end_index[idx] + 1)]
            asp_ind = [f_asp_start_index[i] - 5, f_asp_end_index[i] - 5]
            opi_ind = [f_opi_start_index[idx] - f_opi_length, f_opi_end_index[idx] - f_opi_length]
            temp_prob = f_asp_prob[i] * f_opi_prob[idx]

            if asp_ind + opi_ind not in forward_pair_list:
                forward_pair_list.append([asp] + [opi])
                forward_pair_prob.append(temp_prob)
                forward_pair_ind_list.append(asp_ind + opi_ind)





    # Backward
    # backward_opi_query = f"[CLS] What opinions ? [SEP] {input}".split()
    # backward_opi_mask = [1 for i in range(len(forward_asp_query))]
    # backward_opi_query_seg = [0]*(len(backward_opi_query) - len(word_list)) + [1]*len(word_list)
    # backward_opi_query = tokenizer.convert_tokens_to_ids(
    #     [word.lower() if word not in ['[CLS]', '[SEP]'] else word in backward_opi_query]
    # )

    return None


def infers(file_input: str=None):

    return


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--test_path', type=str, default=None,
                        help='Path to the test file.')
    parser.add_argument('--input', type=str, default=None,
                        help='The text input.')
    parser.add_argument('--model_type', type=str, default="bert-base-uncased")
    parser.add_argument('--hidden_size', type=int, default=768)
    parser.add_argument('--inference_beta', type=float, default=0.8)
    parser.add_argument("--version", type=str, default='bidirectional',
                        choices=['uni', 'bi', 'unidirectional', 'bidirectional'],
                        help="`model_type` options in ['unidirectional', 'bidirectional'].")
    parser.add_argument('--model_file_path', type=str, default='./models/14rest/best_model.pt',
                        help='Path to the pretrained model.')

    args = parser.parse_args()

    tokenizer = BertTokenizer.from_pretrained(args.model_type)
    model = BMRCBertModel(args)
    print(f'Loading model path: `{args.model_file_path}`.')
    checkpoint = torch.load(args.model_file_path)
    model.load_state_dict(checkpoint['net'])
    model = model.cuda()

    args.input = "Ambiance and music funky , which I enjoy ."

    if not args.test_path and not args.input:
        raise ValueError(f"Must be a given text input or file path input.")

    if args.input:
        output = infer(args.input, tokenizer, model, args.version)

    if args.test_path:
        raise NotImplementedError()
