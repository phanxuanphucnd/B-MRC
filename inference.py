# -*- coding: utf-8 -*-
# Copyright (c) 2022 by Phuc Phan

import os
import torch
import utils
import timeit
import argparse

from typing import Any
from constants import *
from model import BMRCBertModel
from torch.nn import functional as F
from transformers import BertTokenizer
from unicodedata import normalize as unormalize


def preprocessing(input: str=None):
    input = unormalize('NFKC', input)
    input = input.lower().strip()

    return input


def infer(input: str=None, tokenizer: Any=None, model: Any=None, version: str=None, inference_beta: float=0.8):
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

    #TODO: Backward
    if args.version.lower() in ['bi', 'bidirectional']:
        b_opi_query = f"[CLS] What opinions ? [SEP] {input}".split()
        backward_opi_query_seg = [0]*(len(b_opi_query) - len(word_list)) + [1]*len(word_list)
        backward_opi_query = tokenizer.convert_tokens_to_ids(
            [word.lower() if word not in ['[CLS]', '[SEP]'] else word for word in b_opi_query]
        )
        backward_opi_query_mask = [1 for i in range(len(b_opi_query))]

        backward_opi_query = torch.tensor(backward_opi_query).unsqueeze(0).long().cuda()
        backward_opi_query_seg = torch.tensor(backward_opi_query_seg).unsqueeze(0).long().cuda()
        backward_opi_query_mask = torch.tensor(backward_opi_query_mask).unsqueeze(0).float().cuda()

        b_opi_start_scores, b_opi_end_scores = model(
            backward_opi_query,
            backward_opi_query_mask,
            backward_opi_query_seg, 0)

        b_opi_start_scores = F.softmax(b_opi_start_scores[0], dim=1)
        b_opi_end_scores = F.softmax(b_opi_end_scores[0], dim=1)
        b_opi_start_prob, b_opi_start_ind = torch.max(b_opi_start_scores, dim=1)
        b_opi_end_prob, b_opi_end_ind = torch.max(b_opi_end_scores, dim=1)

        b_opi_start_prob_temp = []
        b_opi_end_prob_temp = []
        b_opi_start_index_temp = []
        b_opi_end_index_temp = []
        for i in range(b_opi_start_ind.size(0)):
            if b_opi_start_ind[i].item() == 1:
                b_opi_start_index_temp.append(i)
                b_opi_start_prob_temp.append(b_opi_start_prob[i].item())
            if b_opi_end_ind[i].item() == 1:
                b_opi_end_index_temp.append(i)
                b_opi_end_prob_temp.append(b_opi_end_prob[i].item())

        b_opi_start_index, b_opi_end_index, b_opi_prob = utils.filter_unpaired(
            b_opi_start_prob_temp, b_opi_end_prob_temp, b_opi_start_index_temp, b_opi_end_index_temp)

        for i in range(len(b_opi_start_index)):
            aspect_query = tokenizer.convert_tokens_to_ids(
                [word.lower() if word not in ['[CLS]', '[SEP]'] else word
                 for word in '[CLS] What aspect does the opinion'.split()]
            )
            for j in range(b_opi_start_index[i], b_opi_end_index[i] + 1):
                aspect_query.append(backward_opi_query[0][j].item())
            aspect_query.append(tokenizer.convert_tokens_to_ids('describe'))
            aspect_query.append(tokenizer.convert_tokens_to_ids('?'))
            aspect_query.append(tokenizer.convert_tokens_to_ids('[SEP]'))
            aspect_query_seg = [0] * len(aspect_query)
            b_asp_length = len(aspect_query)
            aspect_query = torch.tensor(aspect_query).long().cuda()
            aspect_query = torch.cat([aspect_query, backward_opi_query[0][5:]], -1).unsqueeze(0)
            aspect_query_seg += [1] * backward_opi_query[0][5:].size(0)
            aspect_query_mask = torch.ones(aspect_query.size(1)).float().cuda().unsqueeze(0)
            aspect_query_seg = torch.tensor(aspect_query_seg).long().cuda().unsqueeze(0)

            b_asp_start_scores, b_asp_end_scores = model(aspect_query, aspect_query_mask, aspect_query_seg, 0)

            b_asp_start_scores = F.softmax(b_asp_start_scores[0], dim=1)
            b_asp_end_scores = F.softmax(b_asp_end_scores[0], dim=1)
            b_asp_start_prob, b_asp_start_ind = torch.max(b_asp_start_scores, dim=1)
            b_asp_end_prob, b_asp_end_ind = torch.max(b_asp_end_scores, dim=1)

            b_asp_start_prob_temp = []
            b_asp_end_prob_temp = []
            b_asp_start_index_temp = []
            b_asp_end_index_temp = []
            for k in range(b_asp_start_ind.size(0)):
                if aspect_query_seg[0, k] == 1:
                    if b_asp_start_ind[k].item() == 1:
                        b_asp_start_index_temp.append(k)
                        b_asp_start_prob_temp.append(b_asp_start_prob[k].item())
                    if b_asp_end_ind[k].item() == 1:
                        b_asp_end_index_temp.append(k)
                        b_asp_end_prob_temp.append(b_asp_end_prob[k].item())

            b_asp_start_index, b_asp_end_index, b_asp_prob = utils.filter_unpaired(
                b_asp_start_prob_temp, b_asp_end_prob_temp, b_asp_start_index_temp, b_asp_end_index_temp)

            for idx in range(len(b_asp_start_index)):
                opi = [backward_opi_query[0][j].item() for j in
                       range(b_opi_start_index[i], b_opi_end_index[i] + 1)]
                asp = [aspect_query[0][j].item() for j in range(b_asp_start_index[idx], b_asp_end_index[idx] + 1)]
                asp_ind = [b_asp_start_index[idx] - b_asp_length, b_asp_end_index[idx] - b_asp_length]
                opi_ind = [b_opi_start_index[i] - 5, b_opi_end_index[i] - 5]
                temp_prob = b_asp_prob[idx] * b_opi_prob[i]
                if asp_ind + opi_ind not in backward_pair_ind_list:
                    backward_pair_list.append([asp] + [opi])
                    backward_pair_prob.append(temp_prob)
                    backward_pair_ind_list.append(asp_ind + opi_ind)

    # print(f"backward_pair_list: {backward_pair_list}")
    # print(f"backward_pair_prob: {backward_pair_prob}")
    # print(f"backward_pair_ind_list: {backward_pair_ind_list}")

    #TODO: Filter
    for idx in range(len(forward_pair_list)):
        if forward_pair_list[idx] in backward_pair_list:
            if forward_pair_list[idx][0] not in final_asp_list:
                final_asp_list.append(forward_pair_list[idx][0])
                final_opi_list.append([forward_pair_list[idx][1]])
                final_asp_ind_list.append(forward_pair_ind_list[idx][:2])
                final_opi_ind_list.append([forward_pair_ind_list[idx][2:]])
            else:
                asp_index = final_asp_list.index(forward_pair_list[idx][0])
                if forward_pair_list[idx][1] not in final_opi_list[asp_index]:
                    final_opi_list[asp_index].append(forward_pair_list[idx][1])
                    final_opi_ind_list[asp_index].append(forward_pair_ind_list[idx][2:])
        else:
            if forward_pair_prob[idx] >= inference_beta:
                if forward_pair_list[idx][0] not in final_asp_list:
                    final_asp_list.append(forward_pair_list[idx][0])
                    final_opi_list.append([forward_pair_list[idx][1]])
                    final_asp_ind_list.append(forward_pair_ind_list[idx][:2])
                    final_opi_ind_list.append([forward_pair_ind_list[idx][2:]])
                else:
                    asp_index = final_asp_list.index(forward_pair_list[idx][0])
                    if forward_pair_list[idx][1] not in final_opi_list[asp_index]:
                        final_opi_list[asp_index].append(forward_pair_list[idx][1])
                        final_opi_ind_list[asp_index].append(forward_pair_ind_list[idx][2:])

    if args.version.lower() in ['bi', 'bidirectional']:
        # backward
        for idx in range(len(backward_pair_list)):
            if backward_pair_list[idx] not in forward_pair_list:
                if backward_pair_prob[idx] >= inference_beta:
                    if backward_pair_list[idx][0] not in final_asp_list:
                        final_asp_list.append(backward_pair_list[idx][0])
                        final_opi_list.append([backward_pair_list[idx][1]])
                        final_asp_ind_list.append(backward_pair_ind_list[idx][:2])
                        final_opi_ind_list.append([backward_pair_ind_list[idx][2:]])
                    else:
                        asp_index = final_asp_list.index(backward_pair_list[idx][0])
                        if backward_pair_list[idx][1] not in final_opi_list[asp_index]:
                            final_opi_list[asp_index].append(backward_pair_list[idx][1])
                            final_opi_ind_list[asp_index].append(backward_pair_ind_list[idx][2:])

    # sentiment
    for idx in range(len(final_asp_list)):
        predict_opinion_num = len(final_opi_list[idx])
        sentiment_query = tokenizer.convert_tokens_to_ids(
            [word.lower() if word not in ['[CLS]', '[SEP]'] else word for word in
             '[CLS] What sentiment given the aspect'.split(' ')])
        sentiment_query += final_asp_list[idx]
        sentiment_query += tokenizer.convert_tokens_to_ids(
            [word.lower() for word in 'and the opinion'.split(' ')]
        )

        for idy in range(predict_opinion_num):
            sentiment_query += final_opi_list[idx][idy]
            if idy < predict_opinion_num - 1:
                sentiment_query.append(tokenizer.convert_tokens_to_ids('/'))

        sentiment_query.append(tokenizer.convert_tokens_to_ids('?'))
        sentiment_query.append(tokenizer.convert_tokens_to_ids('[SEP]'))

        sentiment_query_seg = [0] * len(sentiment_query)
        sentiment_query = torch.tensor(sentiment_query).long().cuda()
        sentiment_query = torch.cat([sentiment_query, forward_asp_query[0][5:]], -1).unsqueeze(0)
        sentiment_query_seg += [1] * forward_asp_query[0][5:].size(0)
        sentiment_query_mask = torch.ones(sentiment_query.size(1)).float().cuda().unsqueeze(0)
        sentiment_query_seg = torch.tensor(sentiment_query_seg).long().cuda().unsqueeze(0)

        sentiment_scores = model(sentiment_query, sentiment_query_mask, sentiment_query_seg, 1)
        sentiment_predicted = torch.argmax(sentiment_scores[0], dim=0).item()

        for idy in range(predict_opinion_num):
            asp_f = []
            asp_f.append(final_asp_ind_list[idx][0])
            asp_f.append(final_asp_ind_list[idx][1])
            if asp_f + [sentiment_predicted] not in asp_pol_predict:
                asp_pol_predict.append(asp_f + [sentiment_predicted])

    real_output = []
    for asp_pol in asp_pol_predict:
        real_output.append([convert_ids_to_text(asp_pol[: 2], input), ID2SENTIMENT[asp_pol[-1]]])

    return real_output


def convert_ids_to_text(ids, text):
    return ' '.join(text.split()[ids[0]: ids[1]])


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
    parser.add_argument('--model_file_path', type=str, default='./models/14rest/bi_best_model.pt',
                        help='Path to the pretrained model.')

    args = parser.parse_args()

    tokenizer = BertTokenizer.from_pretrained(args.model_type)
    model = BMRCBertModel(args)
    print(f'Loading model path: `{args.model_file_path}`.')
    checkpoint = torch.load(args.model_file_path)
    model.load_state_dict(checkpoint['net'])
    model = model.cuda()

    args.input = "Owner is pleasant and entertaining ."

    if not args.test_path and not args.input:
        raise ValueError(f"Must be a given text input or file path input.")

    if args.input:
        start_time = timeit.default_timer()
        output = infer(args.input, tokenizer, model, args.version, args.inference_beta)
        end_time = timeit.default_timer()
        print(f"Output: {output}")
        print(f"Inference time: {(end_time - start_time) * 1000} ms.")

    if args.test_path:
        with open(args.test_path, 'r', encoding='utf-8') as f:
            text_lines = [line.split('####')[0].strip() for line in f.readlines()]

        inference_times = []
        for text in text_lines:
            start_time = timeit.default_timer()
            output = infer(args.input, tokenizer, model, args.version, args.inference_beta)
            end_time = timeit.default_timer()
            inference_times.append((end_time - start_time) * 1000)

        print(f"Inference time average in Test: {sum(inference_times) / len(inference_times)} ms.")
