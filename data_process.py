# -*- coding: utf-8 -*-
# Copyright (c) 2022 by Phuc Phan

import os
import torch
import pickle
import argparse
from tqdm import  tqdm
from dataset import DualSample

def get_text(lines):
    text_list = []
    # aspect_list = []
    # opinion_list = []
    for line in lines:
        # temp = line.split('####')
        # assert len(temp) == 3

        word_list = line.split()
        # aspect_label_list = [t.split('=')[-1] for t in temp[1].split()]
        # opinion_label_list = [t.split('=')[-1] for t in temp[2].split()]
        # assert len(word_list) == len(aspect_label_list) == len(opinion_label_list)

        text_list.append(word_list)
        # aspect_list.append(aspect_label_list)
        # opinion_list.append(opinion_label_list)

    # return text_list, aspect_list, opinion_list
    return text_list


def valid_data(triplet, aspect, opinion):
    for t in triplet[0][0]:
        assert aspect[t] != ['O']

    for t in triplet[0][1]:
        assert opinion[t] != ['O']


def fusion_dual_triplet(triplet, backward=False):
    triplet_aspect = []
    triplet_opinion = []
    triplet_sentiment = []
    dual_opinion = []
    dual_aspect = []

    for t in triplet:
        if t[0] not in triplet_aspect:
            triplet_aspect.append(t[0])
            triplet_opinion.append([t[1]])
            triplet_sentiment.append(t[2])
        else:
            idx = triplet_aspect.index(t[0])
            triplet_opinion[idx].append(t[1])
            assert triplet_sentiment[idx] == t[2]

        if backward:
            if t[1] not in dual_opinion:
                dual_opinion.append(t[1])
                dual_aspect.append([t[0]])
            else:
                idx = dual_opinion.index(t[1])
                dual_aspect[idx].append(t[0])

    return triplet_aspect, triplet_opinion, triplet_sentiment, dual_opinion, dual_aspect


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument("--data_path", type=str, default='./data/14lap',
                        help="Path to the dataset.")
    parser.add_argument("--version", type=str, default='bidirectional',
                        choices=['uni', 'bi', 'unidirectional', 'bidirectional'],
                        help="`model_type` options in ['unidirectional', 'bidirectional'].")
    parser.add_argument('--output_path', type=str, default='./data/14lap/preprocess',
                        help='Path to the saved data.')

    args = parser.parse_args()

    DATASET_TYPE_LIST = ['train', 'dev', 'test']

    for dataset_type in DATASET_TYPE_LIST:
        #TODO: Read triple
        with open(f'{args.data_path}/{dataset_type}_pair.pkl', 'rb') as f:
            triple_data = pickle.load(f)

        #TODO: Read text
        with open(f'{args.data_path}/{dataset_type}.txt', 'r', encoding='utf-8') as f:
            text_lines = f.readlines()

        #TODO: Get text
        text_list = get_text(text_lines)
        # text_list, aspect_list, opinion_list = get_text(text_lines)

        sample_list = []
        header_fmt = 'Processing {:>5s}'
        for i in tqdm(range(len(text_list)), desc=f"{header_fmt.format(dataset_type.upper())}"):
            triplet = triple_data[i]
            text = text_list[i]
            #TODO: Valid data
            # valid_data(triplet, aspect_list[i], opinion_list[i])
            triplet_aspect, triplet_opinion, triplet_sentiment, dual_opinion, dual_aspect = fusion_dual_triplet(
                triplet,
                backward=args.version.lower() in ['bi', 'bidirectional']
            )

            forward_query_list = []
            forward_answer_list = []
            backward_query_list = []
            backward_answer_list = []
            sentiment_query_list = []
            sentiment_answer_list = []

            forward_query_list.append("What aspects ?".split())
            start = [0]*len(text)
            end = [0]*len(text)
            for ta in triplet_aspect:
                start[ta[0]] = 1
                end[ta[-1]] = 1
            forward_answer_list.append([start, end])

            for idx in range(len(triplet_aspect)):
                ta = triplet_aspect[idx]
                #TODO: Opinion query
                query = f"What opinion given the aspect {' '.join(text[ta[0]: ta[-1] + 1])} ?".split()
                forward_query_list.append(query)
                start = [0]*len(text)
                end = [0]*len(text)
                for to in triplet_opinion[idx]:
                    start[to[0]] = 1
                    end[to[-1]] = 1
                forward_answer_list.append([start, end])
                #TODO: Sentiment query
                temp_opinion = []
                for idy in range(len(triplet_opinion[idx]) - 1):
                    to = triplet_opinion[idx][idy]
                    temp_opinion.append(' '.join(text[to[0]: to[-1] + 1]) + ' /')
                to = triplet_opinion[idx][-1]
                temp_opinion.append(' '.join(text[to[0]: to[-1] + 1]))

                query = f"What sentiment given the aspect {' '.join(text[ta[0]: ta[-1] + 1])} and " \
                        f"the opinion {' '.join(temp_opinion)} ?".split()
                sentiment_query_list.append(query)
                sentiment_answer_list.append(triplet_sentiment[idx])

            if args.version.lower() in ['bi', 'bidirectional']:
                backward_query_list.append("What is opinions ?".split())
                start = [0]*len(text)
                end = [0]*len(text)
                for do in dual_opinion:
                    start[do[0]] = 1
                    end[do[-1]] = 1
                backward_answer_list.append([start, end])

                for idx in range(len(dual_opinion)):
                    do = dual_opinion[idx]
                    #TODO: Aspect query
                    query = f"What aspect does the opinion {' '.join(text[do[0]: do[-1] + 1])} describe ?".split()
                    backward_query_list.append(query)
                    start = [0]*len(text)
                    end = [0]*len(text)
                    for da in dual_aspect[idx]:
                        start[da[0]] = 1
                        end[da[-1]] = 1
                    backward_answer_list.append([start, end])

            sample = DualSample(
                text_lines[i],
                text,
                forward_query_list,
                forward_answer_list,
                backward_query_list,
                backward_answer_list,
                sentiment_query_list,
                sentiment_answer_list
            )
            sample_list.append(sample)

        #TODO: Storages samples to .pt file
        if not os.path.exists(args.output_path):
            os.makedirs(args.output_path)

        output_path = f"{args.output_path}/{dataset_type}_dual.pt"
        print(f"Saved data to `{output_path}`.")
        torch.save(sample_list, output_path)
