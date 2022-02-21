# -*- coding: utf-8 -*-
# Copyright (c) 2022 by Phuc Phan

import os
import ast
import torch
import pickle
import argparse
import constants

from tqdm import  tqdm


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument("--data_path", type=str, default='./data/raw/14lap',
                        help="Path to the dataset.")
    parser.add_argument('--output_path', type=str, default='./data/14lap',
                        help='Path to the saved data.')

    args = parser.parse_args()

    DATASET_TYPE_LIST = ['train', 'dev', 'test']

    for dataset_type in DATASET_TYPE_LIST:
        with open(f"{args.data_path}/{dataset_type}_triplets.txt", 'r', encoding='utf-8') as f:
            lines = [line.strip() for line in f.readlines()]

            data_pair = []
            texts = []

            for line in tqdm(lines, desc=f"Make {dataset_type} pair"):
                splited_line = line.split('####')

                texts.append(splited_line[0])
                triplets = ast.literal_eval(splited_line[1])

                temp_triplets = []
                for triplet in triplets:
                    tuple_triplet = (triplet[0], triplet[1], constants.SENTIMENT2ID[triplet[2]])
                    temp_triplets.append(tuple_triplet)

                data_pair.append(temp_triplets)

        if not os.path.exists(args.output_path):
            os.makedirs(args.output_path)

        assert len(texts) == len(data_pair)

        with open(f"{args.output_path}/{dataset_type}.txt", 'w', encoding='utf-8') as f:
            for text in tqdm(texts, desc=f"Write {dataset_type}.txt"):
                f.writelines(text + '\n')

        with open(f"{args.output_path}/{dataset_type}_pair.pkl", 'wb') as f:
            pickle.dump(data_pair, f)
