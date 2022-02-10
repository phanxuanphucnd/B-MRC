#-*- coding: utf-8 -*-

import os
import torch
import argparse
from tqdm import tqdm
from transformers import  BertTokenizer
from dataset import DualSample, TokenizedSample, OriginalDataset


def tokenize_data(data, mode='train'):
    max_forward_asp_query_length = 0
    max_forward_opi_query_length = 0
    max_backward_asp_query_length = 0
    max_backward_opi_query_length = 0
    max_sentiment_query_length = 0

    max_aspect_num = 0
    max_opinion_num = 0
    tokenized_sample_list = []

    header_fmt = 'Tokenize data {:>5s}'
    for sample in tqdm(data, desc=f"{header_fmt.format(mode.upper())}"):
        forward_queries = []
        forward_answers = []
        backward_queries = []
        backward_answers = []
        sentiment_queries = []
        sentiment_answers = []

        forward_queries_seg = []
        backward_queries_seg = []
        sentiment_queries_seg = []

        if int(len(sample.forward_queries) - 1) > max_aspect_num:
            max_aspect_num = int(len(sample.forward_queries) - 1)
        if int(len(sample.backward_queries) - 1) > max_opinion_num:
            max_opinion_num = int(len(sample.backward_queries) - 1)

        for idx in range(len(sample.forward_queries)):
            temp_query = sample.forward_queries[idx]
            temp_text = sample.text
            temp_answer = sample.forward_answers[idx]
            temp_query_to = ['[CLS]'] + temp_query + ['[SEP]'] + temp_text
            temp_query_seg = [0] * (len(temp_query) + 2) + [1] * len(temp_text)
            temp_answer[0] = [-1] * (len(temp_query) + 2) + temp_answer[0]
            temp_answer[1] = [-1] * (len(temp_query) + 2) + temp_answer[1]

            assert len(temp_answer[0]) == len(temp_answer[1]) == len(temp_query_to) == len(temp_query_seg)

            if idx == 0:
                if len(temp_query_to) > max_forward_asp_query_length:
                    max_forward_asp_query_length = len(temp_query_to)
            else:
                if len(temp_query_to) > max_forward_opi_query_length:
                    max_forward_opi_query_length = len(temp_query_to)

            forward_queries.append(temp_query_to)
            forward_answers.append(temp_answer)
            forward_queries_seg.append(temp_query_seg)

        for idx in range(len(sample.backward_queries)):
            temp_query = sample.backward_queries[idx]
            temp_text = sample.text
            temp_answer = sample.backward_answers[idx]
            temp_query_to = ['[CLS]'] + temp_query + ['[SEP]'] + temp_text
            temp_query_seg = [0] * (len(temp_query) + 2) + [1] * len(temp_text)
            temp_answer[0] = [-1] * (len(temp_query) + 2) + temp_answer[0]
            temp_answer[1] = [-1] * (len(temp_query) + 2) + temp_answer[1]

            assert len(temp_answer[0]) == len(temp_answer[1]) == len(temp_query_to) == len(temp_query_seg)

            if idx == 0:
                if len(temp_query_to) > max_backward_opi_query_length:
                    max_backward_opi_query_length = len(temp_query_to)
            else:
                if len(temp_query_to) > max_backward_asp_query_length:
                    max_backward_asp_query_length = len(temp_query_to)

            backward_queries.append(temp_query_to)
            backward_answers.append(temp_answer)
            backward_queries_seg.append(temp_query_seg)

        for idx in range(len(sample.sentiment_queries)):
            temp_query = sample.sentiment_queries[idx]
            temp_text = sample.text
            temp_answer = sample.sentiment_answers[idx]
            temp_query_to = ['[CLS]'] + temp_query + ['[SEP]'] + temp_text
            temp_query_seg = [0] * (len(temp_query) + 2) + [1] * len(temp_text)

            assert len(temp_query_to) == len(temp_query_seg)

            if len(temp_query_to) > max_sentiment_query_length:
                max_sentiment_query_length = len(temp_query_to)

            sentiment_queries.append(temp_query_to)
            sentiment_answers.append(temp_answer)
            sentiment_queries_seg.append(temp_query_seg)

        temp_sample = TokenizedSample(sample.original_sample, forward_queries, forward_answers, backward_queries,
                                       backward_answers, sentiment_queries, sentiment_answers, forward_queries_seg,
                                       backward_queries_seg, sentiment_queries_seg)
        # print(temp_sample)
        tokenized_sample_list.append(temp_sample)

    max_attributes = {
        'mfor_asp_len': max_forward_asp_query_length,
        'mfor_opi_len': max_forward_opi_query_length,
        'mback_asp_len': max_backward_asp_query_length,
        'mback_opi_len': max_backward_opi_query_length,
        'max_sent_len': max_sentiment_query_length,
        'max_aspect_num': max_aspect_num,
        'max_opinion_num': max_opinion_num
    }
    return tokenized_sample_list, max_attributes


def preprocessing(sample_list, max_len, mode='train'):
    _tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    _forward_asp_query = []
    _forward_opi_query = []
    _forward_asp_answer_start = []
    _forward_asp_answer_end = []
    _forward_opi_answer_start = []
    _forward_opi_answer_end = []
    _forward_asp_query_mask = []
    _forward_opi_query_mask = []
    _forward_asp_query_seg = []
    _forward_opi_query_seg = []

    _backward_asp_query = []
    _backward_opi_query = []
    _backward_asp_answer_start = []
    _backward_asp_answer_end = []
    _backward_opi_answer_start = []
    _backward_opi_answer_end = []
    _backward_asp_query_mask = []
    _backward_opi_query_mask = []
    _backward_asp_query_seg = []
    _backward_opi_query_seg = []

    _sentiment_query = []
    _sentiment_answer = []
    _sentiment_query_mask = []
    _sentiment_query_seg = []

    _aspect_num = []
    _opinion_num = []

    header_fmt = 'Preprocessing {:>5s}'
    for instance in tqdm(sample_list, desc=f"{header_fmt.format(mode.upper())}"):
        f_query_list = instance.forward_queries
        f_answer_list = instance.forward_answers
        f_query_seg_list = instance.forward_seg
        b_query_list = instance.backward_queries
        b_answer_list = instance.backward_answers
        b_query_seg_list = instance.backward_seg
        s_query_list = instance.sentiment_queries
        s_answer_list = instance.sentiment_answers
        s_query_seg_list = instance.sentiment_seg

        # _aspect_num: 1/2/3/...
        _aspect_num.append(int(len(f_query_list) - 1))
        _opinion_num.append(int(len(b_query_list) - 1))

        # Forward
        # Aspect
        # query
        assert len(f_query_list[0]) == len(f_answer_list[0][0]) == len(f_answer_list[0][1])
        f_asp_pad_num = max_len['mfor_asp_len'] - len(f_query_list[0])

        _forward_asp_query.append(_tokenizer.convert_tokens_to_ids(
            [word.lower() if word not in ['[CLS]', '[SEP]'] else word for word in f_query_list[0]]))
        _forward_asp_query[-1].extend([0] * f_asp_pad_num)

        # query_mask
        _forward_asp_query_mask.append([1 for i in range(len(f_query_list[0]))])
        _forward_asp_query_mask[-1].extend([0] * f_asp_pad_num)

        # answer
        _forward_asp_answer_start.append(f_answer_list[0][0])
        _forward_asp_answer_start[-1].extend([-1] * f_asp_pad_num)
        _forward_asp_answer_end.append(f_answer_list[0][1])
        _forward_asp_answer_end[-1].extend([-1] * f_asp_pad_num)

        # seg
        _forward_asp_query_seg.append(f_query_seg_list[0])
        _forward_asp_query_seg[-1].extend([1] * f_asp_pad_num)

        # Opinion
        single_opinion_query = []
        single_opinion_query_mask = []
        single_opinion_query_seg = []
        single_opinion_answer_start = []
        single_opinion_answer_end = []
        for i in range(1, len(f_query_list)):
            assert len(f_query_list[i]) == len(f_answer_list[i][0]) == len(f_answer_list[i][1])
            pad_num = max_len['mfor_opi_len'] - len(f_query_list[i])
            # query
            single_opinion_query.append(_tokenizer.convert_tokens_to_ids(
                [word.lower() if word not in ['[CLS]', '[SEP]'] else word for word in f_query_list[i]]))
            single_opinion_query[-1].extend([0] * pad_num)

            # query_mask
            single_opinion_query_mask.append([1 for i in range(len(f_query_list[i]))])
            single_opinion_query_mask[-1].extend([0] * pad_num)

            # query_seg
            single_opinion_query_seg.append(f_query_seg_list[i])
            single_opinion_query_seg[-1].extend([1] * pad_num)

            # answer
            single_opinion_answer_start.append(f_answer_list[i][0])
            single_opinion_answer_start[-1].extend([-1] * pad_num)
            single_opinion_answer_end.append(f_answer_list[i][1])
            single_opinion_answer_end[-1].extend([-1] * pad_num)

        # PAD: max_aspect_num
        _forward_opi_query.append(single_opinion_query)
        _forward_opi_query[-1].extend(
            [[0 for i in range(max_len['mfor_opi_len'])]] * (max_len['max_aspect_num'] - _aspect_num[-1]))

        _forward_opi_query_mask.append(single_opinion_query_mask)
        _forward_opi_query_mask[-1].extend(
            [[0 for i in range(max_len['mfor_opi_len'])]] * (max_len['max_aspect_num'] - _aspect_num[-1]))

        _forward_opi_query_seg.append(single_opinion_query_seg)
        _forward_opi_query_seg[-1].extend(
            [[0 for i in range(max_len['mfor_opi_len'])]] * (max_len['max_aspect_num'] - _aspect_num[-1]))

        _forward_opi_answer_start.append(single_opinion_answer_start)
        _forward_opi_answer_start[-1].extend(
            [[-1 for i in range(max_len['mfor_opi_len'])]] * (max_len['max_aspect_num'] - _aspect_num[-1]))
        _forward_opi_answer_end.append(single_opinion_answer_end)
        _forward_opi_answer_end[-1].extend(
            [[-1 for i in range(max_len['mfor_opi_len'])]] * (max_len['max_aspect_num'] - _aspect_num[-1]))

        # Backward
        # opinion
        # query
        assert len(b_query_list[0]) == len(b_answer_list[0][0]) == len(b_answer_list[0][1])
        b_opi_pad_num = max_len['mback_opi_len'] - len(b_query_list[0])

        _backward_opi_query.append(_tokenizer.convert_tokens_to_ids(
            [word.lower() if word not in ['[CLS]', '[SEP]'] else word for word in b_query_list[0]]))
        _backward_opi_query[-1].extend([0] * b_opi_pad_num)

        # mask
        _backward_opi_query_mask.append([1 for i in range(len(b_query_list[0]))])
        _backward_opi_query_mask[-1].extend([0] * b_opi_pad_num)

        # answer
        _backward_opi_answer_start.append(b_answer_list[0][0])
        _backward_opi_answer_start[-1].extend([-1] * b_opi_pad_num)
        _backward_opi_answer_end.append(b_answer_list[0][1])
        _backward_opi_answer_end[-1].extend([-1] * b_opi_pad_num)

        # seg
        _backward_opi_query_seg.append(b_query_seg_list[0])
        _backward_opi_query_seg[-1].extend([1] * b_opi_pad_num)

        # Aspect
        single_aspect_query = []
        single_aspect_query_mask = []
        single_aspect_query_seg = []
        single_aspect_answer_start = []
        single_aspect_answer_end = []
        for i in range(1, len(b_query_list)):
            assert len(b_query_list[i]) == len(b_answer_list[i][0]) == len(b_answer_list[i][1])
            pad_num = max_len['mback_asp_len'] - len(b_query_list[i])
            # query
            single_aspect_query.append(_tokenizer.convert_tokens_to_ids(
                [word.lower() if word not in ['[CLS]', '[SEP]'] else word for word in b_query_list[i]]))
            single_aspect_query[-1].extend([0] * pad_num)

            # query_mask
            single_aspect_query_mask.append([1 for i in range(len(b_query_list[i]))])
            single_aspect_query_mask[-1].extend([0] * pad_num)

            # query_seg
            single_aspect_query_seg.append(b_query_seg_list[i])
            single_aspect_query_seg[-1].extend([1] * pad_num)

            # answer
            single_aspect_answer_start.append(b_answer_list[i][0])
            single_aspect_answer_start[-1].extend([-1] * pad_num)
            single_aspect_answer_end.append(b_answer_list[i][1])
            single_aspect_answer_end[-1].extend([-1] * pad_num)

        # PAD: max_opinion_num
        _backward_asp_query.append(single_aspect_query)
        _backward_asp_query[-1].extend(
            [[0 for i in range(max_len['mback_asp_len'])]] * (max_len['max_opinion_num'] - _opinion_num[-1]))

        _backward_asp_query_mask.append(single_aspect_query_mask)
        _backward_asp_query_mask[-1].extend(
            [[0 for i in range(max_len['mback_asp_len'])]] * (max_len['max_opinion_num'] - _opinion_num[-1]))

        _backward_asp_query_seg.append(single_aspect_query_seg)
        _backward_asp_query_seg[-1].extend(
            [[0 for i in range(max_len['mback_asp_len'])]] * (max_len['max_opinion_num'] - _opinion_num[-1]))

        _backward_asp_answer_start.append(single_aspect_answer_start)
        _backward_asp_answer_start[-1].extend(
            [[-1 for i in range(max_len['mback_asp_len'])]] * (max_len['max_opinion_num'] - _opinion_num[-1]))
        _backward_asp_answer_end.append(single_aspect_answer_end)
        _backward_asp_answer_end[-1].extend(
            [[-1 for i in range(max_len['mback_asp_len'])]] * (max_len['max_opinion_num'] - _opinion_num[-1]))

        # Sentiment
        single_sentiment_query = []
        single_sentiment_query_mask = []
        single_sentiment_query_seg = []
        single_sentiment_answer = []
        for j in range(len(s_query_list)):
            sent_pad_num = max_len['max_sent_len'] - len(s_query_list[j])
            single_sentiment_query.append(_tokenizer.convert_tokens_to_ids(
                [word.lower() if word not in ['[CLS]', '[SEP]'] else word for word in s_query_list[j]]))
            single_sentiment_query[-1].extend([0] * sent_pad_num)

            single_sentiment_query_mask.append([1 for i in range(len(s_query_list[j]))])
            single_sentiment_query_mask[-1].extend([0] * sent_pad_num)

            # query_seg
            single_sentiment_query_seg.append(s_query_seg_list[j])
            single_sentiment_query_seg[-1].extend([1] * sent_pad_num)

            single_sentiment_answer.append(s_answer_list[j])

        _sentiment_query.append(single_sentiment_query)
        _sentiment_query[-1].extend(
            [[0 for i in range(max_len['max_sent_len'])]] * (max_len['max_aspect_num'] - _aspect_num[-1]))

        _sentiment_query_mask.append(single_sentiment_query_mask)
        _sentiment_query_mask[-1].extend(
            [[0 for i in range(max_len['max_sent_len'])]] * (max_len['max_aspect_num'] - _aspect_num[-1]))

        _sentiment_query_seg.append(single_sentiment_query_seg)
        _sentiment_query_seg[-1].extend(
            [[0 for i in range(max_len['max_sent_len'])]] * (max_len['max_aspect_num'] - _aspect_num[-1]))

        _sentiment_answer.append(single_sentiment_answer)
        _sentiment_answer[-1].extend([-1] * (max_len['max_aspect_num'] - _aspect_num[-1]))

    result = {
        "_forward_asp_query": _forward_asp_query,
        "_forward_opi_query": _forward_opi_query,
        "_forward_asp_answer_start": _forward_asp_answer_start,
        "_forward_asp_answer_end": _forward_asp_answer_end,
        "_forward_opi_answer_start": _forward_opi_answer_start,
        "_forward_opi_answer_end": _forward_opi_answer_end,
        "_forward_asp_query_mask": _forward_asp_query_mask,
        "_forward_opi_query_mask": _forward_opi_query_mask,
        "_forward_asp_query_seg": _forward_asp_query_seg,
        "_forward_opi_query_seg": _forward_opi_query_seg,
        "_backward_asp_query": _backward_asp_query,
        "_backward_opi_query": _backward_opi_query,
        "_backward_asp_answer_start": _backward_asp_answer_start,
        "_backward_asp_answer_end": _backward_asp_answer_end,
        "_backward_opi_answer_start": _backward_opi_answer_start,
        "_backward_opi_answer_end": _backward_opi_answer_end,
        "_backward_asp_query_mask": _backward_asp_query_mask,
        "_backward_opi_query_mask": _backward_opi_query_mask,
        "_backward_asp_query_seg": _backward_asp_query_seg,
        "_backward_opi_query_seg": _backward_opi_query_seg,
        "_sentiment_query": _sentiment_query,
        "_sentiment_answer": _sentiment_answer,
        "_sentiment_query_mask": _sentiment_query_mask,
        "_sentiment_query_seg": _sentiment_query_seg,
        "_aspect_num": _aspect_num,
        "_opinion_num": _opinion_num
    }

    return OriginalDataset(result)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--data_path', type=str, default='./data/14lap/preprocess',
                        help='Path to the processed data from `data_process.py`')
    parser.add_argument('--output_path', type=str, default='./data/14lap/preprocess',
                        help='Path to the saved data.')
    parser.add_argument("--a2o", action='store_true',
                        help='Setup mode forward Aspect to Opinions')
    parser.add_argument("--o2a", action='store_true',
                        help='Setup mode backward Opinion to Aspects')

    args = parser.parse_args()

    train_data_path = f"{args.data_path}/train_DUAL.pt"
    dev_data_path = f"{args.data_path}/dev_DUAL.pt"
    test_data_path = f"{args.data_path}/test_DUAL.pt"

    train_data = torch.load(train_data_path)
    dev_data = torch.load(dev_data_path)
    test_data = torch.load(test_data_path)

    train_tokenized, train_max_len = tokenize_data(train_data, mode='train')
    dev_tokenized, dev_max_len = tokenize_data(dev_data, mode='dev')
    test_tokenized, test_max_len = tokenize_data(test_data, mode='test')

    train_preprocess = preprocessing(train_tokenized, train_max_len, mode='train')
    dev_preprocess = preprocessing(dev_tokenized, dev_max_len, mode='dev')
    test_preprocess = preprocessing(test_tokenized, test_max_len, mode='test')

    if not os.path.exists(args.output_path):
        os.makedirs(args.output_path)
    output_path = f"{args.output_path}/data.pt"
    print(f"Saved data : `{output_path}`.")
    torch.save({
        'train': train_preprocess,
        'dev': dev_preprocess,
        'test': test_preprocess
    }, output_path)
