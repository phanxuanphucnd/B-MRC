# -*- coding: utf-8 -*-
# Copyright (c) 2022 by Phuc Phan

import  numpy as np
from typing import  Any
from torch.utils.data import Dataset, DataLoader


class DualSample(object):
    def __init__(
            self,
            original_sample: str=None,
            text: str=None,
            forward_queries: list=None,
            forward_answers: list=None,
            backward_queries: list=None,
            backward_answers: list=None,
            sentiment_queries: list=None,
            sentiment_answers: list=None
    ):
        self.original_sample = original_sample
        self.text = text
        self.forward_queries = forward_queries
        self.forward_answers = forward_answers
        self.backward_queries = backward_queries
        self.backward_answers = backward_answers
        self.sentiment_queries = sentiment_queries
        self.sentiment_answers = sentiment_answers


class TokenizedSample(object):
    def __init__(
            self,
            original_sample: Any=None,
            forward_queries: Any=None,
            forward_answers: Any=None,
            backward_queries: Any=None,
            backward_answers: Any=None,
            sentiment_queries: Any=None,
            sentiment_answers: Any=None,
            forward_seg: Any=None,
            backward_seg: Any=None,
            sentiment_seg: Any=None
    ):
        self.original_sample = original_sample
        self.forward_queries = forward_queries
        self.forward_answers = forward_answers
        self.backward_queries = backward_queries
        self.backward_answers = backward_answers
        self.sentiment_queries = sentiment_queries
        self.sentiment_answers = sentiment_answers
        self.forward_seg = forward_seg
        self.backward_seg = backward_seg
        self.sentiment_seg = sentiment_seg

    def __str__(self):
        str = '----------------------------------------\n'
        str += f"original_sample: {self.original_sample}\n"
        str += f"forward_queries: {self.forward_queries}\n"
        str += f"forward_answers: {self.forward_answers}\n"
        str += f"backward_queries: {self.backward_queries}\n"
        str += f"backward_answers: {self.backward_answers}\n"
        str += f"sentiment_queries: {self.sentiment_queries}\n"
        str += f"sentiment_answers: {self.sentiment_answers}\n"
        str += f"forward_seg: {self.forward_seg}\n"
        str += f"backward_seg: {self.backward_seg}\n"
        str += f"sentiment_seg: {self.sentiment_seg}\n"
        str += '----------------------------------------\n'

        return str


class OriginalDataset(Dataset):
    def __init__(self, pre_data):
        self._forward_asp_query = pre_data.get('_forward_asp_query', None)
        self._forward_opi_query = pre_data.get('_forward_opi_query', None)  # [max_aspect_num, max_opinion_query_length]
        self._forward_asp_answer_start = pre_data.get('_forward_asp_answer_start', None)
        self._forward_asp_answer_end = pre_data.get('_forward_asp_answer_end', None)
        self._forward_opi_answer_start = pre_data.get('_forward_opi_answer_start', None)
        self._forward_opi_answer_end = pre_data.get('_forward_opi_answer_end', None)
        self._forward_asp_query_mask = pre_data.get('_forward_asp_query_mask', None)  # [max_aspect_num, max_opinion_query_length]
        self._forward_opi_query_mask = pre_data.get('_forward_opi_query_mask', None)  # [max_aspect_num, max_opinion_query_length]
        self._forward_asp_query_seg = pre_data.get('_forward_asp_query_seg', None)  # [max_aspect_num, max_opinion_query_length]
        self._forward_opi_query_seg = pre_data.get('_forward_opi_query_seg', None)  # [max_aspect_num, max_opinion_query_length]

        self._backward_asp_query = pre_data.get('_backward_asp_query', None)
        self._backward_opi_query = pre_data.get('_backward_opi_query', None)  # [max_aspect_num, max_opinion_query_length]
        self._backward_asp_answer_start = pre_data.get('_backward_asp_answer_start', None)
        self._backward_asp_answer_end = pre_data.get('_backward_asp_answer_end', None)
        self._backward_opi_answer_start = pre_data.get('_backward_opi_answer_start', None)
        self._backward_opi_answer_end = pre_data.get('_backward_opi_answer_end', None)
        self._backward_asp_query_mask = pre_data.get('_backward_asp_query_mask', None)  # [max_aspect_num, max_opinion_query_length]
        self._backward_opi_query_mask = pre_data.get('_backward_opi_query_mask', None)  # [max_aspect_num, max_opinion_query_length]
        self._backward_asp_query_seg = pre_data.get('_backward_asp_query_seg', None)  # [max_aspect_num, max_opinion_query_length]
        self._backward_opi_query_seg = pre_data.get('_backward_opi_query_seg', None)  # [max_aspect_num, max_opinion_query_length]

        self._sentiment_query = pre_data.get('_sentiment_query', None)  # [max_aspect_num, max_sentiment_query_length]
        self._sentiment_answer = pre_data.get('_sentiment_answer', None)
        self._sentiment_query_mask = pre_data.get('_sentiment_query_mask', None)  # [max_aspect_num, max_sentiment_query_length]
        self._sentiment_query_seg = pre_data.get('_sentiment_query_seg', None)  # [max_aspect_num, max_sentiment_query_length]

        self._aspect_num = pre_data.get('_aspect_num', None)
        self._opinion_num = pre_data.get('_opinion_num', None)

    def __str__(self):
        str = '----------------------------------------\n'
        str += f"_forward_asp_query: {self._forward_asp_query}\n"
        str += f"_forward_opi_query: {self._forward_opi_query}\n"
        str += f"_forward_asp_answer_start: {self._forward_asp_answer_start}\n"
        str += f"_forward_asp_answer_end: {self._forward_asp_answer_end}\n"
        str += f"_forward_opi_answer_start: {self._forward_opi_answer_start}\n"
        str += f"_forward_opi_answer_end: {self._forward_opi_answer_end}\n"
        str += f"_forward_asp_query_mask: {self._forward_asp_query_mask}\n"
        str += f"_forward_opi_query_mask: {self._forward_opi_query_mask}\n"
        str += f"_forward_asp_query_seg: {self._forward_asp_query_seg}\n"
        str += f"_forward_opi_query_seg: {self._forward_opi_query_seg}\n"

        str += f"_backward_asp_query: {self._backward_asp_query}\n"
        str += f"_backward_opi_query: {self._backward_opi_query}\n"
        str += f"_backward_asp_answer_start: {self._backward_asp_answer_start}\n"
        str += f"_backward_asp_answer_end: {self._backward_asp_answer_end}\n"
        str += f"_backward_opi_answer_end: {self._backward_opi_answer_end}\n"
        str += f"_backward_opi_answer_end: {self._backward_opi_answer_end}\n"
        str += f"_backward_asp_query_mask: {self._backward_asp_query_mask}\n"
        str += f"_backward_opi_query_mask: {self._backward_opi_query_mask}\n"
        str += f"_backward_asp_query_seg: {self._backward_asp_query_seg}\n"
        str += f"_backward_opi_query_seg: {self._backward_opi_query_seg}\n"

        str += f"_sentiment_query: {self._sentiment_query}\n"
        str += f"_sentiment_answer: {self._sentiment_answer}\n"
        str += f"_sentiment_query_mask: {self._sentiment_query_mask}\n"
        str += f"_sentiment_query_seg: {self._sentiment_query_seg}\n"
        str += f"_aspect_num: {self._aspect_num}\n"
        str += f"_opinion_num: {self._opinion_num}\n"
        str += '----------------------------------------\n'

        return str


class BMRCDataset(Dataset):
    def __init__(self, train=None, dev=None, test=None, set='train', version='bidirectional'):
        self._train_set = train
        self._dev_set = dev
        self._test_set = test
        self.version = version
        if set == 'train':
            self._dataset = self._train_set
        elif set == 'dev':
            self._dataset = self._dev_set
        elif set == 'test':
            self._dataset = self._test_set

        self._forward_asp_query = self._dataset._forward_asp_query
        self._forward_opi_query = self._dataset._forward_opi_query
        self._forward_asp_answer_start = self._dataset._forward_asp_answer_start
        self._forward_asp_answer_end = self._dataset._forward_asp_answer_end
        self._forward_opi_answer_start = self._dataset._forward_opi_answer_start
        self._forward_opi_answer_end = self._dataset._forward_opi_answer_end
        self._forward_asp_query_mask = self._dataset._forward_asp_query_mask
        self._forward_opi_query_mask = self._dataset._forward_opi_query_mask
        self._forward_asp_query_seg = self._dataset._forward_asp_query_seg
        self._forward_opi_query_seg = self._dataset._forward_opi_query_seg
        self._backward_asp_query = self._dataset._backward_asp_query
        self._backward_opi_query = self._dataset._backward_opi_query
        self._backward_asp_answer_start = self._dataset._backward_asp_answer_start
        self._backward_asp_answer_end = self._dataset._backward_asp_answer_end
        self._backward_opi_answer_start = self._dataset._backward_opi_answer_start
        self._backward_opi_answer_end = self._dataset._backward_opi_answer_end
        self._backward_asp_query_mask = self._dataset._backward_asp_query_mask
        self._backward_opi_query_mask = self._dataset._backward_opi_query_mask
        self._backward_asp_query_seg = self._dataset._backward_asp_query_seg
        self._backward_opi_query_seg = self._dataset._backward_opi_query_seg
        self._sentiment_query = self._dataset._sentiment_query
        self._sentiment_answer = self._dataset._sentiment_answer
        self._sentiment_query_mask = self._dataset._sentiment_query_mask
        self._sentiment_query_seg = self._dataset._sentiment_query_seg
        self._aspect_num = self._dataset._aspect_num
        self._opinion_num = self._dataset._opinion_num

    def get_batch_num(self, batch_size):
        return len(self._forward_asp_query) // batch_size

    def __len__(self):
        return len(self._forward_asp_query)

    def __getitem__(self, item):
        #TODO: Forward
        forward_asp_query = self._forward_asp_query[item]
        forward_opi_query = self._forward_opi_query[item]
        forward_asp_answer_start = self._forward_asp_answer_start[item]
        forward_asp_answer_end = self._forward_asp_answer_end[item]
        forward_opi_answer_start = self._forward_opi_answer_start[item]
        forward_opi_answer_end = self._forward_opi_answer_end[item]
        forward_asp_query_mask = self._forward_asp_query_mask[item]
        forward_opi_query_mask = self._forward_opi_query_mask[item]
        forward_asp_query_seg = self._forward_asp_query_seg[item]
        forward_opi_query_seg = self._forward_opi_query_seg[item]

        # TODO: Backward
        if self.version.lower() in ['bi','bidirectional']:
            backward_asp_query = self._backward_asp_query[item]
            backward_opi_query = self._backward_opi_query[item]
            backward_asp_answer_start = self._backward_asp_answer_start[item]
            backward_asp_answer_end = self._backward_asp_answer_end[item]
            backward_opi_answer_start = self._backward_opi_answer_start[item]
            backward_opi_answer_end = self._backward_opi_answer_end[item]
            backward_asp_query_mask = self._backward_asp_query_mask[item]
            backward_opi_query_mask = self._backward_opi_query_mask[item]
            backward_asp_query_seg = self._backward_asp_query_seg[item]
            backward_opi_query_seg = self._backward_opi_query_seg[item]
        else:
            backward_asp_query = []
            backward_opi_query = []
            backward_asp_answer_start = []
            backward_asp_answer_end = []
            backward_opi_answer_start = []
            backward_opi_answer_end = []
            backward_asp_query_mask = []
            backward_opi_query_mask = []
            backward_asp_query_seg = []
            backward_opi_query_seg = []

        #TODO: Sentiment
        sentiment_query = self._sentiment_query[item]
        sentiment_answer = self._sentiment_answer[item]
        sentiment_query_mask = self._sentiment_query_mask[item]
        sentiment_query_seg = self._sentiment_query_seg[item]

        aspect_num = self._aspect_num[item]
        opinion_num = self._opinion_num[item]

        return {
            "forward_asp_query": np.array(forward_asp_query),
            "forward_opi_query": np.array(forward_opi_query),
            "forward_asp_answer_start": np.array(forward_asp_answer_start),
            "forward_asp_answer_end": np.array(forward_asp_answer_end),
            "forward_opi_answer_start": np.array(forward_opi_answer_start),
            "forward_opi_answer_end": np.array(forward_opi_answer_end),
            "forward_asp_query_mask": np.array(forward_asp_query_mask),
            "forward_opi_query_mask": np.array(forward_opi_query_mask),
            "forward_asp_query_seg": np.array(forward_asp_query_seg),
            "forward_opi_query_seg": np.array(forward_opi_query_seg),
            "backward_asp_query": np.array(backward_asp_query),
            "backward_opi_query": np.array(backward_opi_query),
            "backward_asp_answer_start": np.array(backward_asp_answer_start),
            "backward_asp_answer_end": np.array(backward_asp_answer_end),
            "backward_opi_answer_start": np.array(backward_opi_answer_start),
            "backward_opi_answer_end": np.array(backward_opi_answer_end),
            "backward_asp_query_mask": np.array(backward_asp_query_mask),
            "backward_opi_query_mask": np.array(backward_opi_query_mask),
            "backward_asp_query_seg": np.array(backward_asp_query_seg),
            "backward_opi_query_seg": np.array(backward_opi_query_seg),
            "sentiment_query": np.array(sentiment_query),
            "sentiment_answer": np.array(sentiment_answer),
            "sentiment_query_mask": np.array(sentiment_query_mask),
            "sentiment_query_seg": np.array(sentiment_query_seg),
            "aspect_num": np.array(aspect_num),
            "opinion_num": np.array(opinion_num)
        }


def generate_fi_batches(dataset, batch_size, shuffle=True, drop_last=True, ifgpu=True):
    dataloader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last
    )
    for data_dict in dataloader:
        out_dict = {}
        for name, tensor in data_dict.items():
            if ifgpu:
                out_dict[name] = data_dict[name].cuda()
            else:
                out_dict[name] = data_dict[name]
        yield out_dict