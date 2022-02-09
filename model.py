# -*- coding: utf-8 -*-

import  torch.nn as nn

from transformers import (
    BertModel,
    BertTokenizer
)

class BMRCBertModel(nn.Module):
    def __init__(self, args):
        hidden_size = args.hidden_size
        super(BMRCBertModel, self).__init__()
        if args.model_type == 'bert-base-uncased':
            self._bert = BertModel.from_pretrained(args.model_type)
            self._tokenizer = BertTokenizer.from_pretrained(args.model_type)
            print(f"Loaded `{args.model_type}` model !")
        else:
            raise KeyError(f"Config.args.model_type should be `bert-base-uncased`.")

        self.cls_start = nn.Linear(hidden_size, 2)
        self.cls_end = nn.Linear(hidden_size, 2)
        self.cls_sentiment = nn.Linear(hidden_size, 3)

    def forward(self, input_ids, attention_mask, token_type_ids, step):
        hidden_states = self._bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )[0]

        if step == 0:   # TODO: Predict entity
            out_scores_start = self.cls_start(hidden_states)
            out_scores_end = self.cls_end(hidden_states)
            return out_scores_start, out_scores_end
        else:           # TODO: Predict sentiment
            cls_hidden_states = hidden_states[:, 0, :]
            cls_hidden_scores = self.cls_sentiment(cls_hidden_states)
            return cls_hidden_scores
