#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import random
from pprint import pprint

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .parser_utils import load_and_preprocess_data


class ParserModel(nn.Module):
    """ Feedforward neural network with an embedding layer and single hidden layer.
    The ParserModel will predict which transition should be applied to a
    given partial parse configuration.
    """

    def __init__(self, embeddings, n_features=36,
                 hidden_size=200, n_classes=3, dropout_prob=0.5):
        """ Initialize the parser model.

        @param embeddings (Tensor): word embeddings (num_words, embedding_size)
        @param n_features (int): number of input features
        @param hidden_size (int): number of hidden units
        @param n_classes (int): number of output classes
        @param dropout_prob (float): dropout probability
        """
        super(ParserModel, self).__init__()
        torch.manual_seed(0)
        self.n_features = n_features
        self.n_classes = n_classes
        self.dropout_prob = dropout_prob
        self.embed_size = embeddings.shape[1]
        self.hidden_size = hidden_size
        self.pretrained_embeddings = nn.Embedding(embeddings.shape[0], self.embed_size)
        self.pretrained_embeddings.weight = nn.Parameter(torch.tensor(embeddings))


        #linear layer with xavier initialization
        self.embed_to_hidden = nn.Linear(self.n_features * self.embed_size, self.hidden_size)
        nn.init.xavier_uniform_(self.embed_to_hidden.weight)
        #dropout layer
        self.dropout = nn.Dropout(p=self.dropout_prob)
        #linear layer with xavier initialization
        self.hidden_to_logits = nn.Linear(self.hidden_size, self.n_classes)
        nn.init.xavier_uniform_(self.hidden_to_logits.weight)

    def embedding_lookup(self, t):
        """ Utilize `self.pretrained_embeddings` to map input `t` from input tokens (integers)
            to embedding vectors.

            PyTorch Notes:
                - `self.pretrained_embeddings` is a torch.nn.Embedding object that we defined in __init__
                - Here `t` is a tensor where each row represents a list of features. Each feature is represented by an integer (input token).
                - In PyTorch the Embedding object, e.g. `self.pretrained_embeddings`, allows you to
                    go from an index to embedding. Please see the documentation (https://pytorch.org/docs/stable/nn.html#torch.nn.Embedding)
                    to learn how to use `self.pretrained_embeddings` to extract the embeddings for your tensor `t`.

            @param t (Tensor): input tensor of tokens (batch_size, n_features)

            @return x (Tensor): tensor of embeddings for words represented in t
                                (batch_size, n_features * embed_size)
        """

        # look up the embeddings for input tokens in t
        x = self.pretrained_embeddings(t)
        # reshape embeddings tensor from (batch_size, n_features, embedding_size) to
        # (batch_size, n_features * embedding_size)
        x = x.view(x.shape[0],x.shape[1]*x.shape[2])

        return x

    def forward(self, t):
        """ Run the model forward.

        @param t (Tensor): input tensor of tokens (batch_size, n_features)

        @return logits (Tensor): tensor of predictions (output after applying the layers of the network)
                                 without applying softmax (batch_size, n_classes)
        """

        ### Note: We do not apply the softmax to the logits here, because
        ### the loss function (torch.nn.CrossEntropyLoss) applies it more efficiently.

        # get the embeddings
        embeddings = self.embedding_lookup(t)
        # apply the embed_to_hidden linear layer to the embeddings, then apply the relu nonlinearity to the linear layer,
        # which gives us the hidden units
        h = nn.functional.relu(self.embed_to_hidden(embeddings))
        # apply our dropout layer to the hidden units
        h_with_dropout = self.dropout(h)
        # get the logits with the hidden_to_logits layer
        logits = self.hidden_to_logits(h_with_dropout)
        ### END CODE HERE
        return logits


if __name__ == '__main__':
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
    np.random.seed(42)
    random.seed(42)
    torch.backends.cudnn.deterministic = True

    _, embeddings, _, _, _ = load_and_preprocess_data()
    model = ParserModel(embeddings)
    t = test_case.test_cases_ip['parser_model']['t']
    model_actual = model.forward(t).data.numpy().tolist()
    model_expected = test_case.test_cases_op['parser_model']
    print("actual output")
    pprint(model_actual)
    print("\n" * 2)
    print("expected output")
    pprint(model_expected)

    print("test passed" if np.isclose(model_actual, model_expected, atol=1e-2).all() else "test failed")
