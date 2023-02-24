#!/usr/bin/env python
import torch
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer
from fclustering.models.plsa import PLSA


# news = fetch_20newsgroups(subset='all')
# data = CountVectorizer(min_df=5, stop_words='english').fit_transform(news.data)

# print(data.toarray().shape)
data = [[[1, 1, 1, 0, 0, 0, 0, 0, 0],
        [1, 1, 1, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 3, 4, 1, 0, 0, 0],
        [0, 0, 0, 5, 4, 4, 0, 0, 0],
        [0, 0, 0, 2, 2, 2, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 4, 2, 1],
        [0, 0, 0, 0, 0, 0, 3, 1, 4],
        [0, 0, 0, 0, 0, 0, 1, 5, 1]]]

plsa = PLSA(3)
plsa.train(torch.tensor(data).float())
plsa.predict()