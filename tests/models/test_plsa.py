import torch

from fclustering.models.plsa import PLSA


def test_plsa():
    test_data = [[1, 0, 0, 0, 2, 3, 4, 0, 1],
                 [0, 0, 3, 0, 0, 1, 2, 0, 1],
                 [1, 0, 2, 2, 0, 3, 0, 1, 4],
                 [1, 1, 1, 0, 0, 0, 0, 0, 1]]
    data = torch.tensor(test_data).float().view(1, 4, -1)
    
    plsa = PLSA(3)
    #plsa.model(data)
    plsa.train(data)
    plsa.predict()
