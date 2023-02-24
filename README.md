# fclustering (Fuzzy Clustering)
fclustering is Pyro-backend Fuzzy Clustering Python library


## Installation

```shell
$ pip install fclustering
```

## how to use

```python
from fclusetring.model import PLSA
from flustering.dataset import Dataset

model = PLSA()
dataset = Dataset.from_csv("hogehoge.csv", sparse=True)
result = model.run(dataset)
result.visualize("hogehoge.png")
result.dump_csv("output.csv")
```
