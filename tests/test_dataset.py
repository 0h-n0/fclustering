from pathlib import Path

from fclustering.dataset import Dataset


def test_from_csv():
    csv_path = Path(__file__)
    Dataset.from_csv(csv_path)
