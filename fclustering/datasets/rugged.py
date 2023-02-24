from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd


from .download import Download
from .preprocess import Preprocess


@dataclass
class RUGGED(Download, Preprocess):
    """ Real GDP per capita for the year 2000;
    """
    df: pd.DataFrame = None
    DATA_URL = "https://d2hg8soec8ck9v.cloudfront.net/datasets/rugged_data.csv"    
    def __post_init__(self) -> RUGGED:
        self.df = self.preprocess()    

    @staticmethod
    def download() -> RUGGED:
        data = pd.read_csv(RUGGED.DATA_URL, encoding="ISO-8859-1")
        df = data[["cont_africa", "rugged", "rgdppc_2000"]]
        return RUGGED(df)

    def preprocess(self) -> RUGGED:
        valid_df = self.df[np.isfinite(self.df.rgdppc_2000)]
        valid_df.loc[:, ["rgdppc_2000"]] = np.log(valid_df["rgdppc_2000"])
        return valid_df