from typing import Literal

import pandas as pd
from pandas import DataFrame
from pandas._typing import CorrelationMethod


class DataAnalyzer:
    @classmethod
    def correlations(cls, data: pd.DataFrame, method: CorrelationMethod = "pearson") -> DataFrame:
        result = data.corr(method=method)
        print(result)
        return result
