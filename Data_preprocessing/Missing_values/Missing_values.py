from enum import Enum


class StrategyTransform(Enum):
    ZERO = 1
    MEAN = 2


class TransformMissingValues:

    def __init__(self):
        return

    def fill_missing_values(self, dataframe, strategy=StrategyTransform.ZERO):
        if strategy == StrategyTransform.ZERO:
            return dataframe.fillna(0)
        elif strategy == StrategyTransform.MEAN:
            return dataframe.fillna(dataframe.mean())

