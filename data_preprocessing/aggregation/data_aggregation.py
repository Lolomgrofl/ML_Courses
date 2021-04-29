import pandas as pd

from .aggregation.missing_values import TransformMissingValues, StrategyTransform


class DataAggregation:

    def __init__(self):
        return

    def transform(self, dataframe, indexes, columns, values, agg_funcs):

        chunks = []
        for column in columns:
            for value in values:
                for agg_func in agg_funcs:
                    pivot = pd.pivot_table(dataframe, index=indexes,
                                           aggfunc=agg_func,
                                           columns=column,
                                           values=value)
                    chunks.append(pivot)
        df = pd.concat(chunks, axis=1)
        t = TransformMissingValues()
        df = t.fill_missing_values(df)
        return df

