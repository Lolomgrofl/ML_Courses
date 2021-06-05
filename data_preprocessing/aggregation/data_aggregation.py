import pandas as pd

from ..missing_values.missing_values import TransformMissingValues

import logging
logging.basicConfig(filename='../../log_file.log', level=logging.INFO, 
                    format='%(asctime)s %(levelname)-8s %(message)s', datefmt='%Y-%m-%d %H:%M:%S')

class DataAggregation:

    def __init__(self):
        return

    def transform(self, dataframe, indexes, columns, values, agg_funcs):
        '''
        Perform data aggregation
            
        Parameters:
        dataframe: pandas.DataFrame
            Pandas Dataframe for which data aggregation is performed
        index: array
            Rows of the new dataframe
        columns: array
            Columns of the new dataframe
        values: array
            Values column in the new dataframe
        agg_funcs: array
            Aggregation function to use
        
        Output:
        Pandas.DataFrame: Pandas DataFrame to use
        '''
        
        try:
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
        except Exception as e:
            logging.error('Error during Data Aggregation: ' + e.msg)

