import pandas as pd

# from data_preprocessing.preprocessing import categorical_coding, numerical_transformations
from data_preprocessing.aggregation.data_aggregation import DataAggregation
from data_preprocessing.missing_values.missing_values import TransformMissingValues

# import matplotlib.pyplot as plt

import logging
logging.basicConfig(filename='log_file.log', level=logging.INFO, 
                    format='%(asctime)s %(levelname)-8s %(message)s', datefmt='%Y-%m-%d %H:%M:%S')

if __name__ == '__main__':

    #----- ----- ----- -----
    logging.info('Process started')
    #----- ----- ----- -----
    
    # ----- 1. DATA ACQUSITION -----
    df_assessments = pd.read_csv('data/assessments.csv')
    df_courses = pd.read_csv('data/courses.csv')
    df_studentAssessment = pd.read_csv('data/studentAssessment.csv')
    df_studentInfo = pd.read_csv('data/studentInfo.csv')
    df_studentRegistration = pd.read_csv('data/studentRegistration.csv')
    df_studentVle = pd.read_csv('data/studentVle.csv')
    df_vle = pd.read_csv('data/vle.csv')
    
    df_studentInfo = df_studentInfo.set_index(['code_module', 'code_presentation', 'id_student']) 

    #----- ----- ----- -----
    logging.info('Data acquising ended')
    #----- ----- ----- -----
    
    # ----- 2. DATA PREPROCESSING -----

    #----- ----- ----- -----
    logging.info('Data preparation started')
    #----- ----- ----- -----
    
    # FILTERING DATA
    filter_data = 30
    
    df_studentVle = df_studentVle.loc[df_studentVle['date'] < filter_data]
    df_studentAssessment = df_studentAssessment.loc[df_studentAssessment['date_submitted'] < filter_data]
    df_studentRegistration = df_studentRegistration.loc[df_studentRegistration['date_unregistration'] < filter_data]
    
    # AGGREGATION - StudentVLE
    ind = ['code_module', 'code_presentation', 'id_student']
    col = df_vle['activity_type']
    cols = [None, col]
    vals = ['sum_click']
    aggs = ['count']
    da = DataAggregation()
    df = da.transform(df_studentVle, ind, cols, vals, aggs)
    
    # AGGREGATION - StudentAssessment
    ind = ['id_student']
    col = df_vle['activity_type']
    cols = [None, col]
    vals = ['score']
    aggs = ['mean']
    da = DataAggregation()
    df_1 = da.transform(df_studentAssessment, ind, cols, vals, aggs)
    
    # NUMERICAL TRANSFORMATIONS (OPTIONAL)
    
    #----- ----- ----- -----
    logging.info('Data aggregation done')
    #----- ----- ----- -----
    
    # MERGING TO FINAL DATASET
    df = df_studentInfo.merge(df, left_index=True, right_index=True)
    df = df.merge(df_1, left_index=True, right_index=True)
    
    #----- ----- ----- -----
    logging.info('Data merged done')
    #----- ----- ----- -----
    
    # DATA PREPROCESSING - MISSING VALUES, DUMMY CODING ETC.
    X = df.drop('final_result', axis=1)
    dummy = categorical_coding.DummyCoding()
    dummy.fit(X, columns=X.select_dtypes(object).columns.values.tolist())
    X = dummy.transform(X)
    missing = TransformMissingValues()
    X = missing.fill_missing_values(X, strategy=Stra)
    
    y = df['final_result']
    
    #----- ----- ----- -----
    logging.info('Dummy and missing data')
    #----- ----- ----- -----
    
    print(X.head(5))
