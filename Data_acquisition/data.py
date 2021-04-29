import pandas as pd
# from Models.Experiment import Experiment
from Data_preprocessing.Data_aggregation import DataAggregation
import matplotlib.pyplot as plt

if __name__ == '__main__':

    # ----- 1. DATA ACQUSITION -----
    df_assessments = pd.read_csv('data/assessments.csv')
    df_courses = pd.read_csv('data/courses.csv')
    df_studentAssessment = pd.read_csv('data/studentAssessment.csv')
    df_studentInfo = pd.read_csv('data/studentInfo.csv')
    df_studentRegistration = pd.read_csv('data/studentRegistration.csv')
    df_studentVle = pd.read_csv('data/studentVle.csv')
    df_vle = pd.read_csv('data/vle.csv')
    
    df_studentInfo = df_studentInfo.set_index(['code_module', 'code_presentation', 'id_student'])

    # ----- 2. DATA PREPROCESSING -----

    # FILTERING DATA
    df_studentVle = df_studentVle.loc[df_studentVle['date'] < 30]
    df_studentAssessment = df_studentAssessment.loc[df_studentAssessment['date_submitted'] < 30]
    df_studentRegistration = df_studentRegistration.loc[df_studentRegistration['date_unregistration'] < 30]
    
    # AGGREGATION - StudentVLE
    ind = ['code_module', 'code_presentation', 'id_student']
    col = df_vle['activity_type']
    cols = [None, col]
    vals = ['sum_click']
    aggs = ['count']
    da = DataAggregation()
    df = da.transform(df_studentVle, ind, cols, vals, aggs)
    
    # AGGREGATION - StudentAssessment (POGLEDACEMO)
    ind = ['code_module', 'code_presentation', 'id_student']
    col = df_vle['activity_type']
    cols = [None, col]
    vals = ['sum_click']
    aggs = ['count']
    da = DataAggregation()
    df_1 = da.transform(df_studentAssessment, ind, cols, vals, aggs)
    
    # NUMERICAL TRANSFORMATIONS (OPTIONAL)
    
    # MERGING TO FINAL DATASET
    df = df_studentInfo.merge(df, left_index=True, right_index=True)
    df = df.merge(df_1, left_index=True, right_index=True)
    
    # DATA PREPROCESSING - MISSING VALUES, DUMMY CODING ETC.
    
    X = df.drop('final_grade', axis=1)
    y = df['final_grade']
    
    print(df.head(5))
