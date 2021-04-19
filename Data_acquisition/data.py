import pandas as pd

from Data_preprocessing.Data_aggregation import DataAggregation


df_assessments = pd.read_csv('data/assessments.csv')
df_courses = pd.read_csv('data/courses.csv')
df_studentAssessment = pd.read_csv('data/studentAssessment.csv')
df_studentInfo = pd.read_csv('data/studentInfo.csv')
df_studentRegistration = pd.read_csv('data/studentRegistration.csv')
df_studentVle = pd.read_csv('data/studentVle.csv')
df_vle = pd.read_csv('data/vle.csv')


if __name__ == '__main__':
    #Filtering on dates
    df_studentVle = df_studentVle.loc[df_studentVle['date'] < 30]
    df_studentAssessment = df_studentAssessment.loc[df_studentAssessment['date_submitted'] < 30]
    df_studentRegistration = df_studentRegistration.loc[df_studentRegistration['date_unregistration'] < 30]
    #Aggregation
    ind = ['code_module', 'code_presentation', 'id_student']
    col = df_vle['activity_type']
    cols = [None, col]
    vals = ['sum_click']
    aggs = ['count']
    da = DataAggregation()
    df = da.transform(df_studentVle, ind, cols, vals, aggs)

    print(df.head())
