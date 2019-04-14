#!/usr/bin/env python3
## All Imports
#import os
#import re
#import numpy as np
import pandas as pd
import pickle
#from keras.utils import np_utils
#from keras.models import load_model, Model
#from keras.layers import Activation, Dense, LSTM, Conv1D, MaxPooling1D, Bidirectional, BatchNormalization
#from keras.optimizers import Adam
#import xgboost as xgb


import sys
#import numpy as np


def create_data(input_file):
     
    data = pd.read_table(input_file, sep = '|')
    data['LabTest'] = len(lab_cols)- data[lab_cols].isnull().sum(axis = 1)
    data = data.fillna(method = 'ffill')
    for col in Normal_value_dic.keys():
        data[col] = data[col].fillna(Normal_value_dic[col])
    
    return data

def get_sepsis_score(model, X_test, threshold):
    test_features = X_test[feature_to_inc]
    y_pred = model.predict_proba(test_features)
    scores = y_pred[:,1]
    labels = (scores >= threshold).astype('int')
    return scores, labels




if __name__ == '__main__':

    Normal_value_dic = {'HR': 75 , 'O2Sat': 100, 'Temp': 37, 'SBP': 120, 'MAP': 85, 'DBP': 80, 'Resp': 14, 'EtCO2': 40,
               'BaseExcess': 0, 'HCO3': 25.5, 'FiO2': 0.500, 'pH':7.4, 'PaCO2':85, 'SaO2': 100, 'AST': 37.5, 'BUN':13.5,
               'Alkalinephos': 90, 'Calcium': 9.35, 'Chloride': 101 , 'Creatinine':0.9, 'Bilirubin_direct':0.3,
               'Glucose':90, 'Lactate':2.2, 'Magnesium':2, 'Phosphate':3.5, 'Potassium':4.25,
               'Bilirubin_total': 0.75, 'TroponinI':0.04, 'Hct':48.5, 'Hgb':13.75, 'PTT':30, 'WBC':7.7,
               'Fibrinogen':375, 'Platelets':300}
    lab_cols = ['BaseExcess', 'HCO3', 'FiO2', 'pH', 'PaCO2', 'SaO2', 'AST', 'BUN',
           'Alkalinephos', 'Calcium', 'Chloride', 'Creatinine', 'Bilirubin_direct',
           'Glucose', 'Lactate', 'Magnesium', 'Phosphate', 'Potassium',
           'Bilirubin_total', 'TroponinI', 'Hct', 'Hgb', 'PTT', 'WBC',
           'Fibrinogen', 'Platelets']
    
    feature_to_inc = ['HR', 'O2Sat', 'Temp', 'SBP', 'MAP', 'DBP', 'Resp', 'EtCO2',
           'BaseExcess', 'HCO3', 'FiO2', 'pH', 'PaCO2', 'SaO2', 'AST', 'BUN',
           'Alkalinephos', 'Calcium', 'Chloride', 'Creatinine', 'Bilirubin_direct',
           'Glucose', 'Lactate', 'Magnesium', 'Phosphate', 'Potassium',
           'Bilirubin_total', 'TroponinI', 'Hct', 'Hgb', 'PTT', 'WBC',
           'Fibrinogen', 'Platelets', 'Age', 'Gender', 'HospAdmTime',
                  'ICULOS', 'LabTest']

    feature_names = ['HR', 'O2Sat', 'Temp', 'SBP', 'MAP', 'DBP', 'Resp', 'EtCO2',
               'BaseExcess', 'HCO3', 'FiO2', 'pH', 'PaCO2', 'SaO2', 'AST', 'BUN',
               'Alkalinephos', 'Calcium', 'Chloride', 'Creatinine', 'Bilirubin_direct',
               'Glucose', 'Lactate', 'Magnesium', 'Phosphate', 'Potassium',
               'Bilirubin_total', 'TroponinI', 'Hct', 'Hgb', 'PTT', 'WBC',
               'Fibrinogen', 'Platelets', 'Age', 'Gender', 'HospAdmTime', 'ICULOS', "LabTest"]
    
    
    xgboost_path = "XGBoost_All_FeatOnly"
    xgboost_model = pickle.load(open(xgboost_path, 'rb'))  
    threshold = 0.02



    # read input data
    data = create_data(sys.argv[1])
    scores, labels = get_sepsis_score(xgboost_model, data, threshold)
    
    
    # write predictions to output file
    
    with open(sys.argv[2], 'w') as f:
        f.write('PredictedProbability|PredictedLabel\n')
        for (s, l) in zip(scores, labels):
            f.write('%g|%d\n' % (s, l))
    

