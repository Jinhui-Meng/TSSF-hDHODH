# load model to predict binding affinity
# usage: python argv1:feature_file argv2:model_type(a or b)

import os
import sys
import joblib
import pandas as pd

os.chdir(os.getcwd())

feature_file = str(sys.argv[1])
model_type = str(sys.argv[2])

if model_type == 'a':
    model_file = '../models/TSSF-hDHODH-a.pkl'
if model_type == 'b':
    model_file = '../models/TSSF-hDHODH-b.pkl'

data_df = pd.read_csv(feature_file, encoding='utf-8', header=0)
X_data = data_df.iloc[:,1:-1]
Y_data = data_df.iloc[:,0]
ID_data = data_df.iloc[:,-1]

best_model = joblib.load(model_file)

pre_data = best_model.predict(X_data)
pre_data_df = pd.DataFrame({'model_predict': pre_data})
ID_data_df = pd.DataFrame({'ID': ID_data})
output_df = pd.concat([ID_data_df, pre_data_df], axis=1)
output_df.to_csv(os.path.join('predict_result_%s.csv' % model_type), sep = ",", index = None)
