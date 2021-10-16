from joblib import dump, load
import pandas as pd

x_test = pd.read_pickle('data/test_prepared.pkl')

model = load('models/bestModel.joblib')
y_pred = model.predict(x_test)
prediction = pd.DataFrame(y_pred, columns=['predictions']).to_csv('inference/predictions.csv')