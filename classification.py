import pandas as pd
import os
from sklearn import ensemble, preprocessing, metrics
import joblib

cwd = os.getcwd()
featureFile = os.path.join(cwd, 'featureResult', 'feature.pkl')
with open(featureFile, 'rb') as f:
    tumorData = joblib.load(f)
case_list = tumorData["case_name"]
tumorData = tumorData.iloc[:, 38:]

pcaFile = os.path.join(cwd, 'pca.joblib')
modelFile = os.path.join(cwd, 'XGboost_classification_model.joblib')

## load model

model = joblib.load(modelFile)
pca = joblib.load(pcaFile)
predictProba = []
tumorDataFeature = tumorData.copy(deep=True) ## shape_columns 
X_pca = pca.transform(tumorDataFeature)
predict_y = model.predict_proba(X_pca)
predictProba = list(predict_y)

out_df = {'case_name': case_list, 'predict_proba': predictProba}
out_df = pd.DataFrame(out_df)
outputPath = os.path.join(cwd, 'classificationResult')
if not os.path.isdir(outputPath):
    os.mkdir(outputPath)
out_df.to_pickle(os.path.join(cwd, 'classificationResult', 'classificationResult.pkl'))
