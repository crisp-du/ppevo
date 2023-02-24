from transformers import BertTokenizer
from pathlib import Path
import torch
from fast_bert.prediction import BertClassificationPredictor
import pandas as pd

MODEL_PATH = "./model/output/model_out/"
LABEL_PATH = Path('./model/labels/')
sentencePredictor = BertClassificationPredictor(MODEL_PATH, LABEL_PATH, multi_label=True, model_type='xlnet')

sentenceList = ['We use the third-party service provider DR globalTech Inc. as our online official re-seller and for certain in-game sales functions.',
           'You may choose to voluntarily provide us with various personal information and non-personal information through your use of the Online Services.']
           
sentenceLabels = sentencePredictor.predict_batch(sentenceList)
outputDataFrame = pd.DataFrame([{item[0]: float(item[1]) for item in prediction} for prediction in sentenceLabels])

outputDataFrame.insert(0,'Category',list(outputDataFrame.idxmax(axis=1)))
outputDataFrame.insert(0,'Text',sentenceList)
outputDataFrame