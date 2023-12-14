# EECS 487: Natural Language Processing - Final Project  
*Members:* Bradley Kerr, Dalton Richardson, Maxcy Denmark

## File Descriptions:
LSTM.ipynb, BERT_VER2.ipynb, and GPT2.ipynb are the final versions of each of those models used to run and evaluate them. BERT_VER1 uses auto model for classification - it is a simpler version. HumanPredictions.ipynb contains the script used to prompt the human predictions.

ChatGPT.ipynb contains the evaluation only. data/chatgpt_responses contains the spreadsheet used to fetch the responses in google sheets.

The dataset is contained in train.csv. We did not use test.csv or sample_submission.csv, which contain the separate test data provided from teh kaggle dataset. Instead, we simply patitioned train.csv into a test set.

LSTM.py is a concatenation of the LSTM notebook. It was used to run remotely in google colab. 
