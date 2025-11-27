

#                                                               FAKE NEWS DETECTOR 
# First Part 
# Made by Arnav 
# Data Collection
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import string
import re
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report


files = [
    "gossipcop_real.csv",
    "facebook-fact-check.csv",
    "Fake_Dataset.csv",
    "gossipcop_fake.csv",
    "websites.csv",
    "news_dataset.csv",
    "news_sample.csv",
    "politifact_fake.csv",
    "politifact_real.csv",
    "test.tsv", 
    "train.tsv", 
    "valid.tsv",
    "True.csv",
]

dataframes = []

for f in files:
    try:
        if f.endswith(".tsv"):
            df = pd.read_csv(f, sep="\t", engine="python", on_bad_lines="skip")
        else:  # default CSV
            df = pd.read_csv(f, engine="python", on_bad_lines="skip")
        dataframes.append(df)
        print(f"Loaded {f} successfully! Rows: {len(df)}")
    except Exception as e:
        print(f"Error loading {f}: {e}")


if dataframes:
    df = pd.concat(dataframes, ignore_index=True)
    print("\n Datasets combined successfully!")
    print(df.info())
    print(df.head())
else:
    print("No datasets were loaded.")


    print("This news is fake")


# Second Part
# Made by Anuj Mishhra
# Data Preprocessing 

def wordopt(text):
    text = text.lower()
    text = re.sub('\[.*?\]','',text)
    text = re.sub("\\W"," ",text)
    text = re.sub('https?://\S+|www\.\S+','',text)
    text = re.sub('<.*?>+',b'',text)
    text = re.sub('[%s]' % re.escape(string.punctuation),'',text)
    text = re.sub('\w*\d\w*','',text)
    return text

data['text'] = data['text'].apply(wordopt)

x = data['text']
y = data['class']

# Fourth Part
# Made by Arpan Pandey
# Training of model

x_train, x_test, y_train, y_test = train_test_split(x,y,test_size = 0.25)

# Third Part
# Made by Anurag Dev Mishra
# Feature Extraction (Vectorisation )

from sklearn.feature_extraction.text import TfidfVectorizer

vectorization = TfidfVectorizer()
xv_train = vectorization.fit_transform(x_train)
xv_test = vectorization.transform(x_test)

# Logistic Regression
from sklearn.linear_model import LogisticRegression

LR = LogisticRegression()
LR.fit(xv_train, y_train)

pred_lr = LR.predict(xv_test)

LR.score(xv_test, y_test)

print (classification_report(y_test, pred_lr))

# Decision Tree Classifier
from sklearn.tree import DecisionTreeClassifier

DT = DecisionTreeClassifier()
DT.fit(xv_train, y_train)

pred_dt = DT.predict(xv_test)

DT.score(xv_test, y_test)

print (classification_report(y_test, pred_lr))

# Gradient Boost Classifier

from sklearn.ensemble import GradientBoostingClassifier

GB = GradientBoostingClassifier(random_state = 0)
GB.fit(xv_train, y_train)

pred_gb = GB.predict(xv_test)

GB.score(xv_test, y_test)

print(classification_report(y_test, pred_gb))

# Random Forest Classifier

from sklearn.ensemble import RandomForestClassifier

RF = RandomForestClassifier(random_state = 0)
RF.fit(xv_train, y_train)

pred_rf = RF.predict(xv_test)

RF.score(xv_test, y_test)

print (classification_report(y_test, pred_rf))


# Fifth Part
# Made by Arpan Yadav
# Evaluation

def output_lable(n):
    if n==0:
        return "Fake News"
    elif n==1:
        return "Not A Fake News"
    
def manual_testing(news):
    testing_news = {"text":[news]}
    new_def_test = pd.DataFrame(testing_news)
    new_def_test['text'] = new_def_test["text"].apply(wordopt)
    new_x_test = new_def_test["text"]
    new_xv_test = vectorization.transform(new_x_test)
    pred_LR = LR.predict(new_xv_test)
    pred_DT = DT.predict(new_xv_test)
    pred_GB = GB.predict(new_xv_test)
    pred_RF = RF.predict(new_xv_test)
    
    return print("\n\nLR Predicition: {} \nDT Prediction: {} \nGBC Prediction: {} \nRFC Prediction:{}".format(output_lable(pred_LR[0]),
                                                                                                             output_lable(pred_DT[0]),
                                                                                                             output_lable(pred_GB[0]),
                                                                                                             output_lable(pred_RF[0])))

news = str(input()) 
manual_testing(news)

news=str(input())
manual_testing(news)

# Sixth Part
# Made by Anurag Dubey
# Deployment




