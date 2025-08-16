# FAKE NEWS DETECTOR 

# First Part
# Made by Arnav 
# Data Collection

import pandas as pd

# List of dataset file paths
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

# Combine all datasets into one DataFrame
if dataframes:
    df = pd.concat(dataframes, ignore_index=True)
    print("\n Datasets combined successfully!")
    print(df.info())
    print(df.head())
else:
    print("No datasets were loaded.")


# kuch bhi import karna ho toh 7 line pe hi karna jha par hum kiye hai 
# Aur bhai log please seriously banana 


# Second Part
# Made by 
# Data Preprocessing 


# Write Your code from here


# Third Part
# Made by 
# Feature Extraction (Vectorisation )


# Write Your code from here


# Fourth Part
# Made by Arpan Pandey
# Training of model


# Write Your code from here


# Fifth Part
# Made by 
# Evaluation


# Write Your code from here


# Sixth Part
# Made by Anurag Dubey
# Deployment


# Write Your code from here

