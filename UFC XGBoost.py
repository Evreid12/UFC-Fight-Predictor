# -*- coding: utf-8 -*-
"""
Created on Tue May 12 12:56:08 2026

@author: evan-
"""

import pandas as pd
from xgboost import XGBClassifier, XGBRegressor
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, log_loss
import matplotlib.pyplot as plt
import seaborn as sns
import xgboost as xgb

df = pd.read_csv(r"C:\Users\evan-\OneDrive\Documents\large_dataset.csv")
print(df.head())


# 1. PREPROCESSING
# Ensure chronological order is critical for Elo
# (Assuming your dataset has a date or is sorted; if not, we reverse if it's new-to-old)
df = df.iloc[::-1].reset_index(drop=True) 

def calculate_elo(df, k_factor=32):
    elo_dict = {}  # Stores current elo of every fighter
    r_elo_list = []
    b_elo_list = []
    
    for idx, row in df.iterrows():
        r_fighter = row['r_fighter']
        b_fighter = row['b_fighter']
        
        # Initialize new fighters at 1500
        r_rating = elo_dict.get(r_fighter, 1500)
        b_rating = elo_dict.get(b_fighter, 1500)
        
        # Record pre-fight elo
        r_elo_list.append(r_rating)
        b_elo_list.append(b_rating)
        
        # Expected scores
        exp_r = 1 / (1 + 10 ** ((b_rating - r_rating) / 400))
        exp_b = 1 / (1 + 10 ** ((r_rating - b_rating) / 400))
        
        # Actual scores
        if row['winner'] == 'Red':
            s_r, s_b = 1, 0
        elif row['winner'] == 'Blue':
            s_r, s_b = 0, 1
        else: # Draw or No Contest
            s_r, s_b = 0.5, 0.5
            
        # Update ratings
        elo_dict[r_fighter] = r_rating + k_factor * (s_r - exp_r)
        elo_dict[b_fighter] = b_rating + k_factor * (s_b - exp_b)
        
    df['r_elo'] = r_elo_list
    df['b_elo'] = b_elo_list
    df['elo_diff'] = df['r_elo'] - df['b_elo']
    return df

# Apply Elo
df = calculate_elo(df)

# 2. FEATURE SELECTION
# We'll use the new Elo features plus some physical stats
features = ['r_elo', 'b_elo', 'elo_diff', 'r_age', 'b_age', 'r_height', 'b_height', 
    'r_reach', 'b_reach','r_sig_str_acc_total', 'b_sig_str_acc_total']

# Drop rows with missing values in our selected features
df_model = df[features + ['winner']].dropna()

# Encode Target: Red = 0, Blue = 1 (Standard binary)
# Note: We ignore draws for this basic binary classifier
df_model = df_model[df_model['winner'].isin(['Red', 'Blue'])]
y = df_model['winner'].map({'Red': 0, 'Blue': 1})
X = df_model[features]

# 3. XGBOOST MODELING
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = xgb.XGBClassifier(
    n_estimators=100,
    learning_rate=0.05,
    max_depth=4,
    subsample=0.8,
    colsample_bytree=0.8,
    use_label_encoder=False,
    eval_metric='logloss')

model.fit(X_train, y_train)

# 4. EVALUATION
preds = model.predict(X_test)
probs = model.predict_proba(X_test)[:, 1]

print(f"Accuracy: {accuracy_score(y_test, preds):.2%}")
print(f"Log Loss: {log_loss(y_test, probs):.4f}")

# Check feature importance
importance = pd.Series(model.feature_importances_, index=features).sort_values(ascending=False)
print("\nTop Features:\n", importance)

