# -*- coding: utf-8 -*-
"""
Created on Fri Sep 27 18:05:15 2019

@author: anje.knottnerus
"""

import pandas as pd
import numpy as np 
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt 
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import seaborn as sns

class Intro(): 
    def __init__(self):
        print("The functions are loaded")
        
    def load_data(self, file):
        file = pd.read_csv(file)
        print("The data is loaded...")
        return file
        
    def format_data(self, file):
        #creating a price column correlated to quality
        file["price"] = file["quality"] * 1.5 + file["alcohol"] * 0.3
        #adding noise
        mu, sigma = 0, 1
        noise = np.random.normal(mu, sigma, len(file["price"]))
        file["price"] = file["price"] + noise
        
        #adding irrelevant columns
        file["ID"]= np.random.randint(1, 900, file.shape[0])
        #file["age"]= np.random.randint(20, 85, file.shape[0])
        file["gender"]= np.random.randint(0, 1, file.shape[0])
        print("The data is formatted...")
        return file
    
    def divide_data(self, df): 
        #split df based on threshold 12, price is correlated with quality
        mask = df['price'] >= 12
        df1 = df[mask]
        df2 = df[~mask]
        print("Two datasets are generated...")
        return df1,df2
    
    def add_demographics(self, df1, df2): 
      	df1["age"]= np.random.randint(40, 45, df1.shape[0])
      	df2["age"]= np.random.randint(20, 85, df2.shape[0])
    
    def show_data(self, df):
        print("The first 20 values of the dataset: ")
        df.head(n=20)
    
    def set_target(self, df, y):
        print("Your prediction target is: ", y)
        y = df[y]
        return y
    
    def set_features(self, df, y):
        X = df.drop(y, axis = 1)
        print("Your features used for predictions are: ", X.columns)
        return X
    
    def split_data(self, df, X, y):
        test_size = 0.3
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = test_size)
        print("The data set is splitted into a train set and test set.")
        print("The test set contains ", (test_size*100), "% of the data.")
        return  X_train, X_test, y_train, y_test
        
    def train_model(self, X_train, y_train):
        mdl = RandomForestClassifier(random_state=42)
        mdl.fit(X_train, y_train)
        print("The Random Forest Classifier model is fitted to the data.")
        return mdl
    
    def make_prediction(self, mdl, X_test):
        pred = mdl.predict(X_test)
        return pred

    def evaluate_prediction(self, y_test, pred):
        correct = accuracy_score(y_test, pred)
        incorrect = 1 - correct
        eval_result = [correct, incorrect]
        print((correct*100), "% of the predictions were correct")
        print((incorrect*100), "% of the predictions were incorrect")
        return eval_result 
    
    def plot_evaluation_result(self, eval_result):
        labels = ['Correct predictions', 'Incorrect predictions']
        plt.pie(eval_result,  autopct='%1.0f%%', colors= ["green", "red"])
        plt.legend(labels)
        
    def plot_demographics(self, df):
        print("The distribution of age of the dataset is plotted...")
        sns.set(rc={'figure.figsize':(11.7,8.27)})
        b=sns.countplot(df['age'])
        b.tick_params(labelsize=8, rotation=90)
