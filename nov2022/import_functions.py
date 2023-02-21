# Step 1-3 + 4 + 5 + 6 + 7
# Importing required libraries
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn import tree
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import learning_curve
import numpy as np
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.feature_selection import VarianceThreshold
import sklearn.metrics as metrics
import matplotlib.pyplot as plt

sns.set(rc={"figure.figsize": (12, 6)})
import warnings

warnings.filterwarnings("ignore")
from dsiad_functions import Check

check = Check()

# importing dataset
wine = pd.read_csv("winequality-red_2.csv")
wine_copy = pd.read_csv("winequality-red_2.csv")

# step 4
number_of_observations = 1599
mean_quality = 5.6
minimum_alcohol = 0
maximum_pH = 0


def load_wine_data():
    # importing dataset
    wine = pd.read_csv("winequality-red_2.csv")
    return wine


# step 5
def execute_pre_step_5():
    return load_wine_data()


# step 6
def execute_pre_step_6() -> pd.DataFrame:
    wine = execute_pre_step_5()
    wine["price"] = wine["price"].fillna(wine["price"].mean())

    # Dividing wine as good and bad by giving the limit for the quality
    bins = (1, 5.5, 10)
    group_names = ["bad", "good"]
    wine["quality"] = pd.cut(wine["quality"], bins=bins, labels=group_names)

    label_quality = LabelEncoder()

    wine["quality"] = label_quality.fit_transform(wine["quality"])
    return wine


# step 6
def execute_pre_step_7():
    wine = execute_pre_step_6()
    unrelated_features = ["ID", "age", "gender"]

    # this function ensures that the unrelated features are dropped
    wine = wine.drop(columns=unrelated_features, axis=1)
    correlated_features = "star_rating"
    wine = wine.drop(columns=correlated_features, axis=1)
    threshold = 0.005
    wine = wine.loc[:, wine.var() > threshold]
    return wine


# step 7
def execute_pre_step_8():
    wine = execute_pre_step_7()
    y = wine["quality"]
    X = wine.drop(["quality"], axis=1)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )

    # Defining the model
    dtc = tree.DecisionTreeClassifier()

    # Making prediction on the test set
    pred_dtc = dtc.predict(X_test)

    # Defining the model
    rfc = RandomForestClassifier(n_estimators=200)

    # fit the model to our training data
    rfc.fit(X_train, y_train)

    # Making prediction on the test set
    pred_rfc = rfc.predict(X_test)

    # Defining the model
    lrg = LogisticRegression()

    # fit the model to our training data
    lrg.fit(X_train, y_train)

    # Making prediction on the test set
    pred_lrg = lrg.predict(X_test)

    return wine, y_test, pred_dtc, pred_lrg, pred_rfc
