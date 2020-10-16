from random import random

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pandas.api.types import is_string_dtype, is_numeric_dtype
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import mean_absolute_error, plot_confusion_matrix, plot_roc_curve, f1_score
from sklearn.model_selection import cross_val_score, train_test_split, cross_validate, GridSearchCV
from sklearn.preprocessing import MinMaxScaler
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.utils import shuffle
from sklearn.feature_selection import GenericUnivariateSelect, f_classif, f_regression, chi2

from xgboost import XGBClassifier, XGBRegressor
from joblib import dump, load
import os

class ML_Backend():
    """ML_Backed

    This class is a framework for automated machine learning
    Determines type e.g regression or classification
    Loads csv from specific file location
    Automated data cleaning based on certain heuristics
    Automated feature selection
    Data correlation plotting
    Model training with gridsearch and cross-validation
    Model training and testing metrics
    """

    def __init__(self, training = True, model_Types = ["LogisticRegression"], dataSetName="", dataSetFilePath="", gridParameters = {
        "LogisticRegression": {"C": np.logspace(-3,6,7), "penalty": ["l2"]},
        "RandomForestClassifier": {"n_estimators": [50, 100, 200], "criterion": ['gini', "entropy"], "bootstrap": [True, False]},
        "DecisionTreeClassifier": {"criterion": ['gini', "entropy"], "splitter": ["best", "random"], "max_depth": [100, 200, 500]},
        "XGBClassifier": {'nthread': [4], "objective": ["binary:logistic"], "learning_rate": [0.05], "max_depth": [6, 12], "min_child_weight": [11], "n_estimators": [1000, 2000, 4000]},
        "RandomForestRegressor": {"n_estimators": [50, 100, 200], "criterion": ['gini', "entropy"], "bootstrap": [True, False]},
        "DecisionTreeRegressor": {"criterion": ['gini', "entropy"], "splitter": ["best", "random"], "max_depth": [100, 200, 500]},
        "XGBRegressor": {'nthread': [4],  "learning_rate": [0.05], "max_depth": [6, 12], "min_child_weight": [11], "n_estimators": [1000, 2000, 4000]}
    }):
        """Initialise ML_Backend

        Keyword arguments:
        training -- choose whether to train or test trained models (default True)
        model_Types -- list of models to train or test (default ["LogisticRegression"])
        dataSetName -- choose a name for the dataset - this sets the save folder for the models (default "")
        dataSetFilePath -- filepath to dataset (default "")
        gridParameters -- parameters for gridsearch, dictionary in the form of {"modelname": {parameter: [values]}}
        
        The type of ML (classification or regression) is automatically determined here based on the first model_Type specified
        For the Case study it's a classification problem and the first model is a LogisticRegression
        """
        self.training = training
        self.model_Types = model_Types
        self.modelPaths = {}
        self.dataSetName = dataSetName
        for i in self.model_Types:
            self.modelPaths[i] = f"models/{self.dataSetName}/{i}model.joblib"
        self.model_Dictionary = {"LogisticRegression": LogisticRegression(max_iter=500, n_jobs=-1), "RandomForestClassifier": RandomForestClassifier(n_jobs=-1), "XGBClassifier": XGBClassifier(), "DecisionTreeRegressor": DecisionTreeRegressor(), "RandomForestRegressor": RandomForestRegressor(n_jobs=-1), "XGBRegressor": XGBRegressor(), "DecisionTreeClassifier": DecisionTreeClassifier()}
        self.dataSetFilePath = dataSetFilePath
        self.gridParameters = gridParameters

        self.type = "classification" if (("classifier" in self.model_Types[0].lower()) or ("logistic" in self.model_Types[0].lower())) else "regression"


    def add_Model(self, model):
        """Add another model to the models you wish to use in the backend
        
        Arguments:
        model -- the sklearn compatible model in string form (e.g. "XGBRegressor")
        """
        self.model_Types.append(model)
        self.modelPaths[model] = f"data/{self.dataSetName}/{model}model.joblib"

    def load_DataSet(self):
        """Loads csv data into a DF
        
        Data is loaded from the filepath specified in the __init__ function
        """

        self.DF = pd.read_csv(self.dataSetFilePath)
        print(f"Loading Data Set Successful.\nNo of rows: {len(self.DF)}\nNo. of columns: {len(self.DF.columns)}\n")
        print(self.DF.head())

    def check_Data(self):
        """Check some preliminary stats on the loaded data.

        Stats checked are:
        The columns in the dataframe
        The data types of the columns
        The statistical info of the numeric columns
        """
        print("Here are all the Columns\n")
        print(self.DF.columns)
        print("The Data Types for the Columns are as follows:\n")
        print(self.DF.dtypes)
        print("\nSome summary stats on the Numeric Columns\n")
        print(self.DF.describe())

    def data_Cleaning(self):
        """Automated cleaning of data in the dataframe
        
        The cleaning has the following steps:
        Drop all columns that are empty
        Drop all rows that are empty
        Impute missing data -- use the mean rounded to an integer for numeric columns, use the string "None" for string columns
        Remove columns which are constant valued
        """
        for i in self.DF.columns:
            print("This is the column -> " + str(i), "Amount of nulls/nas -> " + str(len(self.DF[i][self.DF[i].isnull()])))
        print("\nRemoving Rows that are Completely Empty")
        self.DF = self.DF.dropna(how="all")
        print("Removing Columns that are Completely Empty\n")
        self.DF = self.DF.dropna(axis=1,how="all")
        print("Adjusting columns with some missing values\n")
        for i in self.DF.columns:
            if is_numeric_dtype(self.DF[i]):
                self.DF.loc[self.DF[i].isnull(), i]= self.DF[i].mean().round()
            else:
                self.DF.loc[self.DF[i].isnull(), i]= "None"
        print("Removing Columns that have only one Value\n")
        self.DF = self.DF.loc[:, self.DF.nunique() != 1]    
        print(f"There are now {len(self.DF)} rows left.\n The Following Columns are left:\n")
        print(self.DF.columns)
        for i in self.DF.columns:
            print("This is the column -> " + str(i), "Amount of nulls/nas -> " + str(len(self.DF[i][self.DF[i].isnull()])))

    def data_Encoding(self, columnwise=False):
        """Encode the non-numeric columns

        Keyword arguements:
        columnwise -- if true expand a column to as many distinct values within it and then use one-hot encoding, when false transform unique values to category codes (default False)
        """
        self.columnwise = columnwise

        if columnwise:
            print("\nHistograms will not work when using columnwise one hot encoding\n")
            for i in self.DF.columns:
                if is_numeric_dtype(self.DF[i]):
                    pass
                else:
                    self.DF = pd.concat([self.DF.drop(i, axis=1), pd.get_dummies(self.DF[i], prefix=i)], axis=1)
        else:
            for i in self.DF.columns:
                if is_numeric_dtype(self.DF[i]):
                    pass
                else:
                    self.DF[i] = self.DF[i].astype("category").cat.codes


    def data_Correlation(self, histogram=False, target = "INCOME_CLASSIFIER"):
        """Plot correlation and histograms for data - relative to target column
        
        Keyword arguments:
        histogram -- plot histograms with heatmapping - can't be used with columnwise encoding (default false)
        target -- the target for correlation and histograms to be calculated with (default "INCOME_CLASSIFIER")
        """
        if histogram == True and self.columnwise == False:
            print("Prepping Axis\n")
            f, axes = plt.subplots(len(self.DF.columns), 1, figsize=(24,15*len(self.DF.columns)))
            print(f"{len(self.DF.columns)} axes have been prepped\n")

            print("Calculating correlation\n")
            # Entire DataFrame
            corr = self.DF.corr(method="kendall")
            print(f"The correlation for {len(self.DF.columns)} has been calculated\n")
            print("Plotting heatmap")
            sns.heatmap(corr, cmap='coolwarm_r', annot_kws={'size':20}, ax=axes[0])
            print("Setting Title")
            axes[0].set_title("Correlation Matrix", fontsize=14)

            print("Plotting Histograms")
            for i,t in enumerate(self.DF.drop([target], axis=1).columns):
                
                h = axes[i + 1].hist2d(self.DF[t], self.DF[target])
                plt.colorbar(h[3], ax=axes[i + 1])
                axes[i + 1].set_xlabel(t)
                axes[i + 1].set_ylabel(target)

            plt.show()
        else:
            print("Prepping Axis\n")
            f, axes = plt.subplots(1, 1, figsize=(24,15*1))
            print(f"1 axes have been prepped\n")

            print("Calculating correlation\n")
            # Entire DataFrame
            corr = self.DF.corr(method="kendall")
            print(f"The correlation for {len(self.DF.columns)} has been calculated\n")
            print("Plotting heatmap")
            sns.heatmap(corr, cmap='coolwarm_r', annot_kws={'size':20}, ax=axes)
            print("Setting Title")
            axes.set_title("Correlation Matrix", fontsize=14)

            plt.show()

    def feature_Selection(self, target=""):
        """Automated feature selection using sklearn GenericUnivariateSelect

        Keyword arguments:
        target -- target column for feature selection (default "")
        """
        data_set = self.DF
        y = data_set[target]
        X = data_set.drop(columns=[target])
        print(f"We are starting with the following columns:\n{X.columns}\n")
        transformer = GenericUnivariateSelect(f_classif if self.type == "classification" else f_regression, mode="fwe")
        self.data = transformer.fit_transform(X, y)
        columns_retained = self.DF.iloc[:, 1:].columns[transformer.get_support()].values
        self.DF = self.DF[columns_retained]
        
        print(f"The following columns are left:\n{self.DF.drop(columns=[target]).columns}")


    def normalise_Data(self):
        """Normalise columns to 0 - 1"""

        column_maxes = self.DF.max()
        column_mins = self.DF.min()
        self.DF = (self.DF - column_mins) / (column_maxes - column_mins)

    def data_Prep(self, target = "INCOME_CLASSIFIER", shuffleData=True):
        """Prep data by splitting into test and train sets

        Keyword arguments:
        target -- target columns, y value (default "INCOME_CLASSIFIER")
        shuffleData -- shuffle X and y data (default True)
        """
        data_set = self.DF
        print("Prepping data set")
        X =  data_set.drop(columns=[target])
        y = data_set[target]
        if shuffleData:
            X, y = shuffle(X, y)
        self.train_X, self.test_X, self.train_y, self.test_y = train_test_split(X, y, random_state=42)
        print("Data set prepped")
        

    def train_Model(self):
        """Train each of the selected models
        
        Uses GridSearchCV with built in crossvalidation
        Refit scoring is F1 for classification and neg_mean_squared_error for regression
        """

        self.models = []
        for i in self.model_Types:
            print(f"Training {i} model")
            self.model_Dictionary[i] = GridSearchCV(self.model_Dictionary[i], self.gridParameters[i], n_jobs=-1, cv=3, scoring=["f1", "recall", "precision"] if self.type == "classification" else ["neg_mean_squared_error", "neg_mean_absolute_error"], refit="f1" if self.type == "classification" else "neg_mean_squared_error").fit(self.train_X, self.train_y)
            print(f"{i} model has been fit")
            try:
                os.makedirs(f"Data/{self.dataSetName}")
            except:
                pass
            dump(self.model_Dictionary[i], self.modelPaths[i])
            print(f"{i} has been saved")

        

    def validate(self):
        """Validate trained models

        If classification f1 scores for training and validation set is printed, along with ROC and confusion matrix
        If regression mean_absolute_error is score for training and validation is printed
        """
        print("Testing models")
        self.val_predictions = {}
        for i in self.model_Types:
            scoretype = "f1" if self.type == "classification" else "neg_mean_squared_error"
            self.val_predictions[i] = self.model_Dictionary[i].predict(self.test_X)
            print(f"\nThe Best Hyperparameters for the {i} model was:\n{self.model_Dictionary[i].best_params_} and the best {scoretype} was {self.model_Dictionary[i].best_score_}")
            correct = 0
            total = 0

            for j in range(len(self.test_y)):
                if self.test_y.iloc[int(j)] == self.val_predictions[i][int(j)]:
                    correct += 1
                total += 1
            accuracy = correct / total * 100
            print(f"Overall accuracy for {i} is {accuracy}\n")
            print (f"The total validation records are {total}, and {correct} were correct\n")
            if self.type == "classification":
                print(f"The validation F1 score is: {f1_score(self.test_y, self.val_predictions[i])}\n")
                print(f"The confusion matrix:\n")
                plot_confusion_matrix(self.model_Dictionary[i], self.test_X,self.test_y)
                plt.show()
                print(f"The ROC curve:\n")
                plot_roc_curve(self.model_Dictionary[i], self.test_X,self.test_y, name=i)
                plt.show()
            else:
                print(f"The validation Mean Absolute Error score is: {mean_absolute_error(self.test_y, self.val_predictions[i])}")

        

    def load_Model(self):
        """Load models from saved files"""

        for i, t in enumerate(self.model_Types):
            print(f"Loading {t} model\n")
            try:
                self.model_Dictionary[t] = load(self.modelPaths[t])
            except:
                print(f"\nSomething went wrong :-(\nAre you sure you have trained a {t}\n")
                self.model_Types.pop(i)


    def execute_Model(self, columnwise=False, histogram=True, target="INCOME_CLASSIFIER", dropColumns=["SN_ID"]):
        """Execute either the training or testing of models in one function call

        Keyword arguments:
        columnswise -- if true expand a column to as many distinct values within it and then use one-hot encoding, when false transform unique values to category codes (default False)
        histogram -- plot histograms with heatmapping - can't be used with columnwise encoding (default False)
        target -- target column to be used in data_Correlation, feature_Selection, data_Prep functions
        dropColumns -- Columns to be dropped before and data manipulation occurs, list of strings (default ["SN_ID"])
        """
        if self.training:
            self.load_DataSet()
            self.check_Data()
            self.DF = self.DF.drop(dropColumns, axis=1)
            self.data_Cleaning()
            self.data_Encoding(columnwise)
            self.data_Correlation(histogram=histogram, target=target)
            self.feature_Selection(target=target)
            self.normalise_Data
            self.data_Prep(target=target, shuffleData=False)
            self.train_Model()
            self.validate()
        else:
            self.load_DataSet()
            self.check_Data()
            self.DF = self.DF.drop(dropColumns, axis=1)
            self.data_Cleaning()
            self.data_Encoding(columnwise)
            self.data_Correlation(histogram=histogram, target=target)
            self.feature_Selection(target=target)
            self.normalise_Data
            self.data_Prep(target=target, shuffleData=False)
            self.load_Model()
            self.validate()