import pandas as pd 
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from matplotlib import pyplot as plt
import seaborn as sns 
import numpy as np


headers = ["symboling","normalized-losses","make","fuel-type","aspiration", "num-of-doors","body-style",
         "drive-wheels","engine-location","wheel-base", "length","width","height","curb-weight","engine-type",
         "num-of-cylinders", "engine-size","fuel-system","bore","stroke","compression-ratio","horsepower",
         "peak-rpm","city-mpg","highway-mpg","price"]

df = pd.read_csv(r"G:\My Drive\Kubicle and other Certificates\Data Science\usedcars.txt", names = headers)
df["highway-mpg"] = pd.to_numeric(df["highway-mpg"], errors= "coerce")
df["price"] = pd.to_numeric(df["price"], errors = "coerce")
df["engine-size"] = pd.to_numeric(df["curb-weight"], errors= "coerce")
df["engine-size"] = pd.to_numeric(df["engine-size"], errors= "coerce")
df["horsepower"] = pd.to_numeric(df["horsepower"], errors = "coerce")
df = df.dropna(subset=['highway-mpg', 'price',"horsepower", "curb-weight", "engine-size"])

# Steps: 1) Normalisation 2) Polynomial transform 3) Linear Regression

 # Create List of tuples

Input = [("polynomial",PolynomialFeatures(degree=2)), ("scale", StandardScaler()), ("Model", LinearRegression)]
Pipe = Pipeline(Input)

# Train model 
Pipe.fit(df[["horsepower", "curb-weight", "engine-size", "highway-mpg"]], y )
X = df["price"]
yhat = Pipe.predict(X[["horsepower", "curb-weight", "engine-size", "highway-mpg"]])