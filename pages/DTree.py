import pandas as pd
import streamlit as st
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

st.header("Decision Tree for Phishing Website Classification")

# Load dataset
df = pd.read_csv("./data/Website Phishing.csv")
st.write(df.head(10))

