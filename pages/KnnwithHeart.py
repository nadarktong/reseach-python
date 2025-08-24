from sklearn.neighbors import KNeighborsClassifier
import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

st.title('การทำนายเว็บไซต์ฟิชชิ่งด้วยเทคนิค K-Nearest Neighbor (KNN)')

# โหลด dataset
dt = pd.read_csv("/mnt/data/Website Phishing.csv")

st.subheader("ข้อมูลส่วนแรก 10 แถว")
st.write(dt.head(10))

st.subheader("ข้อมูลส่วนสุดท้าย 10 แถว")
st.write(dt.tail(10))

