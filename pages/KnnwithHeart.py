from sklearn.neighbors import KNeighborsClassifier
import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

st.title('การทำนายเว็บไซต์ฟิชชิ่งด้วยเทคนิค K-Nearest Neighbor (KNN)')

# โหลด dataset
dt = pd.read_csv("./data/Website Phishing.csv")

st.subheader("ข้อมูลส่วนแรก 10 แถว")
st.write(dt.head(10))

st.subheader("ข้อมูลส่วนสุดท้าย 10 แถว")
st.write(dt.tail(10))

# สถิติพื้นฐาน
st.subheader("📈 สถิติพื้นฐานของข้อมูล")
st.write(dt.describe())

# การเลือกฟีเจอร์เพื่อดูการกระจายข้อมูล
st.subheader("📌 เลือกฟีเจอร์เพื่อดูการกระจายข้อมูล")
feature = st.selectbox("เลือกฟีเจอร์", dt.columns[:-1])

# วาดกราฟ boxplot
st.write(f"### 🎯 Boxplot: {feature} แยกตามผลลัพธ์")
fig, ax = plt.subplots()
sns.boxplot(data=dt, x='Result', y=feature, ax=ax)
st.pyplot(fig)

# Pairplot
if st.checkbox("แสดง Pairplot (ใช้เวลาประมวลผลเล็กน้อย)"):
    st.write("### 🌺 Pairplot: การกระจายของข้อมูลทั้งหมด")
    fig2 = sns.pairplot(dt, hue='Result')
    st.pyplot(fig2)

st.subheader("🔎 ทำนายข้อมูลเว็บไซต์")

# กรอกค่าฟีเจอร์
sfh = st.number_input('Insert SFH (-1, 0, 1)', -1, 1, 0)
popup_window = st.number_input('Insert popUpWidnow (-1, 0, 1)', -1, 1, 0)
ssl_final_state = st.number_input('Insert SSLfinal_State (-1, 0, 1)', -1, 1, 0)
request_url = st.number_input('Insert Request_URL (-1, 0, 1)', -1, 1, 0)
url_of_anchor = st.number_input('Insert URL_of_Anchor (-1, 0, 1)', -1, 1, 0)
web_traffic = st.number_input('Insert web_traffic (-1, 0, 1)', -1, 1, 0)
url_length = st.number_input('Insert URL_Length (-1, 0, 1)', -1, 1, 0)
age_of_domain = st.number_input('Insert age_of_domain (-1, 0, 1)', -1, 1, 0)
having_ip_address = st.number_input('Insert having_IP_Address (0, 1)', 0, 1, 0)

if st.button("ทำนายผล"):
    # แยก features/target
    X = dt.drop('Result', axis=1)
    y = dt['Result']

    # โมเดล KNN
    Knn_model = KNeighborsClassifier(n_neighbors=3)
    Knn_model.fit(X, y)  
    
    # สร้าง input
    x_input = np.array([[sfh, popup_window, ssl_final_state,
                         request_url, url_of_anchor, web_traffic,
                         url_length, age_of_domain, having_ip_address]])
    
    out = Knn_model.predict(x_input)

    label_map = {-1: "Suspicious", 0: "Legitimate", 1: "Phishing"}
    st.write(f"### ✅ ผลการทำนาย: {label_map.get(out[0], 'Unknown')}")
else:
    st.write("⏳ รอการทำนาย...")