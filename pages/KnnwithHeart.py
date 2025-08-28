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

st.subheader("🔎 วิเคราะห์ข้อมูลเว็บไซต์")

# กรอกค่าฟีเจอร์
sfh = st.number_input('SFH (การส่งฟอร์ม, -1, 0, 1)', -1, 1, 0)
popup_window = st.number_input('popUpWidnow (หน้าต่างป๊อปอัพ, -1, 0, 1)', -1, 1, 0)
ssl_final_state = st.number_input('SSLfinal_State (สถานะ SSL, -1, 0, 1)', -1, 1, 0)
request_url = st.number_input('Request_URL (การเรียกใช้ URL, -1, 0, 1)', -1, 1, 0)
url_of_anchor = st.number_input('URL_of_Anchor (ลิงก์ Anchor, -1, 0, 1)', -1, 1, 0)
web_traffic = st.number_input('web_traffic (ปริมาณการเข้าชมเว็บ, -1, 0, 1)', -1, 1, 0)
url_length = st.number_input('URL_Length (ความยาว URL, -1, 0, 1)', -1, 1, 0)
age_of_domain = st.number_input('age_of_domain (อายุโดเมน, -1, 0, 1)', -1, 1, 0)
having_ip_address = st.number_input('having_IP_Address (มีการใช้ IP Address หรือไม่, 0, 1)', 0, 1, 0)

if st.button("วิเคราะห์ข้อมูล"):
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
    st.write(f"### ✅ ผลการวิเคราะห์ข้อมูล: {label_map.get(out[0], 'Unknown')}")
else:
    st.write("⏳ รอวิเคราะห์ข้อมูล...")