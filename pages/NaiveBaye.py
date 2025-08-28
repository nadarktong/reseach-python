from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
import pandas as pd
import streamlit as st

st.title("Naive Bayes: การจำแนกเว็บไซต์ฟิชชิ่ง")

# โหลดข้อมูล Website Phishing
df = pd.read_csv("./data/Website Phishing.csv")

# แยก features และ target
X = df.drop('Result', axis=1)
y = df['Result']

# แบ่งข้อมูล train/test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# สร้างและฝึกโมเดล Naive Bayes
clf = GaussianNB()
clf.fit(X_train, y_train)

st.subheader("กรุณาป้อนข้อมูลเพื่อพยากรณ์")

sfh = st.number_input('SFH (การส่งฟอร์ม, -1, 0, 1)', -1, 1, 0)
popup_window = st.number_input('popUpWidnow (หน้าต่างป๊อปอัพ, -1, 0, 1)', -1, 1, 0)
ssl_final_state = st.number_input('SSLfinal_State (สถานะ SSL, -1, 0, 1)', -1, 1, 0)
request_url = st.number_input('Request_URL (การเรียกใช้ URL, -1, 0, 1)', -1, 1, 0)
url_of_anchor = st.number_input('URL_of_Anchor (ลิงก์ Anchor, -1, 0, 1)', -1, 1, 0)
web_traffic = st.number_input('web_traffic (ปริมาณการเข้าชมเว็บ, -1, 0, 1)', -1, 1, 0)
url_length = st.number_input('URL_Length (ความยาว URL, -1, 0, 1)', -1, 1, 0)
age_of_domain = st.number_input('age_of_domain (อายุโดเมน, -1, 0, 1)', -1, 1, 0)
having_ip_address = st.number_input('having_IP_Address (มีการใช้ IP Address หรือไม่, 0, 1)', 0, 1, 0)

if st.button("วิเคราะห์"):
    # เตรียมข้อมูล input
    x_input = [[sfh, popup_window, ssl_final_state,
                request_url, url_of_anchor, web_traffic,
                url_length, age_of_domain, having_ip_address]]
    
    y_predict = clf.predict(x_input)
    
    label_map = {-1: "Suspicious", 0: "Legitimate", 1: "Phishing"}
    st.write(f"### ✅ ผลการพยากรณ์: {label_map.get(y_predict[0], 'Unknown')}")
    
    st.button("ไม่วิเคราะห์")
else:
    st.button("ไม่วิเคราะห์")
