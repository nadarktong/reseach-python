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

sfh = st.number_input('Insert SFH (-1, 0, 1)', -1, 1, 0)
popup_window = st.number_input('Insert popUpWidnow (-1, 0, 1)', -1, 1, 0)
ssl_final_state = st.number_input('Insert SSLfinal_State (-1, 0, 1)', -1, 1, 0)
request_url = st.number_input('Insert Request_URL (-1, 0, 1)', -1, 1, 0)
url_of_anchor = st.number_input('Insert URL_of_Anchor (-1, 0, 1)', -1, 1, 0)
web_traffic = st.number_input('Insert web_traffic (-1, 0, 1)', -1, 1, 0)
url_length = st.number_input('Insert URL_Length (-1, 0, 1)', -1, 1, 0)
age_of_domain = st.number_input('Insert age_of_domain (-1, 0, 1)', -1, 1, 0)
having_ip_address = st.number_input('Insert having_IP_Address (0, 1)', 0, 1, 0)

if st.button("พยากรณ์"):
    # เตรียมข้อมูล input
    x_input = [[sfh, popup_window, ssl_final_state,
                request_url, url_of_anchor, web_traffic,
                url_length, age_of_domain, having_ip_address]]
    
    y_predict = clf.predict(x_input)
    
    label_map = {-1: "Suspicious", 0: "Legitimate", 1: "Phishing"}
    st.write(f"### ✅ ผลการพยากรณ์: {label_map.get(y_predict[0], 'Unknown')}")
    
    st.button("ไม่พยากรณ์")
else:
    st.button("ไม่พยากรณ์")
