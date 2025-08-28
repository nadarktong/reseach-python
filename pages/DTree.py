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

# Define features and target
features = ['SFH', 'popUpWidnow', 'SSLfinal_State', 'Request_URL',
            'URL_of_Anchor', 'web_traffic', 'URL_Length',
            'age_of_domain', 'having_IP_Address']
X = df[features]
y = df['Result']

# Train/test split
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=200)

# Train model
ModelDtree = DecisionTreeClassifier()
dtree = ModelDtree.fit(x_train, y_train)

st.subheader("กรุณาป้อนข้อมูลเพื่อวิเคราะห์ข้อมูล")

# Input fields
sfh = st.number_input('SFH (การส่งฟอร์ม, -1, 0, 1)', -1, 1, 0)
popup_window = st.number_input('popUpWidnow (หน้าต่างป๊อปอัพ, -1, 0, 1)', -1, 1, 0)
ssl_final_state = st.number_input('SSLfinal_State (สถานะ SSL, -1, 0, 1)', -1, 1, 0)
request_url = st.number_input('Request_URL (การเรียกใช้ URL, -1, 0, 1)', -1, 1, 0)
url_of_anchor = st.number_input('URL_of_Anchor (ลิงก์ Anchor, -1, 0, 1)', -1, 1, 0)
web_traffic = st.number_input('web_traffic (ปริมาณการเข้าชมเว็บ, -1, 0, 1)', -1, 1, 0)
url_length = st.number_input('URL_Length (ความยาว URL, -1, 0, 1)', -1, 1, 0)
age_of_domain = st.number_input('age_of_domain (อายุโดเมน, -1, 0, 1)', -1, 1, 0)
having_ip_address = st.number_input('having_IP_Address (มีการใช้ IP Address หรือไม่, 0, 1)', 0, 1, 0)

if st.button("Predict"):
    x_input = [[sfh, popup_window, ssl_final_state, request_url,
                url_of_anchor, web_traffic, url_length, 
                age_of_domain, having_ip_address]]
    y_pred = dtree.predict(x_input)[0]

    label_map = {-1: "Suspicious", 0: "Legitimate", 1: "Phishing"}
    st.write(f"The website is predicted to be: **{label_map.get(y_pred, 'Unknown')}**")

    
# Accuracy
y_predict = dtree.predict(x_test)
score = accuracy_score(y_test, y_predict)
st.write(f'Prediction accuracy: **{(score * 100):.2f}%**')

# Decision tree visualization
fig, ax = plt.subplots(figsize=(20, 15))
tree.plot_tree(dtree, feature_names=features,
               class_names=['Legitimate', 'Phishing', 'Suspicious'],
               filled=True, fontsize=10, ax=ax)
st.pyplot(fig)
