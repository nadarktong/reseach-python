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

st.subheader("Please enter data for prediction")

# Input fields
sfh = st.number_input('Insert SFH (-1, 0, or 1)', -1, 1, 0)
popup_window = st.number_input('Insert popUpWidnow (-1, 0, or 1)', -1, 1, 0)
ssl_final_state = st.number_input('Insert SSLfinal_State (-1, 0, or 1)', -1, 1, 0)
request_url = st.number_input('Insert Request_URL (-1, 0, or 1)', -1, 1, 0)
url_of_anchor = st.number_input('Insert URL_of_Anchor (-1, 0, or 1)', -1, 1, 0)
web_traffic = st.number_input('Insert web_traffic (-1, 0, or 1)', -1, 1, 0)
url_length = st.number_input('Insert URL_Length (-1, 0, or 1)', -1, 1, 0)
age_of_domain = st.number_input('Insert age_of_domain (-1, 0, or 1)', -1, 1, 0)
having_ip_address = st.number_input('Insert having_IP_Address (0 or 1)', 0, 1, 0)

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
