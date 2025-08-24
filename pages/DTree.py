import pandas as pd
import streamlit as st
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

st.header("Decision Tree for Phishing Website Classification")

# Load the dataset
df = pd.read_csv("MultipleFiles/Website Phishing.csv")
st.write(df.head(10))

# Define features and target variable
# The features are all columns except 'Result'
features = ['SFH', 'popUpWidnow', 'SSLfinal_State', 'Request_URL', 'URL_of_Anchor', 'web_traffic', 'URL_Length', 'age_of_domain', 'having_IP_Address']
X = df[features]
y = df['Result'] # The target variable is 'Result'

# Split the data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=200)

# Initialize and train the Decision Tree Classifier
ModelDtree = DecisionTreeClassifier()
dtree = ModelDtree.fit(x_train, y_train)

st.subheader("Please enter data for prediction")

# Create input fields for each feature
sfh = st.number_input('Insert SFH (-1, 0, or 1)', min_value=-1, max_value=1, value=0)
popup_window = st.number_input('Insert popUpWidnow (-1, 0, or 1)', min_value=-1, max_value=1, value=0)
ssl_final_state = st.number_input('Insert SSLfinal_State (-1, 0, or 1)', min_value=-1, max_value=1, value=0)
request_url = st.number_input('Insert Request_URL (-1, 0, or 1)', min_value=-1, max_value=1, value=0)
url_of_anchor = st.number_input('Insert URL_of_Anchor (-1, 0, or 1)', min_value=-1, max_value=1, value=0)
web_traffic = st.number_input('Insert web_traffic (-1, 0, or 1)', min_value=-1, max_value=1, value=0)
url_length = st.number_input('Insert URL_Length (-1, 0, or 1)', min_value=-1, max_value=1, value=0)
age_of_domain = st.number_input('Insert age_of_domain (-1, 0, or 1)', min_value=-1, max_value=1, value=0)
having_ip_address = st.number_input('Insert having_IP_Address (0 or 1)', min_value=0, max_value=1, value=0)

if st.button("Predict"):
    # Prepare input data for prediction
    x_input = [[sfh, popup_window, ssl_final_state, request_url, url_of_anchor, web_traffic, url_length, age_of_domain, having_ip_address]]
    y_predict2 = dtree.predict(x_input)
    
    # Map the numerical prediction to a more understandable label
    prediction_label = "Phishing" if y_predict2[0] == 1 else "Legitimate" if y_predict2[0] == 0 else "Suspicious"
    st.write(f"The website is predicted to be: **{prediction_label}**")
    st.button("Do Not Predict")
else:
    st.button("Do Not Predict")

# Evaluate the model's accuracy on the test set
y_predict = dtree.predict(x_test)
score = accuracy_score(y_test, y_predict)
st.write(f'Prediction accuracy: **{(score * 100):.2f}%**')

# Visualize the Decision Tree
fig, ax = plt.subplots(figsize=(20, 15)) # Increased figure size for better readability
tree.plot_tree(dtree, feature_names=features, class_names=['Legitimate', 'Phishing', 'Suspicious'], filled=True, ax=ax, fontsize=10)
st.pyplot(fig)
