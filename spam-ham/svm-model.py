import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
import nltk
from nltk.corpus import stopwords
import string
import streamlit as st

# Download stopwords for text preprocessing
nltk.download('stopwords')

# Load the dataset (replace with the path to your dataset)
df = pd.read_csv("D:/PYTHON_ML/Data Sets/MailSpamDataset.csv")

# Preprocess the text (remove punctuation, lowercasing, and stopwords)
def preprocess_text(text):
    # Convert to lowercase
    text = text.lower()
    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))
    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    words = text.split()
    words = [word for word in words if word not in stop_words]
    return ' '.join(words)

# Apply preprocessing function to the 'message' column
df['message_content'] = df['message_content'].apply(preprocess_text)

# Features (X) and target (y)
X = df['message_content']
y = df['is_spam']

# Vectorize the text data using TfidfVectorizer
tfidf = TfidfVectorizer(max_features=5000, stop_words='english')
X_tfidf = tfidf.fit_transform(X)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_tfidf, y, test_size=0.2, random_state=42)

# Train the model (Support Vector Classifier - SVC)
model = SVC(kernel='linear')
model.fit(X_train, y_train)

# Predict on the test set
y_pred = model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
classification_rep = classification_report(y_test, y_pred)

# Function to predict spam or not based on user input
def predict_spam(user_input):
    user_input_processed = preprocess_text(user_input)
    user_input_tfidf = tfidf.transform([user_input_processed])
    prediction = model.predict(user_input_tfidf)
    
    if prediction == 1:
        return "The message is SPAM"
    else:
        return "The message is NOT SPAM"

# Streamlit user interface
st.title("Spam Message Classifier")
st.write("""
    This application predicts whether a message is spam or not. 
    Simply type a message below, and click on "Predict" to see the result.
""")

# Sidebar navigation for multiple pages
page = st.sidebar.radio("Select a Page", ["Predict Spam", "Model Performance"])

# Page 1: Predict Spam
if page == "Predict Spam":
    # Text input for user to enter a message
    user_input = st.text_area("Enter a message to check if it's spam or not:")

    # Button to trigger the prediction
    if st.button("Predict"):
        if user_input:
            result = predict_spam(user_input)
            st.success(result)
        else:
            st.warning("Please enter a message.")

# Page 2: Model Performance
elif page == "Model Performance":
    st.subheader("Model Performance Metrics")

    # Displaying accuracy and classification report
    st.write(f"**Accuracy on Test Set**: {accuracy:.2f}")
    st.text("**Classification Report**:")
    st.text(classification_rep)

