import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

# Load and preprocess the data
data = pd.read_csv('Processed_Copper_Set.csv')
data.columns = data.columns.str.strip()
data = data[data['status'].isin(['Won', 'Draft', 'To be approved', 'Lost', 'Not lost for AM',
                                 'Wonderful', 'Revised', 'Offered', 'Offerable'])]

label_encoder = LabelEncoder()
data['status'] = label_encoder.fit_transform(data['status'])
data['item_type'] = label_encoder.fit_transform(data['item_type'])
data['application'] = label_encoder.fit_transform(data['application'])
data['material_ref'] = label_encoder.fit_transform(data['material_ref'])
data['product_ref'] = label_encoder.fit_transform(data['product_ref'])

X = data[['quantity_tons', 'item_type', 'application', 'thickness', 'width', 'material_ref', 'product_ref']]
y = data['status']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train models
rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
rf_classifier.fit(X_train, y_train)

linear_regressor = LinearRegression()
linear_regressor.fit(X_train, y_train)

# Initialize session state for navigation
if "page" not in st.session_state:
    st.session_state.page = 1

# Function to navigate between pages
def navigate_to(page_num):
    st.session_state.page = page_num

# Page 1: User Inputs
if st.session_state.page == 1:
    st.title("Copper Industry Lead Prediction")
    st.write("### Enter values for the following features:")
    
    inputs = {}
    for col in X.columns:
        inputs[col] = st.number_input(f"{col}", value=0.0)

    if st.button("Next"):
        st.session_state.inputs = inputs
        navigate_to(2)

# Page 2: Model Selection and Predictions
elif st.session_state.page == 2:
    st.title("Model Selection and Predictions")
    st.write("### Choose a model:")
    model_choice = st.selectbox("Select a model:", ["Random Forest Classifier", "Linear Regression"])

    if st.button("Predict"):
        input_data = pd.DataFrame([st.session_state.inputs])
        if model_choice == "Random Forest Classifier":
            prediction = rf_classifier.predict(input_data)
            status = label_encoder.inverse_transform(prediction)[0]
            st.session_state.prediction = f"Predicted Status using Random Forest: {status}"
        elif model_choice == "Linear Regression":
            prediction = linear_regressor.predict(input_data)
            predicted_class = int(round(prediction[0]))
            status = label_encoder.inverse_transform([predicted_class])[0]
            st.session_state.prediction = f"Predicted Status using Linear Regression: {status}"
        navigate_to(3)

# Page 3: Evaluation Metrics
elif st.session_state.page == 3:
    st.title("Evaluation Metrics")
    st.write("### Prediction Result:")
    st.success(st.session_state.prediction)

    st.write("### Model Performance:")

    # Random Forest Metrics
    rf_y_pred = rf_classifier.predict(X_test)
    rf_accuracy = accuracy_score(y_test, rf_y_pred)
    st.subheader("Random Forest Classifier")
    st.write(f"Accuracy: {rf_accuracy:.2f}")
    st.text("Classification Report:")
    st.text(classification_report(y_test, rf_y_pred, zero_division=0))

    # Linear Regression Metrics
    linear_y_pred = linear_regressor.predict(X_test)
    linear_y_pred = [int(round(pred)) for pred in linear_y_pred]
    linear_accuracy = accuracy_score(y_test, linear_y_pred)
    st.subheader("Linear Regression")
    st.write(f"Accuracy: {linear_accuracy:.2f}")

    if st.button("Next"):
        navigate_to(4)

# Page 4: Visualizations
elif st.session_state.page == 4:
    st.title("Dashboard")

    # Line Chart: Quantity over Thickness
    st.subheader("Quantity vs Thickness")
    fig, ax = plt.subplots()
    sns.lineplot(x=data['thickness'], y=data['quantity_tons'], ax=ax)
    st.pyplot(fig)

    # Bar Graph: Item Type Distribution
    st.subheader("Item Type Distribution")
    fig, ax = plt.subplots()
    data['item_type'].value_counts().plot(kind='bar', ax=ax, color='skyblue')
    st.pyplot(fig)

    # Scatter Plot: Thickness vs Width
    st.subheader("Scatter Plot: Thickness vs Width")
    fig, ax = plt.subplots()
    sns.scatterplot(x=data['thickness'], y=data['width'], hue=data['status'], ax=ax)
    st.pyplot(fig)

    # Data Preview
    st.subheader("Data Overview")
    st.dataframe(data.head())

    if st.button("Start Over"):
        navigate_to(1)

