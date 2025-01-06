# import pandas as pd
# from sklearn.preprocessing import MinMaxScaler

# data = df = pd.read_csv('D:\PYTHON_ML\Data Sets\Copper_Set.xlsx - Result 1.csv')

# # Parse the first sheet (assumes the dataset is in the first sheet)
# # df = data.parse(data.sheet_names[0])

# # Step 1: Clean Column Names (remove leading/trailing whitespaces)
# df.columns = df.columns.str.strip()

# # Step 2: Identify and Convert Scientific Notation Columns
# # Replace 'your_column_name' with the actual column name
# column_name = "thickness"  # Example column with scientific notation
# if column_name in df.columns:
#     try:
#         df[column_name] = df[column_name].astype(float)  # Convert scientific notation to float
#     except Exception as e:
#         print(f"Error converting column '{column_name}' to float: {e}")

# # Step 3: Handle Missing Values
# # Drop rows with missing values (optional: choose appropriate strategy)
# df = df.dropna()

# # Step 4: Remove Outliers (using IQR method)
# numeric_columns = df.select_dtypes(include=['float64', 'int64']).columns
# for col in numeric_columns:
#     Q1 = df[col].quantile(0.25)
#     Q3 = df[col].quantile(0.75)
#     IQR = Q3 - Q1
#     df = df[(df[col] >= Q1 - 1.5 * IQR) & (df[col] <= Q3 + 1.5 * IQR)]

# # Step 5: Normalize Numeric Data
# scaler = MinMaxScaler()
# df[numeric_columns] = scaler.fit_transform(df[numeric_columns])

# df.rename(columns={'item type':'item_type'}, inplace=True)

# df.rename(columns={'quantity tons':'quantity_tons'}, inplace=True)

# df.rename(columns={'delivery date':'delivery_date'}, inplace=True)

# Step 6: Save the Processed Data
# output_file = "Processed_Copper_Set.csv"
# df.to_csv(output_file, index=False)
# print(f"Processed dataset saved to {output_file}")

print("execution done")

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import LabelEncoder

data = df = pd.read_csv('Processed_Copper_Set.csv')

# Clean the data (remove whitespace, handle missing values, etc.)
df.columns = df.columns.str.strip()

# Step 1: Filter only WON and LOST status
df = df[df['status'].isin(['Won', 'Draft', 'To be approved', 'Lost', 'Not lost for AM',
       'Wonderful', 'Revised', 'Offered', 'Offerable'])]

# Step 2: Encode categorical variables (if any)
label_encoder = LabelEncoder()
df['status'] = label_encoder.fit_transform(df['status'])  # Convert WON/LOST to 1/0
df['item_type'] = label_encoder.fit_transform(df['item_type'])  # Convert categorical to numerical
df['application'] = label_encoder.fit_transform(df['application'])  # Convert categorical to numerical
df['material_ref'] = label_encoder.fit_transform(df['material_ref'])  # Convert categorical to numerical
df['product_ref'] = label_encoder.fit_transform(df['product_ref'])  # Convert categorical to numerical

# Step 3: Split into Features (X) and Target (y)
X = df[['quantity_tons','item_type', 'application', 'thickness', 'width','material_ref','product_ref']]  # Features (all columns except 'Status')
y = df['status']  # Target ('Status' column)

# Step 4: Handle missing values (optional) or fill in the missing values
# X = X.fillna(X.mean())  # Fill missing values with mean

# Step 5: Train-test split (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 6: Train the Random Forest Classifier
rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
rf_classifier.fit(X_train, y_train)

# Step 7: Make Predictions
y_pred = rf_classifier.predict(X_test)

# Step 8: Evaluate the Model
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred, zero_division=0))

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

# Sidebar for navigation
st.sidebar.title("Navigation")
pages = st.sidebar.radio("Go to:", ["Predict Status", "Model Performance", "Dashboard"])

# Page 1: Predict Status
if pages == "Predict Status":
    st.title("Copper Industry Lead Prediction")
    st.write("Enter values for the following columns:")

    inputs = {}
    for col in X.columns:
        inputs[col] = st.sidebar.number_input(f"Enter value for {col}", value=0.0)

    model_choice = st.sidebar.selectbox("Choose a model:", ["Random Forest Classifier", "Linear Regression"])

    if st.sidebar.button("Predict Status"):
        input_data = pd.DataFrame([inputs])
        if model_choice == "Random Forest Classifier":
            prediction = rf_classifier.predict(input_data)
            status = label_encoder.inverse_transform(prediction)[0]
            st.success(f"The predicted Status using Random Forest is: {status}")
        elif model_choice == "Linear Regression":
            prediction = linear_regressor.predict(input_data)
            predicted_class = int(round(prediction[0]))
            status = label_encoder.inverse_transform([predicted_class])[0]
            st.success(f"The predicted Status using Linear Regression is: {status}")

# Page 2: Model Performance
elif pages == "Model Performance":
    st.title("Model Performance Metrics")

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

# Page 3: Dashboard
elif pages == "Dashboard":
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
