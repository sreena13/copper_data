import pandas as pd
from sklearn.preprocessing import MinMaxScaler

data = df = pd.read_csv('D:\PYTHON_ML\Data Sets\Copper_Set.xlsx - Result 1.csv')

# Parse the first sheet (assumes the dataset is in the first sheet)
# df = data.parse(data.sheet_names[0])

# Step 1: Clean Column Names (remove leading/trailing whitespaces)
df.columns = df.columns.str.strip()

# Step 2: Identify and Convert Scientific Notation Columns
# Replace 'your_column_name' with the actual column name
column_name = "thickness"  # Example column with scientific notation
if column_name in df.columns:
    try:
        df[column_name] = df[column_name].astype(float)  # Convert scientific notation to float
    except Exception as e:
        print(f"Error converting column '{column_name}' to float: {e}")

# Step 3: Handle Missing Values
# Drop rows with missing values (optional: choose appropriate strategy)
df = df.dropna()

# Step 4: Remove Outliers (using IQR method)
numeric_columns = df.select_dtypes(include=['float64', 'int64']).columns
for col in numeric_columns:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    df = df[(df[col] >= Q1 - 1.5 * IQR) & (df[col] <= Q3 + 1.5 * IQR)]

# Step 5: Normalize Numeric Data
scaler = MinMaxScaler()
df[numeric_columns] = scaler.fit_transform(df[numeric_columns])

df.rename(columns={'item type':'item_type'}, inplace=True)

df.rename(columns={'quantity tons':'quantity_tons'}, inplace=True)

df.rename(columns={'delivery date':'delivery_date'}, inplace=True)

# Step 6: Save the Processed Data
output_file = "Processed_Copper_Set.csv"
df.to_csv(output_file, index=False)
print(f"Processed dataset saved to {output_file}")

print("execution done")

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import LabelEncoder

data = df = pd.read_csv('D:\PYTHON_ML\Data Sets\Processed_Copper_Set.csv')

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
from sklearn.metrics import accuracy_score

# Load and preprocess the data
data = pd.read_csv('D:\PYTHON_ML\Data Sets\Processed_Copper_Set.csv')

# Clean the data
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

# Streamlit user interface
st.title("Copper Industry Lead Prediction")

st.write("Enter values for the following columns:")

# Input fields for each column
inputs = {}
for col in X.columns:
    inputs[col] = st.number_input(f"Enter value for {col}", value=0.0)

# Model selection
model_choice = st.selectbox("Choose a model:", ["Random Forest Classifier", "Linear Regression"])

# Make prediction based on inputs
if st.button("Predict Status"):
    input_data = pd.DataFrame([inputs])
    
    if model_choice == "Random Forest Classifier":
        prediction = rf_classifier.predict(input_data)
        status = label_encoder.inverse_transform(prediction)[0]
        st.write(f"The predicted Status using Random Forest is: {status}")
    
    elif model_choice == "Linear Regression":
        prediction = linear_regressor.predict(input_data)
        predicted_class = int(round(prediction[0]))
        status = label_encoder.inverse_transform([predicted_class])[0]
        st.write(f"The predicted Status using Linear Regression is: {status}")
