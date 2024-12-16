# Copper Industry Lead Prediction 🏭

Welcome to the **Copper Industry Lead Prediction** app! This tool is designed to help predict the status of copper-related projects using machine learning models. It includes a variety of options to analyze your data and make predictions.

## 📋 Data Processing Steps

### 1. **Data Cleaning** 🧹
The dataset is cleaned by:
- Removing any leading/trailing spaces in column names.
- Converting scientific notation columns to regular float format.
- Handling missing values by removing rows with missing data.
- Removing outliers using the IQR (Interquartile Range) method.
- Normalizing numeric data using MinMax scaling.

### 2. **Data Transformation** 🔄
- Columns such as **item_type**, **application**, **material_ref**, and **product_ref** are transformed using **Label Encoding** to convert categorical data into numerical format.

### 3. **Data Splitting** 🔀
The dataset is split into:
- **Features (X)**: Columns like `quantity_tons`, `item_type`, `thickness`, etc.
- **Target (y)**: The **status** of each project (Won/Lost).

### 4. **Model Training** 📊
We train two models:
- **Random Forest Classifier** (for classification of project status).
- **Linear Regression** (to predict status on a continuous scale).

### 5. **Model Evaluation** 📝
The performance of each model is evaluated using metrics like **accuracy** and the **classification report**.

## 📊 Dashboard Options

### 1. **Predict Status** 🔮
- **Input Features**: Enter values for key features such as `quantity_tons`, `item_type`, `thickness`, and more.
- **Choose Model**: Select either the **Random Forest Classifier** or **Linear Regression** to make predictions.
- **Prediction**: The app will output the predicted status of the project (e.g., **Won**, **Lost**).

### 2. **Model Performance** 📈
- **Random Forest Classifier**: View performance metrics such as **Accuracy** and a **Classification Report**.
- **Linear Regression**: See the accuracy of the regression model and evaluate its prediction performance.

### 3. **Data Insights** 🔍
- **Quantity vs Thickness**: A **Line Chart** displaying the relationship between `quantity_tons` and `thickness`.
- **Item Type Distribution**: A **Bar Chart** showing the distribution of `item_type`.
- **Thickness vs Width**: A **Scatter Plot** to analyze how `thickness` correlates with `width`.
- **Data Preview**: A snapshot of the first few rows of your dataset for quick reference.

## ⚙️ How to Use the App

### URL TO ACCESS - https://copperdata.streamlit.app/
### 📝 Enter Data
- Use the sidebar to enter feature values and select your model.
- Click the **"Predict Status"** button to get your result.

### 📊 View Performance
- Navigate to **"Model Performance"** to see metrics for both models.

### 📈 Explore the Dashboard
- Visualize your data with interactive charts under the **"Dashboard"** section.

## 📌 Key Features
- **Random Forest Classifier**: A powerful ensemble learning method that predicts project status.
- **Linear Regression**: A simple yet effective regression model to predict continuous outcomes.
- **Interactive Dashboard**: Visualize trends, distributions, and relationships in the dataset.

## 💬 Get in Touch
For any queries or feedback, feel free to reach out!

Good luck with your analysis and predictions! 👍
