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

Step 6: Save the Processed Data
output_file = "Processed_Copper_Set.csv"
df.to_csv(output_file, index=False)
print(f"Processed dataset saved to {output_file}")
