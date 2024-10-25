import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

# Load the dataset
file_path = "/content/Textile_dataset.csv"
data = pd.read_csv(file_path)

# Encoding categorical features
label_encoders = {}
categorical_columns = ['Item', 'Material', 'Supplier', 'Location', 'Season', 'Quality', 'Availability']

for column in categorical_columns:
    le = LabelEncoder()
    data[column] = le.fit_transform(data[column])
    label_encoders[column] = le

# Selecting features and target variable
X = data[['Item', 'Material', 'Supplier', 'Location', 'Season', 'Cost Price', 'Quality', 'Availability']]
y = data['Sale Price']

# Splitting the data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Training Linear Regression model
linear_reg = LinearRegression()
linear_reg.fit(X_train, y_train)
y_pred_lr = linear_reg.predict(X_test)

# Training Random Forest model
random_forest = RandomForestRegressor(random_state=42)
random_forest.fit(X_train, y_train)
y_pred_rf = random_forest.predict(X_test)

# Evaluating the models
lr_mse = mean_squared_error(y_test, y_pred_lr)
lr_r2 = r2_score(y_test, y_pred_lr)

rf_mse = mean_squared_error(y_test, y_pred_rf)
rf_r2 = r2_score(y_test, y_pred_rf)

# Comparing the performance
print("Linear Regression:")
print(f"Mean Squared Error: {lr_mse}")
print(f"R2 Score: {lr_r2}")

print("\nRandom Forest:")
print(f"Mean Squared Error: {rf_mse}")
print(f"R2 Score: {rf_r2}")
