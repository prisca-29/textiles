# Import necessary libraries
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder

# Load the dataset
import pandas as pd
file_path = '/content/Textile_dataset.csv'
data = pd.read_csv(file_path)

# Prepare the data (selected features and label)
X = data[['Cost Price', 'Sale Price', '      MRP', 'Season', 'Quality']]  # Adjusted 'MRP' column name
y = data['Material']

# Convert categorical features to numeric using Label Encoding
X = X.copy()
label_encoders = {}
for column in ['Season', 'Quality']:
    le = LabelEncoder()
    X[column] = le.fit_transform(X[column])
    label_encoders[column] = le

# Encode the target labels
y = LabelEncoder().fit_transform(y)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Logistic Regression (similar to Linear Regression but for classification)
logistic_model = LogisticRegression(max_iter=200)
logistic_model.fit(X_train, y_train)
y_pred_logistic = logistic_model.predict(X_test)
accuracy_logistic = accuracy_score(y_test, y_pred_logistic)

# Random Forest Classifier (instead of Regressor)
rf_model = RandomForestClassifier(random_state=42)
rf_model.fit(X_train, y_train)
y_pred_rf = rf_model.predict(X_test)
accuracy_rf = accuracy_score(y_test, y_pred_rf)

# Print results for comparison
print(f"Logistic Regression Accuracy: {accuracy_logistic}")
print(f"Random Forest Accuracy: {accuracy_rf}")
print("\nClassification Report for Random Forest:")
print(classification_report(y_test, y_pred_rf))
