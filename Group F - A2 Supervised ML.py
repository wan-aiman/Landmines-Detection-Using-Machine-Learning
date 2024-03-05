import pandas as pd

# Dataset location
path = 'g:/My Drive/degree/intelligence-systems/Python/Mine_Dataset.xls'

try:
    minedata = pd.read_excel(path)
    print(minedata.head())
except FileNotFoundError:
    print(f"File not found at the specified path: {path}")

# Display the first few rows of the dataset
print(minedata.head())

# Assign data from the first 3 columns to x variable
x = minedata.iloc[:, 0:3]

# Assign data from the last column to y variable
y = minedata.iloc[:, 3:4]

# Display unique values in the target column 'M'
print(y.M.unique())  # Output value

from sklearn import preprocessing
le = preprocessing.LabelEncoder()

# Convert categorical labels to numerical labels
y = y.apply(le.fit_transform)

# Display unique values in the transformed target column 'M'
print(y.M.unique())

# Train-test split
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.20)

# Feature scaling
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()

# Fit the scaler to the training data and transform both training and test data
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

# Training and predictions using Multi-Layer Perceptron (MLP) Classifier
from sklearn.neural_network import MLPClassifier
mlp = MLPClassifier(hidden_layer_sizes=(10, 10, 10), max_iter=1000)
mlp.fit(x_train, y_train.values.ravel())

# Making predictions on the test set
predictions = mlp.predict(x_test)

# Evaluating the algorithm
from sklearn.metrics import classification_report, confusion_matrix
print(confusion_matrix(y_test, predictions))
print(classification_report(y_test, predictions))
