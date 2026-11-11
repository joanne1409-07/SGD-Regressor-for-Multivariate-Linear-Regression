## SGD Regressor for Multivariate Linear Regression
# 1. Aim
To develop a Python program using the Stochastic Gradient Descent (SGD) Regressor to predict the median house price and average number of occupants per household using the California Housing dataset.
# 2. Software and Hardware Requirements
Hardware
- PC or laptop
- Minimum 4 GB RAM recommended
Software
- Python 3.x
- Anaconda distribution
- Jupyter Notebook or any Python IDE
- Required Python libraries:
  - NumPy
  - Scikit-learn
# 3. Introduction
Multivariate linear regression is a regression technique that uses multiple input features to predict a target value. In this experiment, the model uses selected housing features to predict two outputs: median house value and average number of occupants per household.

Stochastic Gradient Descent (SGD) is an optimization method that updates model parameters using individual training samples or small batches. SGDRegressor in Scikit-learn trains a linear regression model using this method.

Since this experiment predicts two target variables, MultiOutputRegressor is used. It creates and trains one SGDRegressor model for each target variable.
# 4. Dataset Description
The program uses the California Housing dataset available through Scikit-learn.
Input features
The program selects the first three features:
1. MedInc – Median income in the block group.
2. HouseAge – Median house age in the block group.
3. AveRooms – Average number of rooms per household.
Target variables
1. MedHouseVal – Median house value, expressed in units of $100,000.
2. AveOccup – Average number of household occupants.
Note: AveOccup represents the average number of occupants per household in a block group. It is not an exact count of occupants in an individual house. Also, the house-price target is scaled in units of $100,000, not individual dollars.
# 5. Algorithm
1. Import NumPy and the required Scikit-learn libraries.
2. Load the California Housing dataset.
3. Select the first three columns as input features.
4. Select median house value and average occupancy as the two target variables.
5. Divide the dataset into training and testing sets.
6. Apply StandardScaler to standardize the input features and target variables. Fit the scalers only on the training data.
7. Create an SGDRegressor model and wrap it with MultiOutputRegressor.
8. Train the model using the scaled training data.
9. Predict the target values for the test data.
10. Convert the predictions back to their original units.
11. Evaluate the model using Mean Squared Error (MSE) and R² score.
12. Display actual and predicted values for the first five test samples.
# 6. Python Program
```
"""
Program: SGD Regressor for Multivariate Linear Regression
Aim: Predict median house value and average number of occupants
     using the California Housing dataset.

Developed by: Joanne Jaison
Register Number: 212225020017
"""

import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.linear_model import SGDRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score


# 1. Load the California Housing dataset
data = fetch_california_housing()

# Select the first three features as input
X = data.data[:, :3]

# Select house value and average occupants as targets
Y = np.column_stack((data.target, data.data[:, 5]))

print("Input shape:", X.shape)
print("Target shape:", Y.shape)

# 2. Split the dataset into training and testing sets
X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.2, random_state=42
)

# 3. Scale the input features and target values
scaler_X = StandardScaler()
scaler_Y = StandardScaler()

X_train_scaled = scaler_X.fit_transform(X_train)
X_test_scaled = scaler_X.transform(X_test)

Y_train_scaled = scaler_Y.fit_transform(Y_train)

# 4. Create and train the multi-output SGD Regressor
sgd = SGDRegressor(
    max_iter=1000,
    tol=1e-3,
    eta0=0.01,
    learning_rate="constant",
    random_state=42
)

model = MultiOutputRegressor(sgd)
model.fit(X_train_scaled, Y_train_scaled)

# 5. Predict the target values
Y_pred_scaled = model.predict(X_test_scaled)

# Convert predictions back to the original units
Y_pred = scaler_Y.inverse_transform(Y_pred_scaled)

# 6. Evaluate the model
mse = mean_squared_error(Y_test, Y_pred)
r2 = r2_score(Y_test, Y_pred, multioutput="raw_values")

print("\\nMean Squared Error:", mse)
print("R2 Score for house price:", r2[0])
print("R2 Score for average occupants:", r2[1])

# 7. Display actual and predicted values for five test samples
print("\\nFirst five predictions:")
for i in range(5):
    print("\\nExample", i + 1)
    print("Actual house price:", Y_test[i, 0])
    print("Predicted house price:", Y_pred[i, 0])
    print("Actual average occupants:", Y_test[i, 1])
    print("Predicted average occupants:", Y_pred[i, 1])
```

# 7. Explanation of Important Functions
- fetch_california_housing() – Loads the California Housing dataset.
- train_test_split() – Divides the dataset into training and testing data.
- StandardScaler() – Standardizes the features and target values.
- SGDRegressor() – Creates a linear regression model trained using stochastic gradient descent.
- MultiOutputRegressor() – Allows the estimator to predict multiple target variables.
- fit() – Trains the model.
- predict() – Predicts the target values for test data.
- mean_squared_error() – Calculates the average squared difference between actual and predicted values.
- r2_score() – Measures how well the model explains variation in the target values.
- inverse_transform() – Converts standardized predictions back to their original scale.
# 8. Output









# 9. Result
Thus, the Python program to implement multivariate linear regression using the SGD Regressor for predicting median house value and average number of occupants per household was written and executed.
