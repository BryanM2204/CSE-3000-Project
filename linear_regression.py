import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

def run_linear_regression(X, y):
    """
    Runs linear regression and visualizes the results.
    """
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train the Linear Regression Model
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Make predictions
    y_pred = model.predict(X_test)

    # Evaluate the model
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print(f"Mean Squared Error (MSE): {mse:.2f}")
    print(f"R-squared (R²): {r2:.2f}")

    # Visualize the results
    plt.scatter(X_test, y_test, color='blue', label='Actual Data')
    plt.plot(X_test, y_pred, color='red', label='Regression Line')
    plt.xlabel('Feature (X)')
    plt.ylabel('Target (y)')
    plt.title('Linear Regression')
    plt.legend()
    plt.show()

    # Optional: Plot residuals
    residuals = y_test - y_pred
    plt.scatter(X_test, residuals, color='green', label='Residuals')
    plt.axhline(y=0, color='red', linestyle='--', label='Zero Residual Line')
    plt.xlabel('Feature (X)')
    plt.ylabel('Residuals')
    plt.title('Residual Plot')
    plt.legend()
    plt.show()