# Fundamental Forecasts Project

## Overview
This project focuses on analyzing and forecasting financial fundamentals using machine learning techniques. It utilizes a dataset of financial features to predict target variables, likely key financial metrics or performance indicators.

## Objective
The objective is to develop a linear model that accurately predicts quarterly revenue growth for companies using a dataset of financial and economic indicators. The model will utilize 4,000 different features from 15,000 observations to make these predictions. Success will be measured by the model's Mean Squared Error (MSE) performance on a test dataset, aiming to provide reliable revenue growth forecasts for investment decision-making.

## Table of Contents
- [Setup](#setup)
- [Data](#data)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Model Details](#model-details)
- [Model Training](#model-training)
- [Model Evaluation](#model-evaluation)
- [Risk Assessment: Value at Risk (VaR)](#risk-assessment-value-at-risk-var)
- [Model Application](#model-application)
- [Future Work](#future-work)
- [Contributing](#contributing)

## Setup

### Prerequisites
- Python 3.x
- Jupyter Notebook
- Required libraries: pandas, numpy, jax, matplotlib, optax

### Installation
1. Clone this repository:
   git clone https://github.com/yourusername/fundamental-forecasts.git
   cd fundamental-forecasts

2. Install the required packages:
   pip install pandas numpy jax matplotlib optax

## Data
The project uses two main data files:
- `X.npy`: Feature matrix
- `y.npy`: Target variable

**Note:** Due to data privacy, these files are not included in the repository. Ensure you have the necessary data files in the same directory as the notebook before running the analysis.

## Usage
1. Place your `X.npy` and `y.npy` files in the same directory as the Jupyter notebook.

2. Open the Jupyter notebook:
   jupyter notebook Fundamental_Forecasts.ipynb

3. Run the cells in the notebook to:
   - Load the data
   - Split it into training (80%), validation (10%), and test (10%) sets
   - Prepare the data for further analysis or model training

## Project Structure
fundamental-forecasts/
│
├── Fundamental_Forecasts.ipynb  # Main Jupyter notebook for analysis
└── README.md                    # This file

## Model Details

### Regularization
The project implements L2 regularization to prevent overfitting. The penalty function is defined as:

def penalty(θ, λ, X):
    return λ / 2 * jnp.sum(θ**2)

where `θ` represents the model parameters, `λ` is the regularization strength, and `X` is the input feature matrix.

### Hyperparameter Tuning
The model uses grid search for hyperparameter tuning. The following hyperparameters are explored:

- Learning rates (`α`): [0.1, 0.05, 0.01, 1e-3, 1e-4, 1e-6]
- Momentum coefficients (`beta`): [0, 0.2, 0.5, 0.9, 0.99, 0.999]
- Adaptive learning rate coefficients (`beta2`): [0, 0.2, 0.5, 0.9, 0.99, 0.999]
- Regularization strengths (`λ`): [0.01, 0.1, 0.2, 0.5]
- Batch sizes: [10, 20, 50, 100]
- Number of models in ensemble: [1, 10, 20, 50]

These hyperparameters are crucial for optimizing the model's performance and will be used in the training process.

### Key Functions

1. `stack(Θ)`: Creates an array of repeated model parameters for ensemble modeling.
2. `mse(a, b)`: Calculates the Mean Squared Error between two arrays.
3. `plot(x, y)`: Visualizes the model's predictions against actual values.
4. `evaluate(Θ)`: JIT-compiled function to evaluate the model on the validation set.
5. `test(Θ)`: Tests the model performance on the test set.
6. `update(Θ, opt_state, X, y, α, β, β2, λ)`: JIT-compiled function to update model parameters using the Adam optimizer.
7. `sample(batch_size)`: Samples a random batch from the training data for mini-batch gradient descent.

### Optimization

The model uses the Adam optimizer for parameter updates. The update function includes both the mean squared error loss and L2 regularization.

### Visualization

The `plot` function allows for visual comparison between the model's predictions and actual values, which can be useful for assessing model performance and identifying potential issues.

## Model Training

The project implements an ensemble learning approach with hyperparameter tuning:

1. **Hyperparameter Selection**: For each model in the ensemble, hyperparameters (learning rate, momentum coefficients, regularization strength) are randomly selected from predefined lists.
2. **Ensemble Creation**: Multiple models are initialized with different hyperparameters to form an ensemble.
3. **Training Loop**: Each model in the ensemble is trained for 5000 iterations using mini-batch gradient descent. The Adam optimizer is used for parameter updates.
4. **Best Model Selection**: After training, the best-performing model from the ensemble is selected based on the lowest validation Mean Squared Error (MSE).
5. **Hyperparameter Exploration**: The process is repeated for different batch sizes and ensemble sizes to explore their impact on model performance.

### Training Parameters
- Number of iterations: 5000
- Batch sizes explored: [10, 20, 50, 100]
- Ensemble sizes explored: [1, 10, 20, 50]

### Model Selection
The best model is selected based on the lowest validation MSE. The following information is stored for each best model:
- Learning rate (α)
- Regularization strength (λ)
- Momentum coefficients (β, β2)
- Batch size
- Number of models in the ensemble
- Validation MSE
- Model parameters (Θ)

This approach allows for a comprehensive exploration of hyperparameters and model configurations to find the optimal setup for the financial forecasting task.

## Model Evaluation

After training and selecting the best model based on validation performance, the model is evaluated on the test set to assess its generalization capability.

### Testing Process

1. **Test Set Evaluation**: The best model, selected based on validation performance, is evaluated on the previously unseen test set.
2. **Performance Metric**: Mean Squared Error (MSE) is used as the primary metric to quantify the model's performance on the test set.
3. **Visualization**: A scatter plot is generated to visually compare the model's predictions against the actual values in the test set. This plot helps in identifying any patterns or biases in the model's predictions.

### Key Outputs

- **Test MSE**: The Mean Squared Error on the test set, providing a quantitative measure of the model's predictive accuracy.
- **Prediction Plot**: A visual representation of the model's predictions vs. actual values, allowing for qualitative assessment of the model's performance.
- **Final Model Parameters**: The parameters (θ) of the best-performing model are stored for potential future use or analysis.

This evaluation phase provides crucial insights into the model's ability to generalize to new, unseen data, which is essential for assessing its practical utility in financial forecasting tasks.

## Risk Assessment: Value at Risk (VaR)

In addition to evaluating the model's predictive accuracy, this project includes a risk assessment component using the Value at Risk (VaR) metric.

### Value at Risk Calculation

The project implements a Historical Value at Risk calculation based on the model's predictions and actual values:

1. **Return Calculation**: Actual and predicted returns are calculated based on the test set values and model predictions.
2. **Residual Calculation**: Residuals are computed as the difference between actual and predicted returns.
3. **Historical VaR**: The Historical VaR is calculated at a 95% confidence level using the percentile method on the residuals.

### Key Features

- **Confidence Level**: The VaR is calculated at a 95% confidence level, which is a standard in financial risk management.
- **Historical Approach**: The VaR calculation uses the historical simulation approach, which doesn't assume any particular distribution of returns.
- **Model Integration**: By using both predicted and actual values, the VaR calculation incorporates the model's predictive capability into the risk assessment.

### Interpretation

The resulting VaR represents the maximum expected loss at the 95% confidence level, based on the historical distribution of the model's prediction errors. This provides valuable insight into the potential downside risk associated with using the model for financial forecasting.

This risk assessment component enhances the project's practical utility by providing a quantitative measure of the potential risk associated with the model's predictions.

## Model Application

After training, evaluating, and assessing the risk of our model, we've encapsulated its predictive capability in a simple, reusable function.

### Prediction Function

The project includes a streamlined prediction function that can be easily applied to new data:

def f(X):
    return X @ final_θ

This function takes a feature matrix `X` as input and returns predictions using the parameters (`final_θ`) of the best-performing model.

### Key Features

- **Simplicity**: The function is concise and easy to understand, making it straightforward to integrate into other systems or workflows.
- **Vectorized Operation**: By using matrix multiplication, the function can efficiently generate predictions for multiple samples at once.
- **Flexibility**: The function can be applied to any properly formatted feature matrix, allowing for easy predictions on new, unseen data.

### Usage

To make predictions on new data:

1. Ensure your new data is formatted similarly to the training data (same number and order of features).
2. Call the function with your new data: `predictions = f(new_data)`.

This prediction function serves as the practical output of the project, allowing the trained model to be easily applied to new financial data for forecasting purposes.

## Future Work
- Implement various machine learning models for prediction
- Conduct feature importance analysis
- Visualize results and derive financial insights
- Optimize model performance and conduct hyperparameter tuning

## Contributing
Contributions are welcome! Please feel free to submit a Pull Request.

---
