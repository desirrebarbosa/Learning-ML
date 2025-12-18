# Cheat Sheet: Linear and Logistic Regression

## 1. Comparing Regression Types

### Simple Linear Regression

**Purpose:** To predict a dependent variable ($y$) based on one independent variable ($x$).

* **Pros:** Easy to implement, interpret, and efficient for small datasets.
* **Cons:** Not suitable for complex relationships; prone to underfitting.
* **Modeling Equation:** 

$$y = \beta_0 + \beta_1x_1$$

```python
from sklearn.linear_model import LinearRegression

# X must be reshaped if it is a single feature array
# X = x.reshape(-1, 1)
model = LinearRegression()
model.fit(X, y)
```

### Polynomial Regression

**Purpose:** To capture nonlinear relationships between variables.

* **Pros:** Better at fitting nonlinear data compared to linear regression.
* **Cons:** Prone to overfitting with high-degree polynomials.
* **Modeling Equation:** 

$$y = \beta_0 + \beta_1x + \beta_2x^2 + ...$$

```python
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

# Transform features to polynomial features
poly = PolynomialFeatures(degree=2)
X_poly = poly.fit_transform(X)

# Fit linear regression on transformed features
model = LinearRegression()
model.fit(X_poly, y)
```

### Multiple Linear Regression

**Purpose:** To predict a dependent variable based on multiple independent variables.

* **Pros:** Accounts for multiple factors influencing the outcome.
* **Cons:** Assumes a linear relationship between predictors and target; sensitive to multicollinearity.
* **Modeling Equation:** 

$$y = \beta_0 + \beta_1x_1 + \beta_2x_2 + ...$$

```python
from sklearn.linear_model import LinearRegression

model = LinearRegression()
model.fit(X, y)
```

### Logistic Regression

**Purpose:** To predict probabilities of categorical outcomes (Binary Classification).

* **Pros:** Efficient for binary classification problems; outputs probabilities.
* **Cons:** Assumes a linear relationship between independent variables and the log-odds.
* **Modeling Equation:** 

$$\log\left(\frac{p}{1-p}\right) = \beta_0 + \beta_1x_1 + ...$$

```python
from sklearn.linear_model import LogisticRegression

model = LogisticRegression()
model.fit(X, y)
```

## 2. Associated Functions (Scikit-Learn)

### Data Preparation

| Function | Description | Code Syntax |
|----------|-------------|-------------|
| `train_test_split` | Splits the dataset into training and testing subsets to evaluate the model's performance. | `from sklearn.model_selection import train_test_split`<br><br>`X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)` |
| `StandardScaler` | Standardizes features by removing the mean and scaling to unit variance. | `from sklearn.preprocessing import StandardScaler`<br><br>`scaler = StandardScaler()`<br><br>`X_scaled = scaler.fit_transform(X)` |

### Performance Metrics

| Metric | Description | Code Syntax |
|--------|-------------|-------------|
| `mean_absolute_error` | Calculates the mean absolute error (MAE) between actual and predicted values. | `from sklearn.metrics import mean_absolute_error`<br><br>`mae = mean_absolute_error(y_true, y_pred)` |
| `mean_squared_error` | Computes the mean squared error (MSE) between actual and predicted values. | `from sklearn.metrics import mean_squared_error`<br><br>`mse = mean_squared_error(y_true, y_pred)` |
| `root_mean_squared_error` | Calculates the RMSE. Interprets error in the same units as the target variable. | `import numpy as np`<br><br>`rmse = np.sqrt(mean_squared_error(y_true, y_pred))` |
| `r2_score` | Computes the R-squared value, indicating how well the model explains the variability of the target. | `from sklearn.metrics import r2_score`<br><br>`r2 = r2_score(y_true, y_pred)` |
| `log_loss` | Calculates the logarithmic loss, a performance metric for classification models. | `from sklearn.metrics import log_loss`<br><br>`loss = log_loss(y_true, y_pred_proba)` |