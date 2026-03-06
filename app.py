import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score


# ---------------------------------------------------
# PAGE CONFIG
# ---------------------------------------------------

st.set_page_config(
    page_title="House Price Prediction",
    page_icon="🏠",
    layout="wide"
)

st.title("🏠 House Price Prediction using Machine Learning")

st.markdown(
"""
This application predicts **house prices** based on several property features.

The model uses **Support Vector Regression (SVR)** with **hyperparameter tuning** to improve prediction accuracy.

Simply enter the property details and the model will estimate the **house price**.
"""
)

st.divider()

# ---------------------------------------------------
# LOAD DATASET
# ---------------------------------------------------

df = pd.read_csv("House_Data_Multi.csv")

X = df.iloc[:, :-1].values
y = df.iloc[:, -1].values
y = y.reshape(-1,1)

# ---------------------------------------------------
# TRAIN TEST SPLIT
# ---------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ---------------------------------------------------
# FEATURE SCALING
# ---------------------------------------------------

sc_X = StandardScaler()
sc_y = StandardScaler()

X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)

y_train = sc_y.fit_transform(y_train)
y_test = sc_y.transform(y_test)

# ---------------------------------------------------
# HYPERPARAMETER TUNING
# ---------------------------------------------------

param_grid = {
    "C": [0.1,1,10,100],
    "gamma": [1,0.1,0.01,0.001],
    "epsilon": [0.1,0.2,0.5]
}

grid = GridSearchCV(
    SVR(kernel="rbf"),
    param_grid,
    scoring="neg_mean_squared_error",
    cv=5
)

grid.fit(X_train, y_train.ravel())

model = grid.best_estimator_

# ---------------------------------------------------
# MODEL EVALUATION
# ---------------------------------------------------

y_pred_scaled = model.predict(X_test).reshape(-1,1)

y_pred = sc_y.inverse_transform(y_pred_scaled)
y_test_actual = sc_y.inverse_transform(y_test)

mse = mean_squared_error(y_test_actual, y_pred)
r2 = r2_score(y_test_actual, y_pred)

# ---------------------------------------------------
# SIDEBAR INPUT
# ---------------------------------------------------

st.sidebar.header("🏡 Enter Property Details")

area = st.sidebar.slider("Area (sq ft)", 800, 2500, 1500)
bedrooms = st.sidebar.slider("Bedrooms", 1, 6, 3)
bathrooms = st.sidebar.slider("Bathrooms", 1, 5, 2)
age = st.sidebar.slider("Age of House (years)", 0, 30, 10)

# ---------------------------------------------------
# PREDICTION
# ---------------------------------------------------

input_data = [[area, bedrooms, bathrooms, age]]

scaled_input = sc_X.transform(input_data)

prediction_scaled = model.predict(scaled_input).reshape(-1,1)

prediction = sc_y.inverse_transform(prediction_scaled)

# ---------------------------------------------------
# DISPLAY RESULT
# ---------------------------------------------------

st.header("💰 Predicted House Price")

st.metric(
    label="Estimated Price",
    value=f"${prediction[0][0]:,.0f}"
)

st.divider()

# ---------------------------------------------------
# MODEL PERFORMANCE
# ---------------------------------------------------

st.header("📊 Model Performance")

col1, col2 = st.columns(2)

with col1:
    st.metric("R² Score", f"{r2:.3f}")

with col2:
    st.metric("Mean Squared Error", f"{mse:,.0f}")

st.info(
"""
**R² Score** measures how well the model explains house prices.

• **Closer to 1 → Better model**  
• **Closer to 0 → Weak model**
"""
)

st.divider()

# ---------------------------------------------------
# BEST HYPERPARAMETERS
# ---------------------------------------------------

st.header("⚙️ Best Model Parameters")

st.write(grid.best_params_)

st.divider()


st.header("🧠 How the Machine Learning Model Works")

st.markdown("""
This application uses a machine learning algorithm called **Support Vector Regression (SVR)**.

SVR learns the relationship between **house features** and **house prices** from historical data.

To control how the model learns, it uses three important parameters:
""")

st.subheader("⚙️ C (Model Strictness)")

st.markdown("""
**C controls how much the model tries to avoid errors.**

• Small **C** → Model allows more mistakes but stays simple  
• Large **C** → Model tries to fit the data very closely  

Think of **C** as how strict the model is when learning from data.
""")

st.subheader("📍 Gamma (Influence of Data Points)")

st.markdown("""
**Gamma controls how much influence each training example has.**

• Small **gamma** → Model looks at broader patterns in the data  
• Large **gamma** → Model focuses heavily on nearby data points  

This affects how smooth or complex the prediction curve becomes.
""")

st.subheader("📏 Epsilon (Error Tolerance)")

st.markdown("""
**Epsilon defines a margin where small prediction errors are ignored.**

Example:

If the real price is **$300,000** and the model predicts **$302,000**,  
the model may consider this close enough and ignore the small difference.

Larger **epsilon** → model ignores more small errors.
""")

st.info("""
These parameters are automatically optimized using **GridSearchCV**, which tests many combinations and selects the best performing model.
""")

# ---------------------------------------------------
# DATASET OVERVIEW
# ---------------------------------------------------

st.header("📂 Dataset Preview")

st.dataframe(df)

st.write("### Dataset Statistics")

st.write(df.describe())

st.divider()

# ---------------------------------------------------
# CHARTS
# ---------------------------------------------------

st.header("📈 Data Visualization")

col1, col2 = st.columns(2)

# Scatter plot
with col1:
    fig1, ax1 = plt.subplots()
    ax1.scatter(df["Area"], df["Price"])
    ax1.set_xlabel("Area")
    ax1.set_ylabel("Price")
    ax1.set_title("Area vs Price")
    st.pyplot(fig1)

# Price distribution
with col2:
    fig2, ax2 = plt.subplots()
    ax2.hist(df["Price"], bins=10)
    ax2.set_title("Price Distribution")
    st.pyplot(fig2)

st.divider()

# ---------------------------------------------------
# ACTUAL VS PREDICTED
# ---------------------------------------------------

st.header("📉 Prediction Accuracy")

fig3, ax3 = plt.subplots()

ax3.scatter(y_test_actual, y_pred)

ax3.set_xlabel("Actual Price")
ax3.set_ylabel("Predicted Price")

ax3.set_title("Actual vs Predicted Prices")

st.pyplot(fig3)

st.divider()

# ---------------------------------------------------
# ABOUT SECTION
# ---------------------------------------------------

st.header("ℹ️ About This Project")

st.markdown(
"""
### Machine Learning Workflow

1️⃣ Load housing dataset  
2️⃣ Split data into training and testing sets  
3️⃣ Apply feature scaling  
4️⃣ Tune SVR model using GridSearchCV  
5️⃣ Train optimized model  
6️⃣ Predict house prices  
7️⃣ Evaluate model performance  

### Technologies Used

• Python  
• Streamlit  
• Scikit-learn  
• Pandas  
• NumPy  
• Matplotlib

This project demonstrates a **complete machine learning pipeline** including **model tuning and evaluation**.
"""
)

st.caption("Created by Chinmay V Chatradamath 🚀")
