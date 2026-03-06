# 🏠 House Price Prediction using Support Vector Regression (SVR)

![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-Web%20App-FF4B4B?logo=streamlit\&logoColor=white)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-Machine%20Learning-orange?logo=scikit-learn)
![License](https://img.shields.io/badge/License-MIT-green)

[![Live Demo](https://img.shields.io/badge/Live-Demo-brightgreen?logo=streamlit)](https://svrwithhyperparametersbychinmay.streamlit.app/)

---

# 📊 Project Overview

This project predicts **house prices** based on property characteristics using a **Support Vector Regression (SVR)** machine learning model.

The application is built as an **interactive web app using Streamlit**, allowing users to enter property details and instantly receive a predicted house price.

The project demonstrates a **complete machine learning workflow**, including:

* Data preprocessing
* Feature scaling
* Train/Test split
* Hyperparameter tuning using **GridSearchCV**
* Model evaluation
* Interactive visualization

---

# 🚀 Features

* Interactive **Streamlit web application**
* Predict house prices using property features
* Hyperparameter tuning with **GridSearchCV**
* Model evaluation metrics
* Multiple data visualizations
* Beginner-friendly explanations of ML concepts
* Clean and intuitive user interface

---

# 🧠 Machine Learning Model

The project uses **Support Vector Regression (SVR)** with an **RBF kernel**.

Three key hyperparameters are optimized:

### **C (Regularization Parameter)**

Controls how strictly the model tries to avoid prediction errors.

* Small C → Simpler model
* Large C → Model fits training data more closely

---

### **Gamma**

Controls how much influence each training point has.

* Small gamma → smoother predictions
* Large gamma → more complex decision boundaries

---

### **Epsilon**

Defines a margin where small prediction errors are ignored.

This helps prevent the model from overreacting to tiny fluctuations in data.

---

# ⚙️ Hyperparameter Tuning

The model uses **GridSearchCV** to automatically test many parameter combinations.

Example search space:

* C → `[0.1, 1, 10, 100]`
* gamma → `[1, 0.1, 0.01, 0.001]`
* epsilon → `[0.1, 0.2, 0.5]`

GridSearch trains multiple models and selects the **best performing configuration**.

---

# 📈 Model Evaluation

The model performance is evaluated using:

### **R² Score**

Measures how well the model explains the variance in house prices.

```
1.0 → perfect predictions
0.0 → model predicts average
```

### **Mean Squared Error (MSE)**

Measures the average squared difference between predicted and actual prices.

Lower values indicate better predictions.

---

# 📊 Visualizations

The app includes several charts to help users understand the dataset and model performance:

* Area vs Price scatter plot
* Price distribution histogram
* Actual vs Predicted price comparison
* Dataset statistics overview

---

# 🖥️ Streamlit User Interface

Users can interactively enter:

* House Area
* Number of Bedrooms
* Number of Bathrooms
* Age of the House

The application then predicts the **estimated house price instantly**.

---

# 📂 Project Structure

```
house-price-svr
│
├── app.py
├── House_Data_Multi.csv
├── requirements.txt
└── README.md
```

---

# ▶️ Run the Project Locally

### Clone the repository

```
git clone https://github.com/yourusername/house-price-svr.git
cd house-price-svr
```

### Install dependencies

```
pip install -r requirements.txt
```

### Run the Streamlit app

```
streamlit run app.py
```

---

# 🛠 Technologies Used

* Python
* Streamlit
* Scikit-learn
* NumPy
* Pandas
* Matplotlib

---

# 👨‍💻 Author

**Chinmay V Chatradamath**

---

⭐ If you found this project useful, consider giving the repository a **star**.
