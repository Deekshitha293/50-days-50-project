Day 3 — Build Your First Machine Learning Model (Iris Classification)

📅 Part of: AI 50-Day Challenge
👩‍💻 Author: Deekshitha Bhairav

🎯 Objective

Train my first supervised machine learning model using the Iris dataset.

Understand the complete ML pipeline: data preparation → training → prediction → evaluation → model saving.

Gain practical experience with Scikit-learn for classification tasks.

🧰 Key Libraries & Purpose
Library	Use Case
Pandas	Data loading & DataFrame processing
NumPy	Numerical operations
Scikit-learn	Train-test split, Logistic Regression, accuracy scoring
Matplotlib	Basic plots for model understanding
Joblib	Saving trained ML models
📌 Quick Reference — ML Steps
Step	Description
Data Loading	Load CSV and assign feature names
Feature Selection	Select X (features) and y (labels)
Train-Test Split	Divide data for training & testing
Model Training	Fit Logistic Regression model
Prediction	Predict species on test data
Evaluation	Measure accuracy & performance
Saving Model	Export .pkl for reuse
🔧 Model Used — Logistic Regression

Suitable for multi-class classification

Fast and simple

Works well with linearly separable data

Performs excellently on Iris dataset

🔍 Key Observations from Model
✔ Model Performance

Achieved high accuracy (95%–100%)

Predictions were stable across multiple runs

Petal measurements were the strongest features for classification

✔ Class Separation

Setosa is perfectly separable

Versicolor and Virginica overlap slightly but still classified well

PCA visualizations revealed clusters clearly

📊 Visual Insights
Plots Used
Plot Type	Purpose	Insight
Scatter Plot	Quick feature comparison	Species grouped distinctly
PCA Plot	Reduce 4D → 2D	Show clear cluster separation
Confusion Matrix	Evaluate predictions	Model performance per class
🧾 Important ML Concepts Learned
1️⃣ Train-Test Split

Separates data into:

Training set → model learns patterns

Test set → checks real-world accuracy

2️⃣ Logistic Regression Basics

Uses softmax for multi-class prediction

Decision boundaries depend on feature weights

Ideal for small, clean datasets like Iris

3️⃣ Accuracy Score

Shows how many predictions were correct:

accuracy_score(y_test, y_pred)

4️⃣ Model Saving

Used Joblib to save trained model locally:

joblib.dump(model, "iris_model.pkl")

🧪 Summary of Code (Simplified)
# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = LogisticRegression(max_iter=200)
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Accuracy
accuracy_score(y_test, y_pred)

🏁 Takeaways

Built my first machine learning model successfully

Learned the full ML workflow step-by-step

Understood the importance of splitting data

Identified which features influence predictions

Learned how to save and reuse a trained model

🌷 “The first model is not just code — it’s the beginning of intelligent decision-making.”
— Deekshitha Bhairav