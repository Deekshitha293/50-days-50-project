# 🌟 Day 4 — Model Improvement & Hyperparameter Tuning (Iris Dataset)

📅 **Part of:** AI 50-Day Challenge  
👩‍💻 **Author:** Deekshitha Bhairav  

---

## 🎯 Objective
- Compare multiple ML algorithms on the Iris dataset.  
- Improve model performance using hyperparameter tuning (GridSearchCV).  
- Select and save the best model for future use.  
- Gain hands-on experience with Python libraries: **Scikit-learn, Pandas, Matplotlib, Joblib**.

---

## 🧰 Key Libraries & Purpose

| Library        | Use Case                                       |
|----------------|-----------------------------------------------|
| Pandas         | Data handling and preprocessing               |
| Scikit-learn   | ML algorithms, train-test split, metrics, GridSearchCV |
| Matplotlib     | Visualize model comparison and results       |
| Joblib         | Save and load trained models                  |

---

## 📌 Quick Reference — Model Training & Evaluation

| Step / Function      | Meaning / Use Case                              |
|---------------------|-------------------------------------------------|
| `train_test_split()` | Split dataset into training and testing sets   |
| `.fit()`             | Train model on training data                   |
| `.predict()`         | Make predictions on test data                  |
| `accuracy_score()`   | Measure model accuracy                          |
| `GridSearchCV()`     | Tune model hyperparameters for best performance|
| `best_params_`       | Best parameters found by GridSearchCV          |
| `best_score_`        | Best cross-validation score                     |
| `joblib.dump()`      | Save trained model                              |
| `joblib.load()`      | Load saved model                                |

---

## 📌 Models Compared

| Model                  | Purpose                                       |
|------------------------|-----------------------------------------------|
| Logistic Regression     | Basic linear classifier                       |
| K-Nearest Neighbors (KNN) | Classifies based on nearest neighbors       |
| Decision Tree           | Flowchart-based decision making               |
| Random Forest           | Ensemble of decision trees, reduces overfitting |
| SVM                     | Finds the best separating boundary            |

---

## 💡 Key Steps for Day 4

1. Load Iris dataset and separate features (`X`) and target (`y`)  
2. Train-test split: 80% training, 20% testing  
3. Train multiple ML models and record accuracy  
4. Visualize model performance with bar chart  
5. Select the best-performing model (**Random Forest**)  
6. Hyperparameter tuning using `GridSearchCV`:
    - `n_estimators`: [50, 100, 200]  
    - `max_depth`: [None, 5, 10]  
    - `min_samples_split`: [2, 5]  
7. Train final model with best parameters  
8. Evaluate final model accuracy on test data  
9. Save model for future use (`best_model.pkl`)  

---

## 🧠 Concepts to Remember
- **Model Comparison:** Always compare multiple algorithms to pick the best.  
- **Hyperparameter Tuning:** Improves accuracy and generalization by finding optimal parameters.  
- **Cross-Validation:** Measures model performance on multiple splits to avoid overfitting.  
- **Model Saving:** Allows reuse of trained models without retraining.

---

## 💡 Key Insights from Day 4
- All models gave similar accuracy initially on the Iris dataset.  
- Random Forest with tuned parameters achieved **1.0 accuracy**.  
- Hyperparameter tuning is essential for real-world ML projects.  
- Visualizing model comparison helps in quick selection of the best model.

---

## 🏁 Takeaways
- Always compare multiple algorithms before finalizing a model.  
- Tuning hyperparameters significantly improves model performance.  
- Saving the trained model ensures it can be deployed or reused.  
- Understanding model selection and tuning is key for interviews and real-world ML projects.

---

## 🌷 Quote
> **“Improving models wisely is the bridge between theory and real-world AI success.” — Deekshitha Bhairav**
