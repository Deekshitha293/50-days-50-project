# 🌟 Day 2 — Data Exploration & Visualization

### 📅 Part of: AI 50-Day Challenge  
### 👩‍💻 Author: Deekshitha Bhairav  

---

## 🎯 Goal of the Day

The goal of **Day 2** was to understand how to **explore, clean, and visualize data** — a crucial step before building any AI or Machine Learning model.  
By the end of this task, we aimed to:
- Verify that Python libraries are working correctly  
- Load and explore a dataset  
- Visualize relationships between features  
- Write insights based on the data  

---

## 🧠 Concepts Covered

| Concept | Description |
|----------|--------------|
| **Data Exploration** | The process of understanding what the data looks like — its shape, columns, and values. |
| **Data Cleaning** | Checking for and handling missing values, incorrect data types, or duplicate rows. |
| **Data Visualization** | Creating plots and charts to easily identify trends and relationships. |

---

## 🧰 Python Libraries Used

| Library | Purpose |
|----------|----------|
| **NumPy (`np`)** | For numerical and array-based operations |
| **Pandas (`pd`)** | For reading, cleaning, and handling tabular data |
| **Matplotlib (`plt`)** | For creating charts and graphs |
| **Seaborn (`sns`)** | For more stylish and advanced visualizations |
| **Scikit-learn (`sklearn.datasets`)** | For loading built-in datasets like Iris |

**Code Snippet:**
```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris
sns.set(style="whitegrid")
