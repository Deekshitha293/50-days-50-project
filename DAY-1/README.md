# 🌟 Day 1 — Data Exploration & Visualization (Interview Notes)

### 📅 Part of: AI 50-Day Challenge  
👩‍💻 **Author:** Deekshitha Bhairav  

---

## 🎯 Objective
- Understand dataset structure, clean data, and visualize key patterns before modeling.

---

## 🧰 Key Libraries & Purpose
| Library       | Use Case                                  |
| ------------- | ---------------------------------------- |
| **NumPy**     | Numerical computations                   |
| **Pandas**    | Data handling, cleaning, analysis        |
| **Matplotlib**| Basic plotting                           |
| **Seaborn**   | Advanced visualization (pairplot, heatmap) |
| **Scikit-learn** | Load datasets, preprocessing           |

---

## 📌 Quick Reference — Data Exploration

| Function / Method        | Meaning / Use Case                            |
| ------------------------ | -------------------------------------------- |
| `data.head()`             | View first 5 rows                             |
| `data.shape`              | Rows × columns                               |
| `data.dtypes`             | Data types of columns                        |
| `data.describe()`         | Summary statistics (mean, std, min, max)    |
| `data.isnull().sum()`     | Count missing values                         |
| `data.corr()`             | Correlation between features                 |

---

## 📌 Key Visualization Tools

| Plot Type       | Purpose                                   | Example Feature      |
| --------------- | ----------------------------------------- | ------------------ |
| **Histogram**   | Distribution of numerical values         | Feature-wise       |
| **Pairplot**    | Relationship & species separation        | Scatter + hue      |
| **Heatmap**     | Correlation between features             | Color-coded values |

---

## 💡 Key Insights from Iris Dataset
- **Rows × Columns:** 150 × 5  
- **No missing values**  
- **Strong correlation:** Petal length ↔ Petal width  
- Petal features better separate species than sepal features  

---

## 🧾 Quick Concepts for Interviews
- **Data Exploration:** Understand dataset before modeling  
- **Summary Statistics:** `.describe()` — mean, std, min, max  
- **Correlation:** Measures strength & direction between features  
- **Visualization:** Identify patterns, outliers, and relationships  

---

## 🏁 Takeaways
- Data should be **cleaned and explored** before ML modeling  
- Visual insights guide **feature selection**  
- Key functions, plots, and correlations are **frequently asked in interviews**  

---

> 🌷 *“Exploring data wisely is the first step toward intelligent decisions.”* — Deekshitha Bhairav
