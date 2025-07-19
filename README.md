# 🧠 Salary Prediction Using Linear Regression

This project builds and evaluates a simple machine learning model to predict salaries based on years of experience. It's an end-to-end implementation including data preprocessing, visualization, training, evaluation, and model persistence.

---

## 📌 Table of Contents

- [📖 Project Overview](#-project-overview)
- [📂 Folder Structure](#-folder-structure)
- [📊 Dataset Description](#-dataset-description)
- [🧪 Tech Stack / Libraries Used](#-tech-stack--libraries-used)
- [🚀 Installation & Setup](#-installation--setup)
- [⚙️ How to Run](#️-how-to-run)
- [📈 Model Training & Evaluation](#-model-training--evaluation)
- [💾 Saving & Loading the Model](#-saving--loading-the-model)
- [📉 Sample Outputs](#-sample-outputs)
- [📌 Future Improvements](#-future-improvements)
- [🙋‍♂️ Author](#-author)


## 📖 Project Overview

The goal is to build a regression model that learns the relationship between `Years of Experience` and `Salary`. This is a common supervised learning problem and is solved here using **Linear Regression**.

The project includes:

- Loading and exploring the dataset
- Visualizing the data trends
- Training a regression model
- Evaluating the model performance
- Saving the trained model using `pickle`
- Deployable codebase structure

## 📂 Folder Structure

salary_prediction/
│
├── data/
│ └── salary_data.csv
│
├── notebooks/
│ └── salary_prediction.ipynb # Jupyter notebook with full workflow
│
├── model/
│ ├── model.pkl # Saved trained model
│ └── model.py # Script version of training code
│
├── requirements.txt
├── .gitignore
└── README.md


---

## 📊 Dataset Description

The dataset contains two columns:

| Feature | Description            |
|---------|------------------------|
| `YearsExperience` | Number of years of professional experience |
| `Salary` | Corresponding annual salary in USD |

> Source: Often from datasets like `salary_data.csv` from Kaggle or GFG ML projects.

---

## 🧪 Tech Stack / Libraries Used

- `Python 3.x`
- `pandas` – data manipulation
- `numpy` – numerical ops
- `matplotlib`, `seaborn` – plotting and visualization
- `scikit-learn` – model training and evaluation
- `pickle` – model serialization

---

## 🚀 Installation & Setup

Clone the repository and install the dependencies:

```bash
git clone https://github.com/Prateeks1315/salary_prediction.git
cd salary_prediction
pip install -r requirements.txt

📈 Model Training & Evaluation
The model uses a basic Linear Regression algorithm from scikit-learn. The pipeline includes:
Data loading with pandas
Data visualization using scatter plots
Model training using LinearRegression()
Performance evaluation using:
R² Score
Mean Absolute Error
Mean Squared Error
📉 Sample Outputs
📊 Plot: Experience vs Salary (Fitted Line)

📝 Predicted Results
yaml
Copy
Edit
Years of Experience: 5.3
Predicted Salary: $75,460.00
📌 Future Improvements
Add more features: Education level, city, industry, etc.
Try more complex models (Polynomial Regression, Decision Trees)
Hyperparameter tuning with GridSearchCV
Deploy with Streamlit or Flask
Add unit tests and CI/CD setup
🙋‍♂️ Author
Made with ❤️ by Prateeks1315
Feel free to ⭐ this repo or reach out for collaboration!

