# CodeAlpha Task 1 - Iris Flower Classification

## Objective
Build a machine-learning classifier that predicts whether an Iris flower is Setosa, Versicolor, or Virginica using four measurements: sepal length, sepal width, petal length, and petal width.

## Tools
- Python
- Pandas
- Scikit-learn
- Matplotlib
- Joblib

## Dataset
This project uses the built-in Iris dataset provided by Scikit-learn (150 rows, 4 numeric features, 3 species). A separate dataset download is not required.
The CodeAlpha PDF also links to: https://www.kaggle.com/datasets/saurabh00007/iriscsv

## Run on your computer
1. Install Python 3.10 or newer.
2. Open this project folder in VS Code.
3. Open Terminal in VS Code.
4. Create a virtual environment (optional but recommended):
   `python -m venv .venv`
5. Activate it:
   - Windows: `.venv\Scripts\activate`
   - macOS/Linux: `source .venv/bin/activate`
6. Install packages: `pip install -r requirements.txt`
7. Run: `python main.py`

## Method
1. Load the Iris dataset.
2. Inspect the rows and check missing values.
3. Split the data into 80% training and 20% testing.
4. Standardize features and train a K-Nearest Neighbors model.
5. Evaluate predictions using accuracy, classification report, and confusion matrix.
6. Save the model and evaluation output in `outputs/`.

## Important
Run the program yourself before submitting. The accuracy is produced by the program; do not claim a result until you have run it. Include your own screenshots in the repository if your internship coordinator asks for them.
