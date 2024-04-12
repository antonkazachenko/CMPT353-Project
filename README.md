# Online Payments Fraud Detection Analysis

##

# Overview

The project aimed to develop a fraud detection system through a systematic approach which included data exploration, preprocessing, and model selection. Initially, thorough exploratory analysis was conducted, including the removal of duplicates and outliers, and the creation of new meaningful features. Following this, the dataset was split into train and test sets, ensuring a balanced representation of fraudulent transactions. Several classification models were then implemented and evaluated based on metrics such as F1 score, with Random Forest being the most effective in detecting fraud. Additionally, boosting algorithms were considered for potential performance enhancement. Overall, the project successfully addressed the challenge of fraud detection by employing a comprehensive methodology spanning from data exploration to model selection and evaluation.

---

# Required Libraries & Commands

This project relies on several external Python libraries. To ensure smooth operation and compatibility, please ensure that you have the following libraries installed. This project is tested with Python 3.8 and above.

### Libraries

- **NumPy:** For numerical computing and array operations.
- **Pandas:** For data manipulation and analysis.
- **Matplotlib:** For creating static, interactive, and animated visualizations.
- **Seaborn:** For making statistical graphics in Python.
- **Scikit-learn:** For machine learning and data processing.

### Installation

You can install all the required libraries at once using the following command:

```markdown
pip install numpy pandas matplotlib seaborn scikit-learn
```

### Optional: Creating a Virtual Environment

Before installing these libraries, it is recommended to create a virtual environment. This keeps dependencies required by different projects separate by creating isolated environments for them. You can create a virtual environment and activate it using:

Create a virtual environment (replace 'myenv' with your desired environment name):

```markdown
python -m venv myenv
```

Activate the virtual environment:

- On Windows:

```markdown
myenv\Scripts\activate
```

- On Unix or MacOS:

```markdown
source myenv/bin/activate
```

Now you can install the libraries within this environment:

```markdown
pip install numpy pandas matplotlib seaborn scikit-learn
```

Deactivate the virtual environment when you're done:

```markdown
deactivate
```

By following these instructions, you'll set up a Python environment with all the necessary libraries to run the project successfully.

---

# How to Run

### Preparation

1. Environment Setup: Ensure that Python and necessary libraries listed in the "Required Libraries & Commands" section of this README are installed.
2. Data Preparation: Before running any scripts or notebooks, make sure that the dataset clean_data.csv is available in the root directory where the scripts will be executed.

### Order of Execution

1. Prepare Data: Execute prepare_data.py to preprocess the data, ensuring it is formatted correctly for analysis.
2. Feature Engineering: Run the features-2-1.ipynb Jupyter notebook to create and select features necessary for model building.
3. Data Splitting: Use split-2-2.ipynb to split the dataset into training and testing sets.
4. Statistical Analysis: Open and run stats-1-2.ipynb for initial statistical analysis and visualization.
5. Dimensionality Reduction: Execute the pca-1-3.ipynb notebook to perform principal component analysis for feature reduction.
6. Model Building:
   - Run models.py to construct base machine learning models.
   - Execute Random_Forest params.py to fine-tune Random Forest parameters.

---

# Files produced/expected

- Preprocessed Data: prepare_data.py outputs a processed version of clean_data.csv, typically named processed_data.csv.
- Feature Set: features-2-1.ipynb creates feature sets saved as features.csv.
- Split Data: split-2-2.ipynb produces train_set.csv and test_set.csv.
- PCA Components: pca-1-3.ipynb generates files containing the principal components, usually named pca_features.csv.
- Model Files:
  - models.py may save model files like base_model.pkl for each algorithm implemented.
  - Random_Forest params.py might save a tuned Random Forest model as rf_tuned_model.pkl.
