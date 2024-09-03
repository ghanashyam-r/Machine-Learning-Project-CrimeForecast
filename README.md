# Machine-Learning-Project-CrimeForecast
Machine Learning Project Done on Kaggle as a part of gaining required credits for achieving Diploma in Data Science IITM.I secured an 'S' grade for the project.The competition was based on a Classification Problem with the LA crime dataset and i managed to achieve an Accuracy of 95.6 percent with my XGBoost Model.

The project procedure involved crossing a cutoff of 80 percent and clearing two vivas, with the Level 2 viva being conducted by an industry professional.
*Code available in the MLP final notebook file*

# Detailed Description of Code
## Imports
I first imported all the relevant libraries:
**NumPy and Pandas**: These are essential libraries for data manipulation and analysis. Pandas is specifically used for reading and handling CSV files.
**Sklearn Libraries**: These are essential for machine learning tasks such as imputing missing values, preprocessing data, model training, and evaluation.
**Matplotlib and Seaborn**: These libraries are used for data visualization, allowing you to understand your data better and communicate findings.

## EDA
The training, testing, and sample datasets are loaded using `pd.read_csv()`.The data is imported and stored into corresponding variables 
After that i wrote code for exploring the basic features of the data such as ‘shape’ to display number of rows and columns in the test and train datasets.'info()' provides detailed information about the dataset, including column names, data types, and non-null counts, which helps in understanding the data structure and identifying any issues such as missing values.'describe()` provides a statistical summary of the data, including count, mean, standard deviation, and other metrics
*From using the describe() function it is noticed that there are some 'zero' values in the latitude and longitude which were found to be outliers.Also there were negative values in the age column.*'isnull().sum()' is used to count the number of missing values in each column. This is an essential step in data cleaning as missing data can affect the performance of machine learning models.

The code first addresses zero values in the `Latitude` and `Longitude` columns of both the training and testing datasets which are outliers.
The scatter plot represents the geographical distribution of different categories of crimes based on their latitude and longitude. Each dot corresponds to a specific crime incident, and the color of the dot indicates the type of crime.

*From this plot, I can observe that crimes are spread across the area, but certain categories seem more concentrated in specific regions. For example, property crimes (in blue) appear widely distributed, while other categories like fraud and white-collar crimes (in purple) might be more localized.*
This visualization helps in identifying **patterns in the geographical occurrence of crimes.** It also highlights the need to explore clustering techniques to group similar locations together, which could simplify the analysis by revealing crime hotspots or areas with distinct crime patterns. Clustering would allow me to better understand and predict where different types of crimes are more likely to occur.

The correlation matrix is calculated using the `.corr()` function, which computes the pairwise correlation of all numeric columns.Area_ID and Reporting District number is same

Different age groups are victim to different kind of crimes so age binning column would help increase the the accuracy.

Identifying different placeholders in victim_sex.

## PREPROCESSING

### 1. **Handling Geographical Data: KMeans Clustering**

- **EDA Step: Plotting Latitude and Longitude**:
    - **Why**: Visualizing the geographical distribution of your data can help you understand if there are natural groupings or clusters based on location.
    - I applied KMeans clustering to the latitude and longitude data to create a new feature, 'Cluster'. This can help the model distinguish between different geographic regions, which might be important for predicting the target variable.
    - a new 'Cluster' feature that groups similar locations. The clustering model is trained on the training data's coordinates and then used to predict clusters for both the training and test datasets. By adding these clusters as a feature, we provide the model with additional spatial information, which can improve its ability to capture patterns related to the target variable.

### 2. **Handling Missing Data in 'Modus_Operandi'**

- **EDA Step: Missing Value Analysis**:
    - **Why**: Analyze the distribution and frequency of missing values in the 'Modus_Operandi' column.
    - **Preprocessing Step**: You replaced NaN values with an empty string, which is often suitable for text data. This allows the model to process these records without losing information.

### 3. **Conversion of 'Weapon_Used_Code' to String**

- **EDA Step: Data Type Inspection**:
    - **Why**: Check the data types of each column to ensure they are appropriate for analysis and modeling.
    - **Preprocessing Step**: The 'Weapon_Used_Code' column might contain categorical data that was being treated as a numeric type. Converting it to a string ensures it’s processed correctly as a categorical variable.

### 4. **Replacing 'X' in 'Victim_Sex' with NaN**

- **EDA Step: Unique Value Counts**:
    - **Why**: Examine the unique values in the 'Victim_Sex' column to identify any anomalies or placeholders like 'X'.
    - **Preprocessing Step**: 'X' is likely a placeholder for missing or unknown data, so replacing it with NaN ensures consistency in handling missing data.

### 5. **Date Handling: Converting and Extracting Components**

- **EDA Step: Distribution of Dates**:
    - **Why**: Analyze the distribution of dates to understand trends over time and identify any incorrect formats or anomalies.
    - **Preprocessing Step**: Converting dates to a consistent format and extracting components (year, month, day) allows the model to utilize time-based features, which can be crucial for time-sensitive data.

### 6. **Handling Negative Ages**

- **EDA Step: Summary Statistics and Outlier Detection**:
    - **Why**: Calculate summary statistics like mean, median, and identify any negative or outlier values in 'Victim_Age'.
    - **Preprocessing Step**: Replacing negative ages with the median ensures data integrity, as negative ages are not realistic and could distort the analysis.

### 7. **Creating Age Bins**

- **EDA Step: Age Distribution Analysis**:
    - **Why**: Analyze the distribution of 'Victim_Age' to decide on appropriate age ranges for binning.
    - **Preprocessing Step**: Binning the ages into ranges simplifies the model's task by grouping ages into categories, which can be particularly useful for models that benefit from categorical rather than continuous variables.
    - 
    
    [Age binning detailed ](https://www.notion.so/Age-binning-detailed-a2868856e53d4ad09e0a31d17163e1ca?pvs=21)
    

### 8. **Creating 'Report_Occur_Diff' Feature**

- **EDA Step: Time Difference Analysis**:
    - **Why**: Calculate the time difference between 'Date_Reported' and 'Date_Occurred' to understand reporting delays.
    - **Preprocessing Step**: Creating a 'Report_Occur_Diff' feature provides valuable information about the time gap, which could be predictive in understanding the nature of the incidents.

### 9. **Dropping Unnecessary Columns**

- **EDA Step: Feature Importance and Correlation Analysis**:
    - **Why**: Evaluate the importance and correlation of features to the target variable to identify and drop unimportant columns.
    - **Preprocessing Step**: Dropping columns like 'Date_Reported', 'Date_Occurred', and 'Area_ID' likely reflects that these features either provided redundant information or were not useful for the modeling process.

### 10. **Handling Categorical and Numerical Features**

- **EDA Step: Categorical and Numerical Feature Analysis**:
    - **Why**: Identify categorical and numerical features and analyze their distribution, missing values, and potential impact on the model.
    - **Preprocessing Step**: You created pipelines for both types of data:
    - **Numeric Transformer**: Imputation and scaling ensure that numeric features are on a comparable scale and have no missing values.
    - **Categorical Transformer**: Imputation and encoding convert categorical variables into a format suitable for machine learning models.
    - **Text Transformer**: Vectorizing the 'Modus_Operandi' column allows the model to process text data as numerical features.

Mean Imputer

Numeric features are standardized using `StandardScaler`, which scales the data to have a mean of 0 and a standard deviation of 1.

**Encoding**: Categorical variables are converted to one-hot encoded vectors using `OneHotEncoder`. The `handle_unknown='ignore'` parameter ensures that any unknown categories in the test data are ignored.
**Vectorization**: Text data is converted into numerical features using `CountVectorizer`, which creates a matrix of token counts.

The `ColumnTransformer` combines these three pipelines into one.

Baseline Model

**`DummyClassifier`**

: A simple classifier that makes predictions using basic rules.

- **`strategy="most_frequent"`**: The classifier will always predict the most frequent class in the training data.

The `fit` method trains the classifier on the training data (`X_train`, `Y_train`). In this case, it simply determines the most frequent class in `Y_train`.
**`dummy_clf.predict(X)`**: Generates predictions for the entire dataset `X`. Since this is a dummy classifier with the "most_frequent" strategy, it will predict the same class for every sample in `X`.

### LOGISTIC REGRESSION

This function trains a logistic regression model using a pipeline that includes data preprocessing. It also uses grid search with cross-validation to find the best hyperparameters for the model, ensuring it generalizes well on unseen data.

```java
logistic_regression = LogisticRegression(max_iter=2000, solver='liblinear')
```

Here, I initialize a logistic regression model. Logistic regression is a linear model used for binary classification tasks. I chose the 'liblinear' solver because it's efficient for small datasets and works well with the regularization techniques I plan to use. I set `max_iter=2000` to allow more iterations for the solver to converge, which was determined based on trial and error

```java
pipeline = Pipeline(steps=[
('preprocessor', preprocessor),
```

```java
('classifier', logistic_regression)
```

```java
])
```

This pipeline is a sequence of steps where the first step is data preprocessing and the second step is the classification model. The `preprocessor` variable contains the preprocessing steps for numeric, categorical, and text data. By including preprocessing in the pipeline, I ensure that the same transformations are applied consistently across all training and testing stages

**Parameter Grid** 
This parameter grid defines the hyperparameters I want to tune. The `C` parameter controls the regularization strength. Lower values of `C` indicate stronger regularization, which can help prevent overfitting. The `penalty` parameter specifies the type of regularization to apply: 'l1' for Lasso (which can result in sparse models) and 'l2' for Ridge (which penalizes large coefficients to prevent overfitting).

**GridSearchCV**

To optimize the model's performance, I use `GridSearchCV`, which performs an exhaustive search over the specified hyperparameter values. It uses 5-fold cross-validation (`cv=5`), meaning the training data is split into 5 parts, and the model is trained and validated 5 times, each time using a different part of the data as the validation set. This helps in finding the best combination of hyperparameters while ensuring that the model isn't overfitting to a specific subset of the data. The scoring metric used is accuracy

```python
pythonCopy code
grid_search.fit(X_train, y_train)

```

**"I then fit the `GridSearchCV` object to the training data. This step trains multiple logistic regression models with different hyperparameter combinations and selects the best model based on cross-validation accuracy."**

```python
return grid_search

```

**"Finally, the function returns the `GridSearchCV` object, which contains the best model found during the search. This model can then be used to make predictions on new data, ensuring it uses the most optimal hyperparameters."**

Overall, this function encapsulates the entire process of model training, hyperparameter tuning, and evaluation within a single pipeline, ensuring that the best possible logistic regression model is selected for the given data.

### **Introduction to XGBoost and Model Definition**

"In this section, we are using the XGBoost model, which is known for its efficiency and performance in classification tasks. XGBoost stands for eXtreme Gradient Boosting, and it’s a powerful algorithm that builds an ensemble of decision trees to make predictions. Here's how we set it up:

```python
pythonCopy code
def xgboost_model(X_train, y_train):
    xgb_classifier = XGBClassifier(use_label_encoder=False, eval_metric='logloss')

```

- **`XGBClassifier`**: This initializes the XGBoost classifier.
- **`use_label_encoder=False`**: This parameter avoids deprecation warnings related to label encoding.
- **`eval_metric='logloss'`**: This specifies that the evaluation metric should be log loss, which measures the performance of the classification model by calculating the distance between the predicted probability and the actual class."
- **Pipeline Construction**
    
    "Next, we set up a pipeline that incorporates our preprocessing steps and the XGBoost classifier:
    
    ```python
    pythonCopy code
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', xgb_classifier)
    ])
    
    ```
    
    - **`Pipeline`**: This helps in chaining together the preprocessing steps and the classifier into a single workflow.
    - **`preprocessor`**: This is the preprocessing pipeline defined earlier, which handles various data transformations such as scaling, encoding, and feature extraction.
    - **`classifier`**: This is the XGBoost classifier."
- **Hyperparameter Grid Definition**
    
    "To find the best parameters for the XGBoost model, we define a parameter grid:
    
    ```python
    pythonCopy code
    param_grid = {
        'classifier__n_estimators': [50, 100, 200],     # Number of trees in the XGBoost model
        'classifier__learning_rate': [0.01, 0.1, 0.2],  # Learning rate
        'classifier__max_depth': [3, 4, 5]              # Maximum depth of each tree
    }
    
    ```
    
    - **`n_estimators`**: This parameter specifies the number of trees (or boosting rounds) in the ensemble. More trees usually improve performance but can lead to overfitting.
    - **`learning_rate`**: This parameter controls how much each tree contributes to the overall prediction. A lower learning rate generally improves performance but requires more trees.
    - **`max_depth`**: This defines the maximum depth of each tree. Deeper trees can capture more complex patterns but may also overfit."
- **Grid Search with Cross-Validation**
    
    "We use GridSearchCV to systematically search for the best combination of hyperparameters:
    
    ```python
    pythonCopy code
    grid_search = GridSearchCV(pipeline, param_grid, cv=5, scoring='accuracy')
    grid_search.fit(X_train, y_train)
    
    ```
    
    - **`GridSearchCV`**: This performs an exhaustive search over the parameter grid defined earlier.
    - **`pipeline`**: The pipeline ensures that preprocessing and model training are conducted in one step.
    - **`param_grid`**: The grid of hyperparameters to be tested.
    - **`cv=5`**: This specifies 5-Fold Cross-Validation to evaluate the model’s performance on different subsets of the training data.
    - **`scoring='accuracy'`**: This metric evaluates the performance based on classification accuracy.
    - **`fit(X_train, y_train)`**: This trains the model on the training data using the best combination of hyperparameters found."
- **Returning the Grid Search Object**
    
    "Finally, the function returns the grid search object, which contains the best model and hyperparameters:
    
    ```python
    pythonCopy code
    return grid_search
    
    ```
    
    - This allows us to access the best parameters and performance metrics after the search is completed."

## Random Forest Model with Hyperparameter Tuning (Explanation for Viva Examiner)

### **Explanation**

### **1. Pipeline**

```python
pythonCopy code
pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(random_state=7))
])

```

- **Pipeline**: This allows you to chain together multiple processing steps and a final estimator.
- **preprocessor**: This is the preprocessing pipeline you have defined earlier. It handles tasks like scaling numeric features, encoding categorical features, and processing text data.
- **RandomForestClassifier**: This is the model you're tuning. Setting `random_state=7` ensures that results are reproducible.

### **2. Parameter Grid**

```python
pythonCopy code
param_grid = {
    'classifier__n_estimators': [100, 200],           # Number of trees in the forest
    'classifier__max_depth': [None, 10, 20],          # Maximum depth of the tree
    'classifier__min_samples_split': [2, 5],          # Minimum number of samples required to split an internal node
    'classifier__min_samples_leaf': [1, 2],           # Minimum number of samples required to be at a leaf node
    'classifier__max_features': ['sqrt', 'log2'],     # Number of features to consider when looking for the best split
    'classifier__bootstrap': [True, False],           # Whether bootstrap samples are used when building trees
    'classifier__class_weight': [None, 'balanced']    # Class weights
}

```

- **`n_estimators`**: The number of trees in the forest. More trees generally improve performance but increase computational cost.
- **`max_depth`**: The maximum depth of the trees. Limiting depth helps prevent overfitting.
- **`min_samples_split`**: The minimum number of samples required to split an internal node. Higher values prevent the model from learning overly specific patterns.
- **`min_samples_leaf`**: The minimum number of samples required at a leaf node. Ensures each leaf has a minimum number of samples.
- **`max_features`**: The number of features to consider when looking for the best split. Can be set to the square root of the number of features (`'sqrt'`) or log base 2 (`'log2'`).
- **`bootstrap`**: Whether to use bootstrap samples when building trees. `True` means that bootstrap sampling is used.
- **`class_weight`**: Balances the weight of classes in the model. Useful for handling class imbalance.

### **3. Cross-Validation Setup**

```python
pythonCopy code
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=7)

```

- **`StratifiedKFold`**: Ensures each fold of the cross-validation has the same proportion of each class as the original dataset. This is crucial for maintaining representative samples in each fold, especially with imbalanced datasets.
- **`n_splits=5`**: Specifies that the data will be split into 5 folds for cross-validation.
- **`shuffle=True`**: Ensures that the data is shuffled before splitting into folds.
- **`random_state=7`**: Ensures reproducibility of the cross-validation splits.

### **4. Grid Search**

```python
pythonCopy code
grid_search = GridSearchCV(pipeline, param_grid, cv=cv, scoring='accuracy', n_jobs=-1, verbose=0)

```

- **`GridSearchCV`**: Performs an exhaustive search over the specified hyperparameter grid.
- **`pipeline`**: The pipeline defined earlier, including preprocessing and the Random Forest model.
- **`param_grid`**: The grid of hyperparameters to search over.
- **`cv=cv`**: The cross-validation strategy.
- **`scoring='accuracy'`**: The metric used to evaluate model performance.
- **`n_jobs=-1`**: Uses all available cores to speed up the search.
- **`verbose=0`**: No output during the search process.

### **5. Fit and Results**

```python
pythonCopy code
grid_search.fit(X_train, y_train)

print("Best parameters found: ", grid_search.best_params_)
print("Best cross-validation score: ", grid_search.best_score_)

```

- **`grid_search.fit(X_train, y_train)`**: Fits the grid search to the training data, performing cross-validation and finding the best hyperparameters.
- **`grid_search.best_params_`**: Displays the best hyperparameters found during the search.
- **`grid_search.best_score_`**: Shows the best cross-validation score achieved with the best hyperparameters.

### Summary

The `random_forest_model` function performs hyperparameter tuning for a Random Forest classifier using a pipeline and grid search with cross-validation. It aims to find the best combination of hyperparameters to optimize model performance. The `StratifiedKFold` cross-validation ensures that each fold is representative of the class distribution in the dataset, and the grid search evaluates different hyperparameter combinations to identify the optimal settings.

## Understanding the Code

The provided code encapsulates a machine learning pipeline that:

1. **Encodes labels:** Converts categorical labels into numerical format for model compatibility.
2. **Splits data:** Divides the dataset into training and validation sets, ensuring class distribution consistency (stratification).
3. **Defines models:** Creates a dictionary of models (Logistic Regression, XGBoost, Random Forest) with corresponding functions.
4. **Trains and evaluates models:** Iterates through models, trains each with hyperparameter tuning, evaluates on validation set, and selects the best performing model.
5. **Predicts on test set:** Uses the best model to predict labels for the test set and prepares a submission file.

### Detailed Explanation

**1. Label Encoding**

XGBoost requires the target labels to be in a numerical format because it processes labels as integers internally during classification tasks.

Python

`label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)`

Use code [with caution.](https://www.notion.so/faq#coding)

- `LabelEncoder` is used to convert categorical labels (e.g., 'A', 'B', 'C') into numerical labels (e.g., 0, 1, 2).
- `fit_transform` learns the mapping from labels to integers and applies it to the target variable `y`.

**2. Data Splitting**

Python

`X_train, X_valid, y_train, y_valid = train_test_split(X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded)`

Use code [with caution.](https://www.notion.so/faq#coding)

- Splits the dataset into training and validation sets.
- `test_size=0.2` indicates 20% of the data will be used for validation.
- `random_state=42` sets a random seed for reproducibility.
- `stratify=y_encoded` ensures that the class distribution in both training and validation sets is similar to the original dataset.

**3. Model Definition**

Python

`models = {
    'Logistic Regression': logistic_regression_model,
    'XGBoost': xgboost_model,
    'Random Forest': random_forest_model
}`

Use code [with caution.](https://www.notion.so/faq#coding)

- Creates a dictionary to store model names and their corresponding functions.

**4. Model Training, Evaluation, and Selection**

Python

`for name, model_func in models.items():
    # ...`

Use code [with caution.](https://www.notion.so/faq#coding)

- Iterates over each model in the dictionary.
- Trains the model using the provided function `model_func`.
- Evaluates the model on the validation set using accuracy score.
- Keeps track of the best model based on validation accuracy.

**5. Predictions and Submission**

Python

`# Predict on the test set using the best model
Y_test_pred = best_model.predict(test)
Y_test_pred = label_encoder.inverse_transform(Y_test_pred)`

Use code [with caution.](https://www.notion.so/faq#coding)

- Uses the best model to predict labels for the test set.
- Converts the predicted numerical labels back to original categorical labels using the `label_encoder`.
- Creates a submission DataFrame and saves it as a CSV file.

### Additional questions:

- **Introduction to XGBoost and Model Definition**
    
    "In this section, we are using the XGBoost model, which is known for its efficiency and performance in classification tasks. XGBoost stands for eXtreme Gradient Boosting, and it’s a powerful algorithm that builds an ensemble of decision trees to make predictions. Here's how we set it up:
    
    ```python
    pythonCopy code
    def xgboost_model(X_train, y_train):
        xgb_classifier = XGBClassifier(use_label_encoder=False, eval_metric='logloss')
    
    ```
    
    - **`XGBClassifier`**: This initializes the XGBoost classifier.
    - **`use_label_encoder=False`**: This parameter avoids deprecation warnings related to label encoding.
    - **`eval_metric='logloss'`**: This specifies that the evaluation metric should be log loss, which measures the performance of the classification model by calculating the distance between the predicted probability and the actual class."
- **Pipeline Construction**
    
    "Next, we set up a pipeline that incorporates our preprocessing steps and the XGBoost classifier:
    
    ```python
    pythonCopy code
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', xgb_classifier)
    ])
    
    ```
    
    - **`Pipeline`**: This helps in chaining together the preprocessing steps and the classifier into a single workflow.
    - **`preprocessor`**: This is the preprocessing pipeline defined earlier, which handles various data transformations such as scaling, encoding, and feature extraction.
    - **`classifier`**: This is the XGBoost classifier."
- **Hyperparameter Grid Definition**
    
    "To find the best parameters for the XGBoost model, we define a parameter grid:
    
    ```python
    pythonCopy code
    param_grid = {
        'classifier__n_estimators': [50, 100, 200],     # Number of trees in the XGBoost model
        'classifier__learning_rate': [0.01, 0.1, 0.2],  # Learning rate
        'classifier__max_depth': [3, 4, 5]              # Maximum depth of each tree
    }
    
    ```
    
    - **`n_estimators`**: This parameter specifies the number of trees (or boosting rounds) in the ensemble. More trees usually improve performance but can lead to overfitting.
    - **`learning_rate`**: This parameter controls how much each tree contributes to the overall prediction. A lower learning rate generally improves performance but requires more trees.
    - **`max_depth`**: This defines the maximum depth of each tree. Deeper trees can capture more complex patterns but may also overfit."
- **Grid Search with Cross-Validation**
    
    "We use GridSearchCV to systematically search for the best combination of hyperparameters:
    
    ```python
    pythonCopy code
    grid_search = GridSearchCV(pipeline, param_grid, cv=5, scoring='accuracy')
    grid_search.fit(X_train, y_train)
    
    ```
    
    - **`GridSearchCV`**: This performs an exhaustive search over the parameter grid defined earlier.
    - **`pipeline`**: The pipeline ensures that preprocessing and model training are conducted in one step.
    - **`param_grid`**: The grid of hyperparameters to be tested.
    - **`cv=5`**: This specifies 5-Fold Cross-Validation to evaluate the model’s performance on different subsets of the training data.
    - **`scoring='accuracy'`**: This metric evaluates the performance based on classification accuracy.
    - **`fit(X_train, y_train)`**: This trains the model on the training data using the best combination of hyperparameters found."
- **Returning the Grid Search Object**
    
    "Finally, the function returns the grid search object, which contains the best model and hyperparameters:
    
    ```python
    pythonCopy code
    return grid_search
    
    ```
    
    - This allows us to access the best parameters and performance metrics after the search is completed."

### **Explanation**

### **1. Pipeline**

```python
pythonCopy code
pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(random_state=7))
])

```

- **Pipeline**: This allows you to chain together multiple processing steps and a final estimator.
- **preprocessor**: This is the preprocessing pipeline you have defined earlier. It handles tasks like scaling numeric features, encoding categorical features, and processing text data.
- **RandomForestClassifier**: This is the model you're tuning. Setting `random_state=7` ensures that results are reproducible.

### **2. Parameter Grid**

```python
pythonCopy code
param_grid = {
    'classifier__n_estimators': [100, 200],           # Number of trees in the forest
    'classifier__max_depth': [None, 10, 20],          # Maximum depth of the tree
    'classifier__min_samples_split': [2, 5],          # Minimum number of samples required to split an internal node
    'classifier__min_samples_leaf': [1, 2],           # Minimum number of samples required to be at a leaf node
    'classifier__max_features': ['sqrt', 'log2'],     # Number of features to consider when looking for the best split
    'classifier__bootstrap': [True, False],           # Whether bootstrap samples are used when building trees
    'classifier__class_weight': [None, 'balanced']    # Class weights
}

```

- **`n_estimators`**: The number of trees in the forest. More trees generally improve performance but increase computational cost.
- **`max_depth`**: The maximum depth of the trees. Limiting depth helps prevent overfitting.
- **`min_samples_split`**: The minimum number of samples required to split an internal node. Higher values prevent the model from learning overly specific patterns.
- **`min_samples_leaf`**: The minimum number of samples required at a leaf node. Ensures each leaf has a minimum number of samples.
- **`max_features`**: The number of features to consider when looking for the best split. Can be set to the square root of the number of features (`'sqrt'`) or log base 2 (`'log2'`).
- **`bootstrap`**: Whether to use bootstrap samples when building trees. `True` means that bootstrap sampling is used.
- **`class_weight`**: Balances the weight of classes in the model. Useful for handling class imbalance.

### **3. Cross-Validation Setup**

```python
pythonCopy code
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=7)

```

- **`StratifiedKFold`**: Ensures each fold of the cross-validation has the same proportion of each class as the original dataset. This is crucial for maintaining representative samples in each fold, especially with imbalanced datasets.
- **`n_splits=5`**: Specifies that the data will be split into 5 folds for cross-validation.
- **`shuffle=True`**: Ensures that the data is shuffled before splitting into folds.
- **`random_state=7`**: Ensures reproducibility of the cross-validation splits.

### **4. Grid Search**

```python
pythonCopy code
grid_search = GridSearchCV(pipeline, param_grid, cv=cv, scoring='accuracy', n_jobs=-1, verbose=0)

```

- **`GridSearchCV`**: Performs an exhaustive search over the specified hyperparameter grid.
- **`pipeline`**: The pipeline defined earlier, including preprocessing and the Random Forest model.
- **`param_grid`**: The grid of hyperparameters to search over.
- **`cv=cv`**: The cross-validation strategy.
- **`scoring='accuracy'`**: The metric used to evaluate model performance.
- **`n_jobs=-1`**: Uses all available cores to speed up the search.
- **`verbose=0`**: No output during the search process.

### **5. Fit and Results**

```python
pythonCopy code
grid_search.fit(X_train, y_train)

print("Best parameters found: ", grid_search.best_params_)
print("Best cross-validation score: ", grid_search.best_score_)

```

- **`grid_search.fit(X_train, y_train)`**: Fits the grid search to the training data, performing cross-validation and finding the best hyperparameters.
- **`grid_search.best_params_`**: Displays the best hyperparameters found during the search.
- **`grid_search.best_score_`**: Shows the best cross-validation score achieved with the best hyperparameters.

### Summary

The `random_forest_model` function performs hyperparameter tuning for a Random Forest classifier using a pipeline and grid search with cross-validation. It aims to find the best combination of hyperparameters to optimize model performance. The `StratifiedKFold` cross-validation ensures that each fold is representative of the class distribution in the dataset, and the grid search evaluates different hyperparameter combinations to identify the optimal settings.
