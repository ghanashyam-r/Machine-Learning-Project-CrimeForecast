# Machine-Learning-Project-CrimeForecast
This Machine Learning project was completed on Kaggle as a part of gaining required credits for achieving Diploma in Data Science IITM.I secured an 'S' grade for the project.The competition was based on a Classification Problem with the LA crime dataset and i managed to achieve an Accuracy of 95.6 percent with my XGBoost Model.

The project procedure required meeting a cutoff of 80 percent and successfully clearing two vivas. The Level 2 viva was conducted by an industry professional.
*Code available in the MLP final notebook file*

## Detailed Description of Code
### Imports
I first imported all the relevant libraries:
**NumPy and Pandas**: These are essential libraries for data manipulation and analysis. Pandas is specifically used for reading and handling CSV files.
**Sklearn Libraries**: These are essential for machine learning tasks such as imputing missing values, preprocessing data, model training, and evaluation.
**Matplotlib and Seaborn**: These libraries are used for data visualization, allowing you to understand your data better and communicate findings.

### EDA
The training, testing, and sample datasets are loaded using `pd.read_csv()`.The data is imported and stored into corresponding variables.
After that i wrote code for exploring the basic features of the data such as ‘shape’ to display number of rows and columns in the test and train datasets.'info()' provides detailed information about the dataset, including column names, data types, and non-null counts, which helps in understanding the data structure and identifying any issues such as missing values.'describe()` provides a statistical summary of the data, including count, mean, standard deviation, and other metrics
*From using the describe() function it is noticed that there are some 'zero' values in the latitude and longitude which were found to be outliers.Also there were negative values in the age column.*'isnull().sum()' is used to count the number of missing values in each column. This is an essential step in data cleaning as missing data can affect the performance of machine learning models.

The code first addresses zero values in the `Latitude` and `Longitude` columns of both the training and testing datasets which are outliers.
The scatter plot represents the geographical distribution of different categories of crimes based on their latitude and longitude. Each dot corresponds to a specific crime incident, and the color of the dot indicates the type of crime.
![Capture-2024-09-04-215824](https://github.com/user-attachments/assets/2ed02978-cfd1-4edb-be57-d59eb24d82d1)

*From this plot, I can observe that crimes are spread across the area, but certain categories seem more concentrated in specific regions. For example, property crimes (in blue) appear widely distributed, while other categories like fraud and white-collar crimes (in purple) might be more localized.*

This visualization helps in identifying **patterns in the geographical occurrence of crimes.** It also highlights the need to explore clustering techniques to group similar locations together, which could simplify the analysis by revealing crime hotspots or areas with distinct crime patterns. Clustering would allow me to better understand and predict where different types of crimes are more likely to occur.

The correlation matrix is calculated using the `.corr()` function, which computes the pairwise correlation of all numeric columns.
![Capture-2024-09-04-215838](https://github.com/user-attachments/assets/12ab4aa4-43cb-4165-aa0f-2b2511e454e8)

*Area_ID and Reporting District number exactly coorelates which means one of them can be safely dropped.*

Box Plot of Agegoup to Victim_Age is plotted.
![Capture-2024-09-04-215854](https://github.com/user-attachments/assets/0e6618aa-883e-4955-bfd8-49d73da7203d)
*Different age groups are victim to different kind of crimes so age binning column would help increase the the accuracy.*

### PREPROCESSING

Geographical data was handeled using KMeans Clustering and Adding a new feature 'Cluster'
Converted the Modus Opperandi coloumn Values to text data for better capturing of the essence of the data.
I also replaced the placeholder values in Victim_Sex with np.nan to be preprocessed later.
I also converted date columns to datetime objects for easier manipulation and feature extraction and created 6 new features like reported day,reported month,occured year..etc
I handled the negative ages in the dataset replacing them with the median.I also binned the ages into ranges which simplifies the model's task by grouping ages into categories.
I also created a feature to understand reporting delays by calculating time difference between 'Date_Reported' and 'Date_Occurred'.I dropped reduntant columns like 'Date_Reported', 'Date_Occurred', and 'Area_ID'.


I created pipelines for Numeric(Imputation and scaling ensure that numeric features are on a comparable scale and have no missing values),
Categorical(Imputation and encoding convert categorical variables into a format suitable for ml models.) and 
Text Transformers(Vectorizing the 'Modus_Operandi' column allows the model to process text data as numerical features.).

The `ColumnTransformer` combines these three pipelines into one.
I then created a Baseline Model to predict the most frequent class in the training data to act as a baseline standard for my ML models.

## Models

### LOGISTIC REGRESSION with Hyperparameter Tuning

This function trains a logistic regression model using a pipeline that includes data preprocessing. It also uses grid search with cross-validation to find the best hyperparameters for the model, ensuring it generalizes well on unseen data.

Logistic regression is a linear model used for binary classification tasks but can be used for Multiclass Classification problems. I chose the 'liblinear' solver because it's efficient for small datasets and works well with the regularization techniques I plan to use. I set `max_iter=2000` to allow more iterations for the solver to converge, which was determined based on trial and error

**Parameter Grid** 
The parameter grid defines the hyperparameters I want to tune. The `C` parameter controls the regularization strength. Lower values of `C` indicate stronger regularization, which can help prevent overfitting. The `penalty` parameter specifies the type of regularization to apply: 'l1' for Lasso (which can result in sparse models) and 'l2' for Ridge (which penalizes large coefficients to prevent overfitting).

**GridSearchCV**

To optimize the model's performance, I used `GridSearchCV`, which performs an exhaustive search over the specified hyperparameter values. It uses 5-fold cross-validation (`cv=5`), meaning the training data is split into 5 parts, and the model is trained and validated 5 times, each time using a different part of the data as the validation set. This helps in finding the best combination of hyperparameters while ensuring that the model isn't overfitting to a specific subset of the data. The scoring metric used is accuracy



### XGBOOST with Hyperparameter Tuning

I also used the XGBoost model, which is known for its efficiency and performance in classification tasks. XGBoost stands for eXtreme Gradient Boosting, and it’s a powerful algorithm that builds an ensemble of decision trees to make predictions. 


- **`eval_metric='logloss'`**: This specifies that the evaluation metric should be log loss, which measures the performance of the classification model by calculating the distance between the predicted probability and the actual class."

- **Hyperparameter Grid Definition**
    
    "To find the best parameters for the XGBoost model, a parameter grid was defined:
    
    ```
    param_grid = {
        'classifier__n_estimators': [50, 100, 200],     # Number of trees in the XGBoost model
        'classifier__learning_rate': [0.01, 0.1, 0.2],  # Learning rate
        'classifier__max_depth': [3, 4, 5]              # Maximum depth of each tree
    }
    
    ```
    
    - **`n_estimators`**: This parameter specifies the number of trees (or boosting rounds) in the ensemble. More trees usually improve performance but can lead to overfitting.
    - **`learning_rate`**: This parameter controls how much each tree contributes to the overall prediction. A lower learning rate generally improves performance but requires more trees.
    - **`max_depth`**: This defines the maximum depth of each tree. Deeper trees can capture more complex patterns but may also overfit."
      
**GridSearchCV**
GridSearchCV with 5 fold cross validation was used to systematically search for the best combination of hyperparameters and evaluate the model’s performance on different subsets of the training data.
    
  


### Random Forest Model with Hyperparameter Tuning 

The `random_forest_model` function performs hyperparameter tuning for a Random Forest classifier using a pipeline and grid search with cross-validation. It aims to find the best combination of hyperparameters to optimize model performance. The `StratifiedKFold` cross-validation ensures that each fold is representative of the class distribution in the dataset, and the grid search evaluates different hyperparameter combinations to identify the optimal settings.



### Running all the models
XGBoost requires the target labels to be in a numerical format because it processes labels as integers internally during classification tasks.
I split the training  dataset using train test split:
`X_train, X_valid, y_train, y_valid = train_test_split(X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded)`

- Splits the dataset into training and validation sets.
- `test_size=0.2` indicates 20% of the data will be used for validation.
- `random_state=42` sets a random seed for reproducibility.
- `stratify=y_encoded` ensures that the class distribution in both training and validation sets is similar to the original dataset.
Classification Reports of all models were also printed.

## Summary and Inferences
XGBoost has the highest validation accuracy and best cross-validation score,indicating that the model generalizes well and is useful when the primary goal is maximizing accuracy.

Logistic Regression can be a good alternaive when a simpler model is required without a significant drop in the accuracy.

Random Forest does not match the accuracy of the other two models, likely due to overfitting or not capturing the underlying data distribution as effectively as XGBoost.

I implemented the best-performing model on the test dataset, made predictions, and saved the results in the required format. The solution was submitted on Kaggle, and I obtained a final accuracy score of 95.6%.
