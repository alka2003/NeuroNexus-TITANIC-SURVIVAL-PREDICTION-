import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.model_selection import GridSearchCV
from xgboost import XGBClassifier
from catboost import CatBoostClassifier

# Load the dataset
data = pd.read_csv('tested.csv')

# Display basic information
print(data.info())
print(data.describe())
print(data.head())

# Check for missing values
print(data.isnull().sum())

# Set style for plots
sns.set_style('whitegrid')

# Survival count
plt.figure(figsize=(8, 6))
sns.countplot(x='Survived', data=data)
plt.title('Survival Count')
plt.show()

# Survival by sex
plt.figure(figsize=(8, 6))
sns.countplot(x='Survived', hue='Sex', data=data)
plt.title('Survival by Sex')
plt.show()

# Survival by passenger class
plt.figure(figsize=(8, 6))
sns.countplot(x='Survived', hue='Pclass', data=data)
plt.title('Survival by Passenger Class')
plt.show()

# Age distribution
plt.figure(figsize=(10, 6))
sns.histplot(data['Age'].dropna(), bins=30, kde=True)
plt.title('Age Distribution')
plt.show()

# Fare distribution
plt.figure(figsize=(10, 6))
sns.histplot(data['Fare'], bins=30, kde=True)
plt.title('Fare Distribution')
plt.show()

# Correlation heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(data.corr(numeric_only=True), annot=True, cmap='coolwarm')
plt.title('Correlation Heatmap')
plt.show()

# Create new features
data['FamilySize'] = data['SibSp'] + data['Parch'] + 1
data['IsAlone'] = 0
data.loc[data['FamilySize'] == 1, 'IsAlone'] = 1

# Extract title from name
data['Title'] = data['Name'].str.extract(' ([A-Za-z]+)\.', expand=False)
data['Title'] = data['Title'].replace(['Lady', 'Countess','Capt', 'Col', 'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')
data['Title'] = data['Title'].replace('Mlle', 'Miss')
data['Title'] = data['Title'].replace('Ms', 'Miss')
data['Title'] = data['Title'].replace('Mme', 'Mrs')

# Fill missing age values based on title
title_age_median = data.groupby('Title')['Age'].median()
for title in data['Title'].unique():
    data.loc[(data['Age'].isnull()) & (data['Title'] == title), 'Age'] = title_age_median[title]

# Fill missing fare values with median fare of passenger's class
data['Fare'] = data['Fare'].fillna(data.groupby('Pclass')['Fare'].transform('median'))

# Fill missing embarked values with mode
data['Embarked'] = data['Embarked'].fillna(data['Embarked'].mode()[0])

# Drop unnecessary columns
data = data.drop(['PassengerId', 'Name', 'Ticket', 'Cabin'], axis=1)

# Convert categorical variables to numerical
data['Sex'] = data['Sex'].map({'male': 0, 'female': 1})
data['Embarked'] = data['Embarked'].map({'S': 0, 'C': 1, 'Q': 2})
data['Title'] = data['Title'].map({'Mr': 0, 'Miss': 1, 'Mrs': 2, 'Master': 3, 'Rare': 4})

# Create age bins
data['AgeBin'] = pd.cut(data['Age'].astype(int), 5, labels=False)

# Create fare bins
data['FareBin'] = pd.qcut(data['Fare'], 4, labels=False)

# Drop original age and fare columns
data = data.drop(['Age', 'Fare'], axis=1)

# Separate features and target
X = data.drop('Survived', axis=1)
y = data['Survived']

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Define preprocessing for numerical and categorical features
numeric_features = ['Pclass', 'SibSp', 'Parch', 'FamilySize', 'IsAlone', 'AgeBin', 'FareBin']
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())])

categorical_features = ['Sex', 'Embarked', 'Title']
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))])

# Combine preprocessing steps
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)])


# Define models
models = {
    'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
    'Random Forest': RandomForestClassifier(random_state=42),
    'SVM': SVC(random_state=42),
    'Gradient Boosting': GradientBoostingClassifier(random_state=42),
    'XGBoost': XGBClassifier(random_state=42, eval_metric='logloss'),
    'CatBoost': CatBoostClassifier(random_state=42, verbose=0)
}

# Evaluate each model
results = {}
for name, model in models.items():
    # Create pipeline
    pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                              ('classifier', model)])
    
    # Fit the model
    pipeline.fit(X_train, y_train)
    
    # Make predictions
    y_pred = pipeline.predict(X_test)
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    
    # Store results
    results[name] = {
        'Accuracy': accuracy,
        'Precision': precision,
        'Recall': recall,
        'F1 Score': f1
    }
    
    # Print classification report
    print(f"\n{name} Classification Report:")
    print(classification_report(y_test, y_pred))
    
    # Plot confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f'{name} Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.show()

# Display results dataframe
results_df = pd.DataFrame(results).T
print("\nModel Comparison:")
print(results_df)

# Let's tune Random Forest as it often performs well
param_grid = {
    'classifier__n_estimators': [100, 200, 300],
    'classifier__max_depth': [None, 5, 10],
    'classifier__min_samples_split': [2, 5, 10],
    'classifier__min_samples_leaf': [1, 2, 4]
}

# Create pipeline with preprocessor and classifier
rf_pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                             ('classifier', RandomForestClassifier(random_state=42))])

# Grid search
grid_search = GridSearchCV(rf_pipeline, param_grid, cv=5, scoring='accuracy', n_jobs=-1)
grid_search.fit(X_train, y_train)

# Best parameters
print("Best parameters:", grid_search.best_params_)
print("Best score:", grid_search.best_score_)

# Evaluate best model
best_model = grid_search.best_estimator_
y_pred = best_model.predict(X_test)

print("\nOptimized Random Forest Classification Report:")
print(classification_report(y_test, y_pred))

# Feature importance
# Get feature names after one-hot encoding
feature_names = numeric_features.copy()
ohe_categories = pipeline.named_steps['preprocessor'].named_transformers_['cat'].named_steps['onehot'].categories_
for i, cat in enumerate(categorical_features):
    for value in ohe_categories[i]:
        feature_names.append(f"{cat}_{value}")

# Get feature importances
importances = best_model.named_steps['classifier'].feature_importances_
indices = np.argsort(importances)[::-1]

# Plot feature importances
plt.figure(figsize=(12, 6))
plt.title("Feature Importances")
plt.bar(range(X_train.shape[1]), importances[indices], align="center")
plt.xticks(range(X_train.shape[1]), [feature_names[i] for i in indices], rotation=90)
plt.tight_layout()
plt.show()


# Based on our evaluation, let's choose the best performing model
final_model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(
        max_depth=None,
        min_samples_leaf=1,
        min_samples_split=2,
        n_estimators=200,
        random_state=42))
])

# Train on full training data
final_model.fit(X_train, y_train)

# Evaluate on test set
y_pred = final_model.predict(X_test)
print("\nFinal Model Classification Report:")
print(classification_report(y_test, y_pred))

# Save the model
import joblib
joblib.dump(final_model, 'titanic_survival_model.pkl')

# Function to make predictions on new data
def predict_survival(model, new_data):
    """
    Predict survival for new passenger data.
    
    Args:
        model: Trained model pipeline
        new_data: DataFrame with passenger data in the same format as training data
        
    Returns:
        Array of predictions (0 = did not survive, 1 = survived)
    """
    # Preprocess new data (same steps as training data)
    new_data = new_data.copy()
    
    # Feature engineering
    new_data['FamilySize'] = new_data['SibSp'] + new_data['Parch'] + 1
    new_data['IsAlone'] = 0
    new_data.loc[new_data['FamilySize'] == 1, 'IsAlone'] = 1
    
    # Extract title from name
    new_data['Title'] = new_data['Name'].str.extract(' ([A-Za-z]+)\.', expand=False)
    new_data['Title'] = new_data['Title'].replace(['Lady', 'Countess','Capt', 'Col', 'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')
    new_data['Title'] = new_data['Title'].replace('Mlle', 'Miss')
    new_data['Title'] = new_data['Title'].replace('Ms', 'Miss')
    new_data['Title'] = new_data['Title'].replace('Mme', 'Mrs')
    
    # Fill missing values
    if 'Age' in new_data.columns:
        new_data['Age'] = new_data['Age'].fillna(data['Age'].median())
    if 'Fare' in new_data.columns:
        new_data['Fare'] = new_data['Fare'].fillna(data.groupby('Pclass')['Fare'].transform('median'))
    if 'Embarked' in new_data.columns:
        new_data['Embarked'] = new_data['Embarked'].fillna(data['Embarked'].mode()[0])
    
    # Convert categorical variables
    new_data['Sex'] = new_data['Sex'].map({'male': 0, 'female': 1})
    new_data['Embarked'] = new_data['Embarked'].map({'S': 0, 'C': 1, 'Q': 2})
    new_data['Title'] = new_data['Title'].map({'Mr': 0, 'Miss': 1, 'Mrs': 2, 'Master': 3, 'Rare': 4})
    
    # Create bins
    if 'Age' in new_data.columns:
        new_data['AgeBin'] = pd.cut(new_data['Age'].astype(int), 5, labels=False)
    if 'Fare' in new_data.columns:
        new_data['FareBin'] = pd.qcut(new_data['Fare'], 4, labels=False)
    
    # Drop unnecessary columns
    cols_to_drop = ['PassengerId', 'Name', 'Ticket', 'Cabin', 'Age', 'Fare']
    cols_to_drop = [col for col in cols_to_drop if col in new_data.columns]
    new_data = new_data.drop(cols_to_drop, axis=1)
    
    # Ensure all required columns are present
    required_columns = ['Pclass', 'Sex', 'SibSp', 'Parch', 'Embarked', 'FamilySize', 'IsAlone', 'Title', 'AgeBin', 'FareBin']
    for col in required_columns:
        if col not in new_data.columns:
            new_data[col] = 0  # Default value if column is missing
    
    # Reorder columns to match training data
    new_data = new_data[required_columns]
    
    # Make prediction
    prediction = model.predict(new_data)
    probability = model.predict_proba(new_data)[:, 1]
    
    return prediction, probability

# Example usage:
# new_passenger = pd.DataFrame({
#     'Pclass': [1],
#     'Name': ['Smith, Mr. John'],
#     'Sex': ['male'],
#     'Age': [30],
#     'SibSp': [0],
#     'Parch': [0],
#     'Ticket': ['PC 12345'],
#     'Fare': [50],
#     'Cabin': ['C123'],
#     'Embarked': ['S']
# })
# pred, prob = predict_survival(final_model, new_passenger)
# print(f"Prediction: {'Survived' if pred[0] == 1 else 'Did not survive'} (Probability: {prob[0]:.2f})")


# Save this as app.py
import streamlit as st
import joblib
import pandas as pd

# Load the model
model = joblib.load('titanic_survival_model.pkl')

st.title('Titanic Survival Prediction')

st.write("""
Enter passenger details to predict survival probability:
""")

# Input fields
pclass = st.selectbox('Passenger Class', [1, 2, 3])
name = st.text_input('Name')
sex = st.selectbox('Sex', ['male', 'female'])
age = st.number_input('Age', min_value=0, max_value=100, value=30)
sibsp = st.number_input('Number of Siblings/Spouses Aboard', min_value=0, max_value=10, value=0)
parch = st.number_input('Number of Parents/Children Aboard', min_value=0, max_value=10, value=0)
fare = st.number_input('Fare', min_value=0, max_value=600, value=50)
embarked = st.selectbox('Port of Embarkation', ['S', 'C', 'Q'])

if st.button('Predict Survival'):
    # Create dataframe
    passenger = pd.DataFrame({
        'Pclass': [pclass],
        'Name': [name],
        'Sex': [sex],
        'Age': [age],
        'SibSp': [sibsp],
        'Parch': [parch],
        'Fare': [fare],
        'Embarked': [embarked]
    })
    
    # Make prediction
    pred, prob = predict_survival(model, passenger)
    result = 'Survived' if pred[0] == 1 else 'Did not survive'
    confidence = prob[0] if pred[0] == 1 else 1 - prob[0]
    
    st.success(f"Prediction: {result} (Confidence: {confidence:.2%})")
    
    # Show feature importance explanation
    st.subheader('Key Factors Influencing Prediction:')
    if sex == 'female':
        st.write("- Being female significantly increased survival chances")
    else:
        st.write("- Being male significantly decreased survival chances")
    
    if pclass == 1:
        st.write("- First class passengers had higher survival rates")
    elif pclass == 3:
        st.write("- Third class passengers had lower survival rates")
    
    if age < 16:
        st.write("- Children had higher survival rates")
    elif age > 50:
        st.write("- Older passengers had lower survival rates")
    
    if sibsp + parch > 0:
        st.write("- Traveling with family members affected survival chances")