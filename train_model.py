import pickle
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier

# 1. Load the data 
df = pd.read_csv('hr_data.xls')


features = [
    'EnvironmentSatisfaction', 
    'JobSatisfaction', 
    'MonthlyIncome', 
    'OverTime', 
    'TotalWorkingYears',
    'YearsAtCompany',
    'WorkLifeBalance'
]
# target column is 'Attrition' (where 'Yes' means left, 'No' means stayed)
target = 'Attrition'

X = df[features]
y = df[target].apply(lambda x: 1 if str(x).strip().lower() == 'yes' else 0) # Convert 'Yes'/'No' to 1/0

# 3. Separate numeric and categorical tracking
numeric_features = ['EnvironmentSatisfaction', 'JobSatisfaction', 'MonthlyIncome', 'TotalWorkingYears', 'YearsAtCompany', 'WorkLifeBalance']
categorical_features = ['OverTime']

# Preprocessing configuration
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numeric_features),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ])

# 4. Bind preprocessing and model together
pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(random_state=42, n_estimators=100))
])

# 5. Train the structural pipeline
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
pipeline.fit(X_train, y_train)

print(f"Model trained successfully! Test Accuracy: {pipeline.score(X_test, y_test):.2f}")

# 6. Save the pipeline package
with open('model.pkl', 'wb') as file:
    pickle.dump(pipeline, file)

print("Production-ready pipeline saved as 'model.pkl'.")