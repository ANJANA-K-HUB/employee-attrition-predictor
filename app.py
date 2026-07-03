import pickle
import shap
import pandas as pd
from flask import Flask, render_template, request

app = Flask(__name__)

# Load the comprehensive pipeline (preprocessor + model)
with open('model.pkl', 'rb') as file:
    pipeline = pickle.load(file)
# Create the SHAP analyzer engine by extracting the classifier step
model_step = pipeline.named_steps['classifier']
explainer = shap.TreeExplainer(model_step)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # 1. Capture the form inputs exactly matching the new data structure columns
    form_data = {
        'EnvironmentSatisfaction': [int(request.form['EnvironmentSatisfaction'])],
        'JobSatisfaction': [int(request.form['JobSatisfaction'])],
        'MonthlyIncome': [float(request.form['MonthlyIncome'])],
        'OverTime': [request.form['OverTime']],
        'TotalWorkingYears': [int(request.form['TotalWorkingYears'])],
        'YearsAtCompany': [int(request.form['YearsAtCompany'])],
        'WorkLifeBalance': [int(request.form['WorkLifeBalance'])]
    }
    
    
    # 2. Convert to DataFrame format so the pipeline can map it seamlessly
    input_df = pd.DataFrame(form_data)

    # 3. Compute inference using the backend pipeline
    prediction = pipeline.predict(input_df)

    # 1. Extract the preprocessor step from your pipeline
    preprocessor_step = pipeline.named_steps['preprocessor']

    # 2. Transform the user input raw data into processed numbers
    transformed_data = preprocessor_step.transform(input_df)

    # 3. Calculate SHAP values for this transformed row
    shap_values = explainer.shap_values(transformed_data)

    # 4. Get feature names from the preprocessor to match with values
    feature_names = preprocessor_step.get_feature_names_out()

    # 5. Extract SHAP values for Class 1 (Attrition) for our single row
    # (Depending on scikit-learn version, it might be shap_values[1][0] or shap_values[0][:, 1])
    row_shap = shap_values[1][0] if isinstance(shap_values, list) else shap_values[0, :, 1]

    # 6. Combine feature names and their impact into a dictionary
    feature_importance = dict(zip(feature_names, row_shap))

    # 7. Sort features to find which positive values push hardest toward Attrition
    sorted_factors = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
    
    
    # Create an empty list to hold clean, distinct feature names
    distinct_reasons = []

    for factor in sorted_factors:
        feature_name = factor[0]
    
        # Clean up the prefix decorators from scikit-learn
        clean_name = feature_name.replace('cat__', '').replace('num__', '').replace('remainder__', '')
        
        # Strip out the '_Yes' or '_No' suffix to find the true root feature name
        root_name = clean_name.split('_')[0]
        
        # Check if we've already added a variation of this root feature
        # We use a tracking condition to check existing entries
        if not any(root_name in existing for existing in distinct_reasons):
            distinct_reasons.append(clean_name)
            
        # Once we have 2 completely unique base features, we stop looking!
        if len(distinct_reasons) == 2:
            break

# Assign your final, distinct strings
    reason1 = distinct_reasons[0]
    reason2 = distinct_reasons[1] if len(distinct_reasons) > 1 else "N/A"

    # 4. Construct response message
    if prediction[0] == 1:
        output_text = "Analysis Result: High Attrition Risk. Flags potential resignation trend."
        explanation_text = f"Top Risk Factors driving this: 1. {reason1}, 2. {reason2}"
    else:
        output_text = "Analysis Result: Stable Employee Profile. Retentive metrics look solid."
        explanation_text = "No immediate workplace adjustments required based on current metrics."

    return render_template('index.html', prediction_text=output_text, 
                       explanation_text=explanation_text)

if __name__ == "__main__":
    app.run(debug=True)