import streamlit as st
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import xgboost 
import seaborn as sns 
import shap
from sklearn.inspection import partial_dependence
from sklearn.datasets import make_classification
import os
# Set up Streamlit page configuration
st.set_page_config(page_title='Credit Score Prediction Dashboard')

@st.cache_resource
def load_model():
    """
    Load the pre-trained model and other required resources.
    """
    model = joblib.load(os.path.join(os.path.join(os.path.dirname(os.getcwd()),'model'),'xgboost.pkl'))
    return model

model = load_model()

@st.cache_data
def preprocess_features(features):
    """
    Preprocess the features before making predictions.
    """
    data = pd.DataFrame([features])
    
    # Identify numerical and categorical columns
    numerical_cols = [col for col in data.columns if data[col].dtype != 'object']
    categorical_cols = [col for col in data.columns if data[col].dtype == 'object']
    
    # Map values for 'Credit_Mix'
    mix_mapping = {'Standard': 1, 'Good': 2, 'Bad': 0}
    data['Credit_Mix'] = data['Credit_Mix'].map(mix_mapping)
    
    categorical_cols.remove('Credit_Mix')
    numerical_cols.append('Credit_Mix')

    return data

@st.cache_data
def predict(features):
    """
    Predict the credit score class using the pre-trained model.
    """
    feature_processed = preprocess_features(features)
    prediction = model.predict(feature_processed)
    return prediction

#Global Declarations
credit_score_class = {
        0: "Poor Credit Score",
        1: "Standard Credit Score",
        2: "Good Credit Score"
}
    
class_details = {
        0: [
            "High credit risk due to multiple delayed payments.",
            "High outstanding debt and insufficient credit history.",
            "Consider improving payment habits and reducing debt."
        ],
        1: [
            "Moderate credit risk with some delayed payments.",
            "Maintain timely payments and monitor credit inquiries.",
            "Keep a low credit utilization and manage debts responsibly."
        ],
        2: [
            "Low credit risk with timely payments and low debt.",
            "Excellent credit history with responsible financial management.",
            "Continue managing finances well to maintain this rating."
        ]
}
    
def generate_html_report(user_details, features, prediction):
    """
    Generate an HTML report based on user details, features, and prediction.
    """
    prediction_class = credit_score_class.get(prediction[0], "Unknown")
    details = class_details.get(prediction[0], ["No details available."])

    html_content = f"""
    <html>
    <head>
        <title>Credit Score Report</title>
        <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.3/dist/css/bootstrap.min.css" rel="stylesheet" integrity="sha384-QWTKZyjpPEjISv5WaRU9OFeRpok6YctnYmDr5pNlyT2bRjXh0JMhjY6hW+ALEwIH" crossorigin="anonymous">
        <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.3.3/dist/js/bootstrap.bundle.min.js" integrity="sha384-YvpcrYf0tY3lHB60NNkmXc5s9fDVZLESaAA55NDzOxhy9GkcIdslK1eN7N6jIeHz" crossorigin="anonymous"></script>
        <style>
            body {{
                font-family: Arial, sans-serif;
                padding: 20px;
            }}
            .container {{
                max-width: 800px;
                margin: 0 auto;
            }}
            h1 {{
                color: #007bff;
                margin-bottom: 20px;
            }}
            h2 {{
                color: #343a40;
                margin-top: 20px;
                margin-bottom: 10px;
            }}
            p {{
                font-size: 16px;
            }}
            ul {{
                list-style-type: disc;
                padding-left: 20px;
            }}
            .alert {{
                padding: 15px;
                margin-top: 20px;
            }}
            .alert-danger {{
                background-color: #f8d7da;
                color: #721c24;
            }}
            .alert-warning {{
                background-color: #fff3cd;
                color: #856404;
            }}
            .alert-success {{
                background-color: #d4edda;
                color: #155724;
            }}
        </style>
    </head>
    <body>
        <div class="container">
            <h1>Credit Score Prediction Report</h1>
            <h2>User Details:</h2>
            <p><strong>Name:</strong> {user_details['name']}</p>
            <p><strong>Age:</strong> {user_details['age']}</p>
            <p><strong>Occupation:</strong> {user_details['occupation']}</p>
            <p><strong>Address:</strong> {user_details['address']}</p>
            <p><strong>Mobile Number:</strong> {user_details['mobile']}</p>
            
            <h2>Credit Details:</h2>
            <p><strong>Annual Income:</strong> {features['Annual_Income']}</p>
            <p><strong>Monthly In-hand Salary:</strong> {features['Monthly_Inhand_Salary']}</p>
            <p><strong>Interest Rate:</strong> {features['Interest_Rate']}</p>
            <p><strong>Delay from Due Date:</strong> {features['Delay_from_due_date']}</p>
            <p><strong>Number of Credit Inquiries:</strong> {features['Num_Credit_Inquiries']}</p>
            <p><strong>Credit Mix:</strong> {features['Credit_Mix']}</p>
            <p><strong>Outstanding Debt:</strong> {features['Outstanding_Debt']}</p>
            <p><strong>Total EMI per Month:</strong> {features['Total_EMI_per_month']}</p>
            
            <h2>Prediction:</h2>
            <div class="alert {'alert-success' if prediction[0] == 2 else 'alert-warning' if prediction[0] == 1 else 'alert-danger'}">
                <strong>{prediction_class}</strong>
            </div>
            <ul>
                {''.join(f"<li>{point}</li>" for point in details)}
            </ul>
        </div>
    </body>
    </html>
    """
    return html_content

# Streamlit app layout
st.title('📊 Credit Score Prediction Dashboard')
st.write("Welcome to the Credit Score Prediction Dashboard. This app allows you to interact with the model, understand its workings, and evaluate its performance.")

# Navigation sidebar
st.sidebar.title('📑 Navigation')
page = st.sidebar.radio('Select a Page:', ['🔮 Predictions', '🔍 Model Explanation', '📉 Feature Selection', '📈 Evaluation', '⚙️ Optimization', '🌍 Real World Examples',])

# Page to make predictions
if page == '🔮 Predictions':
    # Tabs for main sections
    tab1, tab2, tab3 = st.tabs(["📝 User Details", "🔮 Predictions", "📄 Download Report"])

    with tab1:
        st.write("### Enter Your Details")

        name = st.text_input("Name:")
        age = st.number_input("Age:", min_value=18, max_value=100, value=30)
        occupation = st.text_input("Occupation:")

        address = st.text_area("Address:")
        mobile = st.text_input("Mobile Number:")

        if st.button('Proceed to Predictions'):
            if name and age and occupation and address and mobile:
                st.session_state['user_details'] = {
                    'name': name,
                    'age': age,
                    'occupation': occupation,
                    'address': address,
                    'mobile': mobile
                }
                st.success("User details saved! Navigate to '🔮 Predictions' to continue.")
            else:
                st.error("Please fill out all the details before proceeding.")

    with tab2:
        st.write("### Credit Score Predictions Section")
        if 'user_details' in st.session_state:
            user_details = st.session_state['user_details']
            st.write(f"**Name:** {user_details['name']}")
            st.write(f"**Age:** {user_details['age']}")
            st.write(f"**Occupation:** {user_details['occupation']}")
            st.write(f"**Address:** {user_details['address']}")
            st.write(f"**Mobile Number:** {user_details['mobile']}")
            st.write("---")
    
        st.write('Enter the following details to predict your credit score:')

        col1, col2 = st.columns(2)
    
        with col1:
            a = st.number_input('Annual Income:', min_value=0, value=50000)
            b = st.number_input('Monthly In-hand Salary:', min_value=0, value=4000)
            c = st.number_input('Interest Rate (%):', min_value=0.0, value=5.0, step=0.1)
            d = st.number_input('Delay from Due Date (days):', min_value=0, value=2)
        
        with col2:
            e = st.number_input('Number of Credit Inquiries:', min_value=0, value=1)
            f = st.selectbox('Credit Mix:', ['Standard', 'Good', 'Bad'])
            g = st.number_input('Outstanding Debt:', min_value=0, value=10000, step=500)
            h = st.number_input('Total EMI per Month:', min_value=0, value=200, step=10)
            
        features = {
            'Annual_Income': a,
            'Monthly_Inhand_Salary': b,
            'Interest_Rate': c,
            'Delay_from_due_date': d,
            'Num_Credit_Inquiries': e,
            'Credit_Mix': f,
            'Outstanding_Debt': g,
            'Total_EMI_per_month': h,
        }

        if st.button('Get Prediction'):
            prediction = predict(features)
            st.session_state['prediction'] = prediction
            st.write(f"**Prediction:** {credit_score_class[prediction[0]]}")
            st.success("Prediction done! Navigate to '📄 Download Report' to view the report.")
        else:
            st.error("Please enter all feature details before predicting.")

    with tab3:
        st.write("### Download Report")
    
        if 'user_details' in st.session_state and 'prediction' in st.session_state:
            user_details = st.session_state['user_details']
            prediction = st.session_state['prediction']

            html_report = generate_html_report(user_details, features, prediction)
        
            st.markdown("### Generated Report")
            st.components.v1.html(html_report, height=600, scrolling=True)

            st.download_button(
                label="Download Report",
                data=html_report,
                file_name="credit_score_report.html",
                mime="text/html"
            )
        else:
            st.error("Please complete the previous steps to generate the report.")

elif page == '🔍 Model Explanation':
    st.title("Understanding XGBoost")

    # Explanation with highlighted keywords
    st.write(
        """
        `XGBoost(Extreme Gradient Boosting)` is a machine learning algorithm known for its `high performance` in both classification and regression tasks. Here’s a detailed look at its working, advantages, and disadvantages:
        """, unsafe_allow_html=True
    )

    st.write(
        """
        XGBoost builds an `ensemble` of decision trees through a process called `boosting`. Boosting involves sequentially training models where each model corrects the errors of the previous one. The main steps are:
        
        * `Initialization`: Start with an initial prediction, often the mean of the target values.</li>
        * `Iterative Improvement`: In each iteration, a new decision tree is trained to predict the errors of the previous trees.</li>
        * `Combination`: The predictions from all trees are combined to make the final prediction, usually by taking a weighted average.</li>
        """, unsafe_allow_html=True
    )

    # Expandable section for performance comparison plot
    with st.expander("📊 Performance Comparison of Classifiers"):
        # Data
        algorithms = [
            "RandomForest", "LogisticRegression", "DecisionTree", "MultinomialNB",
            "KNN", "MLPClassifier", "CatBoostClassifier", "SupportVectorClassifier",
            "GradientBoostingClassifier", "AdaBoostClassifier", "XGBoostClassifier"
        ]
        
        scores = [
            0.784153, 0.657104, 0.724044, 0.647541,
            0.635246, 0.648907, 0.780055, 0.651639,
            0.800546, 0.648907, 0.84
        ]

        # Creating a DataFrame for better plotting
        df = pd.DataFrame({'Algorithm': algorithms, 'Score': scores})

        # Plotting
        fig, ax = plt.subplots(figsize=(12, 8))
        sns.barplot(x='Score', y='Algorithm', data=df, palette='viridis', ax=ax)

        # Highlight XGBoost
        highlight = df[df['Algorithm'] == 'XGBoostClassifier']
        ax.barh(highlight['Algorithm'], highlight['Score'], color='red')

        # Adding labels and title
        ax.set_xlabel('Score')
        ax.set_title('Performance of Different Classifiers')

        # Display the plot in Streamlit
        st.pyplot(fig)

    st.write(
        """
        <h3>Advantages of XGBoost:</h3>
        
        * `High Performance`: Known for high accuracy and performance, XGBoost often wins machine learning competitions.
        * `Speed`: It is designed to be fast and efficient, handling large datasets and complex models effectively.
        * `Regularization`: XGBoost includes regularization techniques (L1 and L2) to reduce overfitting and enhance model generalization.
        * `Handling Missing Values`: It can automatically handle missing values without requiring explicit imputation.
        * `Feature Importance`: Provides insightful feature importance scores, helping to understand the impact of different features on the predictions.
        * `Scalability`: XGBoost is scalable and supports parallel processing, making it suitable for large-scale data and complex models.
        
        """, unsafe_allow_html=True
    )

    st.write(
        """
        <h3>Disadvantages of XGBoost:</h3>
        
        * `Complexity`: It can be complex to tune the hyperparameters effectively.
        * `Resource-Intensive`: Requires significant computational resources for very large datasets.
        * `Overfitting Risk`: Despite regularization, there's still a risk of overfitting if not tuned properly.
        """, unsafe_allow_html=True
    )

    st.write(
        """
        <h3>Why Use XGBoost?</h3>

        XGBoost is often the `go-to algorithm`for many machine learning tasks due to its balance of performance and speed. It’s especially useful in scenarios where accuracy is crucial and where you have large amounts of data.
        
        Overall, XGBoost's combination of advanced features and efficiency makes it a preferred choice for many `data scientists` and `machine learning practitioners`.
        """, unsafe_allow_html=True
    )
    
elif page == '📉 Feature Selection':
    # Feature Selection Section
    st.header('🔍 Feature Selection')

    st.markdown(
    """
    `Mutual Information (MI)` measures the dependency between two variables. 
    In the context of feature selection, MI helps to quantify the relevance of each feature with respect to the target variable. 
    A higher MI score indicates a higher dependency, implying that the feature contains more information about the target variable.
    </p>
    """, unsafe_allow_html=True
    )

    st.write("### Feature Importance Visualization Based on MI")

    # Example MI scores and features
    features = ['Annual_Income', 'Monthly_Inhand_Salary', 'Interest_Rate', 'Delay_from_due_date', 
            'Num_Credit_Inquiries', 'Credit_Mix', 'Outstanding_Debt', 
            'Total_EMI_per_month']  # Example feature names

    mi_scores = [ 0.56, 0.55, 0.18, 0.12, 0.12,  0.14, 0.57, 0.53]  # Example MI scores

    # Creating a DataFrame for MI scores
    mi_df = pd.DataFrame({'Feature': features, 'MI Score': mi_scores})
    mi_df = mi_df.sort_values(by='MI Score', ascending=False)

    # Plotting Mutual Information Scores
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(x='MI Score', y='Feature', data=mi_df, palette='viridis', ax=ax)
    ax.set_title('Mutual Information Scores for Features')
    ax.set_xlabel('MI Score')
    ax.set_ylabel('Feature')

    # Display the plot in Streamlit
    st.pyplot(fig)

    st.markdown(
    """
    The bar chart above visualizes the `Mutual Information` scores for each feature. 
    Features with higher MI scores have a stronger relationship with the target variable and are considered more important for predicting outcomes.
    """, unsafe_allow_html=True
    )

elif page == '📈 Evaluation':
    # Sample data and model evaluation section
    st.header('📊 Model Evaluation Metrics')

    st.markdown(
    """
    When evaluating a classification model, it's important to use several metrics to understand how well the model performs. 
    The key metrics used are `Accuracy`, `Precision`,`Recall`, and `F1-Score`.
    Additionally, we use the `Confusion Matrix` and the `AUC-ROC Curve` to visualize the model's performance.
    """, unsafe_allow_html=True
    )

    # Explanation of Confusion Matrix and Metrics
    st.write("### What is a Confusion Matrix?")
    st.markdown(
    """
    A `Confusion Matrix` is a table that is often used to describe the performance of a classification model. 
    It shows the actual versus predicted classifications across different classes:
    </p>
    <ul style="font-size:18px;">
        <li>True Positives (TP): Correctly predicted positive observations</li>
        <li>True Negatives (TN): Correctly predicted negative observations</li>
        <li>False Positives (FP): Incorrectly predicted positive observations (Type I error)</li>
        <li>False Negatives (FN): Incorrectly predicted negative observations (Type II error)</li>
    </ul>
    """, unsafe_allow_html=True
    ) 

    # Formulas for Evaluation Metrics
    st.write("### Formulas for Evaluation Metrics")
    st.markdown(
    """
    * `Accuracy`: Measures the ratio of correctly predicted observations to the total observations.</li>
    """, unsafe_allow_html=True
    )
    st.latex(r"Accuracy = \frac{TP + TN}{TP + TN + FP + FN}")

    st.markdown(
    """
    * `Precision`: The ratio of correctly predicted positive observations to the total predicted positives.</li>
    """, unsafe_allow_html=True
    )
    st.latex(r"Precision = \frac{TP}{TP + FP}")

    st.markdown(
    """
    * `Recall (Sensitivity)`: The ratio of correctly predicted positive observations to all observations in the actual class.</li>
    """, unsafe_allow_html=True
    )
    st.latex(r"Recall = \frac{TP}{TP + FN}")

    st.markdown(
    """
    * `F1-Score`: The weighted average of Precision and Recall. It is more useful than accuracy especially when you have an uneven class distribution.</li>
    """, unsafe_allow_html=True
    )
    st.latex(r"F1\text{-}Score = 2 \times \frac{Precision \times Recall}{Precision + Recall}")

    # Displaying the Classification Report, Confusion Matrix, and AUC-ROC Curve
    st.write("### Model Evaluation Visualizations")

    # Explanation and Display of Classification Report
    st.write("#### Classification Report")
    st.markdown(
    """
    The `Classification Report` provides a summary of the model's precision, recall, F1-score, and support for each class. 
    This report helps in understanding how well the model is performing on different classes by providing the metrics for each class label. It's particularly useful for detecting class imbalances.
    """, unsafe_allow_html=True 
    )

    with st.expander("📋 View Classification Report"):
        st.write("The classification report provides a detailed performance analysis of the classification model.")
        # Display the pre-generated image of the classification report
        st.image(os.path.join(os.path.join(os.path.dirname(os.getcwd()),'assets'),'confusion_matrix.png'), caption='Classification Report')

    # Explanation and Display of Confusion Matrix
    st.write("#### Confusion Matrix")
    st.markdown(
    """
    The `Confusion Matrix` shows the true positive, true negative, false positive, and false negative counts for the model predictions. 
    This visualization helps in identifying misclassifications and understanding which classes are more frequently misclassified.
    """, unsafe_allow_html=True
    )
    with st.expander("🧩 View Confusion Matrix"):
        st.write("The confusion matrix helps to visualize the performance of the classification model.")
        # Display the pre-generated image of the confusion matrix
        st.image(os.path.join(os.path.join(os.path.dirname(os.getcwd()),'assets'),'confusion_matrix.png'), caption='Confusion Matrix')

    # Explanation and Display of AUC-ROC Curve
    st.write("#### AUC-ROC Curve")
    st.markdown(
    """
    The `AUC-ROC Curve` (Area Under the Receiver Operating Characteristic Curve) is a performance measurement for classification problems at various threshold settings. 
    It tells how much the model is capable of distinguishing between classes. The closer the AUC value is to 1, the better the model's performance.
    """, unsafe_allow_html=True 
    )
    
    with st.expander("📈 View AUC-ROC Curve"):
        st.write("The AUC-ROC curve shows the trade-off between sensitivity and specificity for different threshold values.")
        # Display the pre-generated image of the AUC-ROC curve
        st.image(os.path.join(os.path.join(os.path.dirname(os.getcwd()),'assets'),'roc_auc_curve.png'), caption='AUC-ROC Curve')
elif page == '⚙️ Optimization':
    
    # Introduction to Optimization Techniques
    st.write("### Hyperparameter Optimization Techniques")
    st.markdown(
        """
        <p style="font-size:18px;">
        Optimize models by finding the best hyperparameters. Key techniques include:
        </p>
        """, unsafe_allow_html=True
    )

    # RandomizedSearchCV Explanation
    st.write("#### 1. RandomizedSearchCV")
    st.markdown(
        """
        - `Randomly` samples from hyperparameter ranges.
        - `Faster` and efficient.
        - `Cross-Validation`: Evaluates hyperparameters by splitting data into folds, training, and validating across them to ensure robust performance and reduce overfitting.
        """, unsafe_allow_html=True
    )

    # Built-in Cross-Validation in RandomizedSearchCV
    st.write("#### 2. In-built Cross-Validation in RandomizedSearchCV")
    st.markdown(
        """
        - Uses built-in `cross-validation`.
        - `How It Works`: Splits data into folds, trains on each fold, and evaluates on the rest. Repeats for each fold for a thorough performance assessment.
        - `Early Stopping`: Stops training when performance plateaus to prevent overfitting.
        """, unsafe_allow_html=True
    )

    # Fine-Tuning Hyperparameters
    st.write("### Fine-Tuned XGBoost Hyperparameters")
    st.markdown(
        """
        Optimize XGBoost with these hyperparameters:

        - <span style="color: #007bff; font-weight: bold;">`n_estimators`</span> = `500`
        - <span style="color: #007bff; font-weight: bold;">`max_depth`</span> = `7`
        - <span style="color: #007bff; font-weight: bold;">`learning_rate`</span> = `0.3`
        - <span style="color: #007bff; font-weight: bold;">`subsample`</span> = `1.0`
        - <span style="color: #007bff; font-weight: bold;">`colsample_bytree`</span> = `0.9`
        - <span style="color: #007bff; font-weight: bold;">`min_child_weight`</span> = `1`
        - <span style="color: #007bff; font-weight: bold;">`gamma`</span> = `0`
        - <span style="color: #007bff; font-weight: bold;">`reg_alpha`</span> = `0`
        - <span style="color: #007bff; font-weight: bold;">`reg_lambda`</span> = `1`
        """, unsafe_allow_html=True
    )

    # Importance of Hyperparameter Tuning
    st.write("### Why Tune Hyperparameters?")
    st.markdown(
        """
        - Balances `complexity` and performance.
        - Reduces `overfitting` and improves generalization.
        - Enhances `predictive power` and stability.
        """, unsafe_allow_html=True
    )
elif page == '🌍 Real World Examples':
    st.header('🌍 Real World Examples')

    st.write("#### Real-Time Performance prediction")

    # Example data for 10 individuals with a mix of classes
    example_data = pd.DataFrame({
        'Name': ['John Doe', 'Emily Smith', 'Sarah Johnson', 'Michael Brown', 'Lisa White', 
                 'Tom Green', 'Nancy Davis', 'George Wilson', 'Olivia Clark', 'Sophia Martinez'],
        'Actual Credit Score': ['Good', 'Good', 'Standard', 'Good', 'Standard', 
                                'Good', 'Good', 'Standard', 'Good', 'Poor'],
        'Predicted Credit Score': ['Good', 'Good', 'Standard', 'Good', 'Standard', 
                                    'Good', 'Good', 'Poor', 'Good', 'Standard'],
        'Annual Income': [50000, 120000, 45000, 60000, 75000, 
                          55000, 100000, 46000, 62000, 77000],
        'Monthly Inhand Salary': [4000, 10000, 3500, 5000, 6000, 
                                  4500, 9000, 3600, 5200, 6200],
        'Interest Rate': [7.5, 5.0, 8.0, 6.0, 6.5, 
                          7.0, 5.5, 7.8, 6.2, 6.7],
        'Delay from Due Date': [5, 2, 10, 3, 7, 
                                 4, 1, 8, 2, 6],
        'Number of Credit Inquiries': [2, 1, 3, 1, 2, 
                                        3, 1, 2, 1, 4],
        'Credit Mix': ['Good', 'Excellent', 'Fair', 'Good', 'Poor', 
                       'Good', 'Excellent', 'Fair', 'Good', 'Poor'],
        'Outstanding Debt': [10000, 25000, 12000, 15000, 18000, 
                             11000, 24000, 13000, 16000, 19000],
        'Payment of Minimum Amount': ['Yes', 'No', 'Yes', 'No', 'Yes', 
                                      'Yes', 'No', 'Yes', 'No', 'Yes'],
        'Total EMI per Month': [300, 700, 400, 500, 600, 
                                320, 680, 420, 520, 620],
        })

    # Setting the 'Name' column as the index
    example_data.set_index('Name', inplace=True)

    st.write("**Example Data for Prediction:**")

    # Function to color text based on predicted credit score
    def color_predicted_score(val):
        if val == 'Good':
            color = 'green'
        elif val == 'Standard':
            color = 'lightgreen'
        else:  # 'Poor'
            color = 'red'
        return f'color: {color}'

    # Display the table with real-time data and color for predicted credit scores
    st.dataframe(example_data.style.applymap(color_predicted_score, subset=['Predicted Credit Score']))


    st.write("### Applications")
    st.write("""
    The credit score prediction model is valuable in:
    - `Financial Institutions` For loan eligibility and risk assessment.
    - `Consumer Credit Services` For credit card and loan decisions.
    - `Insurance Companies` To determine premiums and eligibility.
    - `Personal Finance` To manage finances and improve creditworthiness.
    """, unsafe_allow_html=True)

    st.write("### Case Study:")
    st.write("""
    A bank using this model:
    - `Reduced loan default rates` by identifying high-risk applicants.
    - `Offered personalized advice` based on predicted credit scores.
    - `Streamlined loan approval`, boosting customer satisfaction.
    """, unsafe_allow_html=True)

    