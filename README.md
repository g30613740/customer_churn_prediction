<h1>Customer Churn Prediction</h1>

<p>An end‑to‑end machine learning project that predicts whether a telecom customer will cancel their subscription. The project covers data exploration, preprocessing, feature engineering, model training, evaluation, interpretation, and deployment via an interactive web application.</p>

<p><strong>Key result:</strong> Logistic Regression achieves 81% recall (catches 8 out of 10 churners) and an ROC‑AUC of 0.85.</p>

<hr/>

<h2>Quick Start</h2>

<pre><code>
# 1. Clone the repository
git clone https://github.com/your-username/customer-churn-predictor.git
cd customer-churn-predictor

# 2. Create a virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate   # On Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run the Streamlit app
streamlit run app.py
</code></pre>

<p>Open your browser at <code>http://localhost:8501</code> and start predicting churn in real time.</p>

<hr/>

<h2>Table of Contents</h2>

<ul>
  <li><a href="#project-overview">Project Overview</a></li>
  <li><a href="#dataset">Dataset</a></li>
  <li><a href="#project-structure">Project Structure</a></li>
  <li><a href="#installation">Installation</a></li>
  <li><a href="#usage">Usage</a></li>
  <li><a href="#pipeline">Pipeline</a></li>
  <li><a href="#model-performance">Model Performance</a></li>
  <li><a href="#feature-importance">Feature Importance</a></li>
  <li><a href="#business-insights">Business Insights</a></li>
  <li><a href="#deployment">Deployment</a></li>
  <li><a href="#future-work">Future Work</a></li>
  <li><a href="#license">License</a></li>
  <li><a href="#acknowledgements">Acknowledgements</a></li>
</ul>

<hr/>

<h2 id="project-overview">Project Overview</h2>

<p>Customer churn is a critical metric for subscription‑based businesses. This project demonstrates a complete data science workflow:</p>

<ul>
  <li><strong>Business understanding:</strong> Identify customers who are likely to leave and understand why.</li>
  <li><strong>Data preparation:</strong> Clean and transform the Telco Customer Churn dataset.</li>
  <li><strong>Feature engineering:</strong> Create new features that capture customer behaviour and commitment.</li>
  <li><strong>Model selection:</strong> Compare Logistic Regression, Random Forest, and XGBoost.</li>
  <li><strong>Evaluation:</strong> Focus on recall (catching churners) and interpretability.</li>
  <li><strong>Deployment:</strong> Package the best model into a user‑friendly web application.</li>
</ul>

<p>The final model is deployed as a Streamlit app, allowing anyone to input customer details and receive an instant churn probability.</p>

<hr/>

<h2 id="dataset">Dataset</h2>

<p>The <a href="https://www.kaggle.com/datasets/blastchar/telco-customer-churn">Telco Customer Churn</a> dataset from Kaggle contains 7,043 customer records with 21 features, including:</p>

<ul>
  <li><strong>Demographics:</strong> gender, SeniorCitizen, Partner, Dependents</li>
  <li><strong>Account information:</strong> tenure, Contract, PaperlessBilling, PaymentMethod, MonthlyCharges, TotalCharges</li>
  <li><strong>Services subscribed:</strong> PhoneService, InternetService, OnlineSecurity, TechSupport, StreamingTV, etc.</li>
  <li><strong>Target:</strong> Churn (Yes/No) – 26.5% churn rate, making it an imbalanced classification problem.</li>
</ul>

<hr/>

<h2 id="project-structure">Project Structure</h2>

<pre>
customer-churn-predictor/
├── main.ipynb                 # Full Jupyter notebook with all steps
├── app.py                     # Streamlit application
├── churn_model.pkl            # Saved Logistic Regression model
├── scaler.pkl                 # Fitted StandardScaler
├── training_columns.pkl       # Column names for alignment
├── requirements.txt           # Python dependencies
├── LICENSE                    # MIT License
└── README.md                  # Project documentation
</pre>

<hr/>

<h2 id="installation">Installation</h2>

<h3>Prerequisites</h3>
<ul>
  <li>Python 3.8 or higher</li>
  <li>pip</li>
</ul>

<h3>Steps</h3>
<ol>
  <li><strong>Clone the repository:</strong>
    <pre><code>git clone https://github.com/your-username/customer-churn-predictor.git
cd customer-churn-predictor</code></pre>
  </li>
  <li><strong>Create a virtual environment (recommended):</strong>
    <pre><code>python -m venv venv
source venv/bin/activate      # On Windows: venv\Scripts\activate</code></pre>
  </li>
  <li><strong>Install dependencies:</strong>
    <pre><code>pip install -r requirements.txt</code></pre>
  </li>
</ol>

<p>The <code>requirements.txt</code> file includes:</p>
<ul>
  <li>pandas, numpy – data manipulation</li>
  <li>matplotlib, seaborn – visualisation</li>
  <li>scikit‑learn – preprocessing, models, metrics</li>
  <li>imbalanced‑learn – SMOTE for oversampling</li>
  <li>xgboost – gradient boosting</li>
  <li>streamlit – web interface</li>
  <li>joblib – model serialisation</li>
</ul>

<hr/>

<h2 id="usage">Usage</h2>

<h3>Run the Jupyter Notebook</h3>
<p>Launch Jupyter and open <code>main.ipynb</code> to explore the full pipeline step by step:</p>
<pre><code>jupyter notebook main.ipynb</code></pre>

<h3>Run the Streamlit App</h3>
<p>Start the interactive predictor:</p>
<pre><code>streamlit run app.py</code></pre>
<p>Then open <code>http://localhost:8501</code> in your browser.</p>

<h3>Use the Saved Model Programmatically</h3>
<pre><code>import joblib
import pandas as pd

model = joblib.load('churn_model.pkl')
scaler = joblib.load('scaler.pkl')
columns = joblib.load('training_columns.pkl')

# Preprocess your new data using the same pipeline,
# then call model.predict_proba(X_scaled)[:, 1]
</code></pre>

<hr/>

<h2 id="pipeline">Pipeline</h2>

<p>The project follows a structured pipeline, which is fully documented in the notebook:</p>

<ol>
  <li><strong>Data Acquisition & Initial Exploration</strong> – Load data, inspect shape, data types, and missing values.</li>
  <li><strong>Exploratory Data Analysis (EDA)</strong> – Visualise distributions, churn by contract type, tenure, monthly charges, payment method, and internet service. Compute correlation heatmap.</li>
  <li><strong>Data Preprocessing</strong> – Convert <code>TotalCharges</code> to numeric and fill missing values with 0; drop <code>customerID</code>; one‑hot encode categorical features; scale numerical features using <code>StandardScaler</code>.</li>
  <li><strong>Feature Engineering</strong> – Add new features:
    <ul>
      <li><code>Avg_Monthly_Spend</code> – total charges divided by tenure.</li>
      <li><code>Tenure_Group</code> – bin tenure into categories (0‑6m, 7‑12m, 1‑2y, 2y+).</li>
      <li><code>Service_Count</code> – number of subscribed services.</li>
      <li><code>Contract_Tenure_Ratio</code> – tenure divided by contract length in months.</li>
    </ul>
  </li>
  <li><strong>Train‑Test Split</strong> – 80/20 split with stratification to preserve churn rate.</li>
  <li><strong>Handling Imbalance</strong> – For tree models, use SMOTE to create a balanced training set; for Logistic Regression, use class weights.</li>
  <li><strong>Model Building</strong> – Train Logistic Regression, Random Forest, and XGBoost. Logistic Regression uses class weights, while the others are trained on SMOTE‑balanced data.</li>
  <li><strong>Evaluation</strong> – Compare models using accuracy, precision, recall, F1, and ROC‑AUC on the untouched test set. Confusion matrices and classification reports are provided.</li>
  <li><strong>Interpretation</strong> – Extract coefficients from Logistic Regression and feature importances from Random Forest to identify key drivers.</li>
  <li><strong>Deployment</strong> – Save the best model and build a Streamlit app that replicates the preprocessing pipeline and delivers real‑time predictions.</li>
</ol>

<hr/>

<h2 id="model-performance">Model Performance</h2>

<p>All models were evaluated on the same test set (1,409 customers) with the original class distribution (26.5% churn).</p>

<table>
  <tr>
    <th>Model</th>
    <th>Accuracy</th>
    <th>Precision (Churn)</th>
    <th>Recall (Churn)</th>
    <th>F1 (Churn)</th>
    <th>ROC‑AUC</th>
  </tr>
  <tr>
    <td>Logistic Regression</td>
    <td>0.746</td>
    <td>0.514</td>
    <td><strong>0.810</strong></td>
    <td>0.629</td>
    <td><strong>0.848</strong></td>
  </tr>
  <tr>
    <td>Random Forest</td>
    <td>0.766</td>
    <td>0.555</td>
    <td>0.596</td>
    <td>0.575</td>
    <td>0.823</td>
  </tr>
  <tr>
    <td>XGBoost</td>
    <td>0.754</td>
    <td>0.535</td>
    <td>0.554</td>
    <td>0.544</td>
    <td>0.810</td>
  </tr>
</table>

<p>Logistic Regression was chosen as the final model because it offers the highest recall (catching 81% of actual churners) and the best ROC‑AUC, while also being the most interpretable.</p>

<hr/>

<h2 id="feature-importance">Feature Importance</h2>

<p>The table below shows the top factors that increase or decrease churn probability, based on Logistic Regression coefficients (after standardisation).</p>

<h3>Top 5 Features Increasing Churn</h3>
<ol>
  <li><strong>InternetService_Fiber optic</strong> – Fiber customers are more likely to churn.</li>
  <li><strong>StreamingMovies_Yes</strong> – Customers with streaming movies tend to churn more.</li>
  <li><strong>StreamingTV_Yes</strong> – Similar to streaming movies.</li>
  <li><strong>PaymentMethod_Electronic check</strong> – Manual payments correlate with higher churn.</li>
  <li><strong>PaperlessBilling_Yes</strong> – Often associated with electronic checks.</li>
</ol>

<h3>Top 5 Features Decreasing Churn</h3>
<ol>
  <li><strong>Contract_Two year</strong> – Long contracts dramatically reduce churn.</li>
  <li><strong>Contract_One year</strong> – Annual contracts also lower churn.</li>
  <li><strong>MonthlyCharges</strong> – Higher monthly charges (often tied to longer contracts) reduce churn.</li>
  <li><strong>tenure</strong> – Loyal customers stay longer.</li>
  <li><strong>OnlineSecurity_Yes</strong> – Security services increase stickiness.</li>
</ol>

<p>Random Forest feature importances confirm these findings, with <code>tenure</code>, <code>TotalCharges</code>, and <code>Contract</code>‑related features ranking highest.</p>

<hr/>

<h2 id="business-insights">Business Insights</h2>

<p>Based on the model interpretation, the following actions are recommended:</p>

<ul>
  <li><strong>Convert month‑to‑month customers</strong> to annual contracts – this is the single most impactful action.</li>
  <li><strong>Promote automatic payment methods</strong> (e.g., bank transfer, credit card) to reduce churn.</li>
  <li><strong>Improve fibre optic service quality</strong> or offer bundled packages to retain fibre customers.</li>
  <li><strong>Focus retention efforts on new customers</strong> – offer onboarding support and welcome incentives.</li>
  <li><strong>Cross‑sell security and tech support services</strong> – they increase switching costs.</li>
</ul>

<hr/>

<h2 id="deployment">Deployment</h2>

<p>The Streamlit app (<code>app.py</code>) provides a clean interface for non‑technical users to:</p>
<ul>
  <li>Enter customer details using sliders, dropdowns, and number inputs.</li>
  <li>Receive an instant churn probability and a clear prediction.</li>
  <li>View a progress bar and a warning/success message for quick interpretation.</li>
</ul>

<p>The app replicates the entire preprocessing and feature engineering pipeline, ensuring predictions are consistent with the training data. The model, scaler, and training columns are serialised with <code>joblib</code> for fast loading.</p>

<p>To run the app, simply execute:</p>
<pre><code>streamlit run app.py</code></pre>

<hr/>

<h2 id="future-work">Future Work</h2>

<p>Potential extensions to this project:</p>
<ul>
  <li><strong>Hyperparameter tuning</strong> – Use GridSearchCV to further improve Random Forest and XGBoost.</li>
  <li><strong>SHAP explanations</strong> – Add SHAP values to explain individual predictions and build trust.</li>
  <li><strong>Online deployment</strong> – Host the app on Streamlit Community Cloud, Heroku, or AWS.</li>
  <li><strong>Database integration</strong> – Store predictions and feedback for continuous improvement.</li>
  <li><strong>Retraining pipeline</strong> – Set up automated retraining with new data.</li>
</ul>

<hr/>

<h2 id="license">License</h2>

<p>This project is licensed under the MIT License – see the <code>LICENSE</code> file for details.</p>

<hr/>

<h2 id="acknowledgements">Acknowledgements</h2>

<ul>
  <li>Dataset: <a href="https://www.kaggle.com/datasets/blastchar/telco-customer-churn">Telco Customer Churn</a> by blastchar on Kaggle.</li>
  <li>Built with Python, scikit‑learn, XGBoost, and Streamlit.</li>
  <li>Developed as a portfolio project to demonstrate data science and machine learning skills.</li>
</ul>

<hr/>

<p><strong>Author:</strong> @g30613740 Philip K.</p>
<p><strong>GitHub:</strong> <a href="https://github.com/your-username/customer-churn-predictor">https://github.com/g30613740/customer-churn-predictor</a></p>