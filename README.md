# Telecom Customer Churn Prediction

## Problem Definition, Current State, Objectives and Benefits
The core objective is predicting customer churn behavior for the TelCo Company. TelCo Company currently has no data-driven and quantitatively-based prediction model. Our team will develop machine learning models, using a 7043-observation (customer) with 20 predictors dataset. The model will be a supervised, binary classification model, since we have a clear binary target variable, churn or no churn. To achieve the optimal outcome, our team will both manually build machine learning models in Python and develop an Auto Machine Learning Model in H20.ai. The core performance will be measured in terms of the Recall Score. Because the underlying assumption is that TeleCo wants to minimize customer churn rate. To do so, the best model used will be the one that maximized the recall score. 

After the optimal model was chosen and pickled, it will be run offline periodically (at most once a day). In other words, the prediction model will build on a batch processing system using Apache Spark, which takes a large amount of input data, runs the pickled prediction model to process it, and produces the prediction outcome. The minimum performance needed for the model is that the recall score is higher than 0.5. Another assumption is that TeleCo has millions of consumers. Therefore, even a marginal improvement from 0.5, such as 0.01 increase, will provide a significant increase in customer retention, revenue, and net operating income. 

Our model will be used by the marketing and consumer retention team so that they could intervene early by either offering targeted online offers and coupons, or having representatives phoning those consumers who are predicted to churn. Also, the model explainability report will be used by middle-level managers to gain insights on the factor that both positively and negatively contribute to the churn rate so they could address such factors accordingly. 

## Project Core Objectives/Hypothesis
### Prediction Task:
Predicting (i.e., classifying) wheather a current customer will churn from TelCo or not.
    
### Core Tasks/Actions:
1. Trained mutiple Python-based Machine Learning models to predict customer churn for the TelCo Company.
2. Programmed a H20.ai-based Auto Machine Learning model and compared the AutoML model performance (in terms of recall and accuracy) againts manually trained models.
3. Tested casual inference on following predictors: gender, SeniorCitizen, Partner, Dependents, PhoneService, and PaperlessBilling by applying Microsoft DoWhy pakacage realized in Python.             
4. Assembled a model interpretability and explainability analysis using SHAP package on XGBoost Classification Model.
5. Composed a Machine Learning Bias Report based on the SHAR analysis to suggest further managerial actions.
6. Lessons learned and next steps

### Null Hypothesis and Types of Errors:
H<sub>0</sub>(Null Hypothesis): The recalll score is less or equal to 0.5 and no predicting power can be generated from the model. 

H<sub>a</sub>(Alternative Hypothesis): The recalll score is greater than 0.5 and some predicting power can be generated from the model.

Types of Erros: Both Type I and Type II errors are expected to be made by the model. The goal of the model is to minimize Type II errors (i.e., reducing false negative rate.)

## Data Description Report - someone?
Document and present leveraged data sources used to create the dataset
• Cover a brief resume of 5.2. Data Acquisition, 5.3. Data Exploration, and 5.4.
Data Preparation
• Profile and present the data before and after going through acquisition, exploration, and preparation.
5.2. Data Acquisition
• List the data you need and how much you need.
• Find and document where you can get that data.
• Check how much space it will take.
• Check legal obligations, and get authorization if necessary.
• Get access authorizations.
• Create a workspace (with enough storage space).
• Get the data.
• Convert the data to a format you can easily manipulate (without changing the data itself).
• Ensure sensitive information is deleted or protected (e.g. anonymized).
• Check the size and type of the data (time series, sample, geographical, etc.)
• Sample a test set, put it aside, and never look at it (no data snooping!).
5.3. Data Exploration
• Create a copy of the data for exploration (sampling it down to a manageable size if necessary).
• Create a Jupyter notebook to keep a record of your data exploration.
• Study each attribute and its characteristics:
• Name
• Type (categorical, int/float, bounded/unbounded, text, structured, etc.)
• % of missing values
• Noisiness and type of noise (stochastic, outliers, rounding errors, etc.)
• Possibly useful for the task?
• Type of distribution (Gaussian, uniform, logarithmic, etc.)
• For supervised learning tasks, identify the target attribute(s).
• Visualize the data.
• Study the correlations between attributes.
• Study how you would solve the problem manually.
• Identify the promising transformations you may want to apply.
• Identify extra data that would be useful (go back to “Get the Data”).
• Document what you have learned.
5.4. Data Preparation
1. Dealing with missing data 2. Cleaning data
3. Data preprocessing
4. Feature subset selection 5. Feature engineering
6. Feature scaling 7. Clustering
refer here: https://www.kaggle.com/blastchar/telco-customer-churn

## Churn Prediction Modeling
### Python-based Model -dev

Model Work Flow:
1. get data and descriptive stats
2. data visualization 
3. train a model:
    1) Regression;
    2) ANN;
    3) Random Forest;
    ...
4. select model and further tune
5. Extra
    1) building a pipeline
    ...... 
### H20 Auto-ML Model -everlyn?

## Model Results 
### Results Summary Report -- everlyn?

### Casual Inference Report --Charlie?

### Churn Model Interpretability and Explainability Report
To access the XGBoost Classification Model interpretability and explainability, we used the SHAR package to visualize the predictors' effect on the target variable, churn. The reason we choose the XGBoost Classification Model to analyze instead of the Random Forest Model because the Random Forest Model takes significantly longer compared to the XGBoost Model, and our team's laptops are unable to provide the results. 

We first can to visualize the first prediction's explanation:

<img src = "Model-Interpretability-Graph/Interpretability1.png" width = 700>

Next, we summarize the effects of all the features:

<img src = "Model-Interpretability-Graph/Interpretability2.png" height = 400>

We can also just take the mean absolute value of the SHAP values for each feature to get a standard bar plot (produces stacked bars for multi-class outputs):

<img src = "Model-Interpretability-Graph/Interpretability3.png" height = 400>

## Threats to validity - someone?
### Uncertainties and Risks
### Data quality issues

## Conclusions -- Jiajun
### Overall Overview
### Model Bias Report 
### Next Step
