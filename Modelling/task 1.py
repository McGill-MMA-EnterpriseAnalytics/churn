import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    auc,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
    accuracy_score,
    precision_score,
    average_precision_score,
    recall_score,
    f1_score,
    precision_recall_curve,
    cohen_kappa_score,
    confusion_matrix,
    classification_report
)
from sklearn.model_selection import (
    train_test_split,
    StratifiedKFold,
    cross_val_score,
    cross_val_predict
)
import seaborn as sns
from sklearn import (
    ensemble,
    model_selection,
    preprocessing,
    tree
)

df = pd.read_csv(r"C:\Users\dkhurm\Documents\Classwork\Enterprise Analytics\WA_Fn-UseC_-Telco-Customer-Churn.csv")

df.dtypes


#import pandas_profiling
#pandas_profiling.ProfileReport(df,  )

df.shape

df.columns

df.describe()

def missing_values(df):
    mis_val = df.isnull().sum()
    mis_val_percent = 100 * df.isnull().sum() / len(df)
    mis_val_table = pd.concat([mis_val, mis_val_percent], axis=1)
    mis_val_table_ren_columns = mis_val_table.rename(columns={
        0: 'Missing Values',
        1: '% of Total Values'
    })
    mis_val_table_ren_columns = mis_val_table_ren_columns[
        mis_val_table_ren_columns.iloc[:, 1] != 0].sort_values(
            '% of Total Values', ascending=False).round(1)
    print("Dataframe has " + str(df.shape[1]) + " columns.")
    print("There are " + str(mis_val_table_ren_columns.shape[0]) +
          " columns that have missing values.")
    
    return mis_val_table_ren_columns

# Missing values statistics
miss_values = missing_values(df)
miss_values.head(20)

import missingno as msno
ax = msno.matrix(df.sample())

ax = msno.bar(df)

df['Churn'].unique()

target = [ 1 if i == 'Yes' else 0 for i in df['Churn']]

df['target'] = target
df['target'].value_counts()

df.drop('Churn', axis=1, inplace=True)

corr = df.corr()['target'].sort_values()

# Display correlations
print('Top 10 - Positive Correlations:')
print('-----------------------------------')
print(corr.tail(10))
print('\nTop 10 - Negative Correlations:')
print('------------------------------')
print(corr.head(10))

correlation = df.corr()
plt.figure(figsize=(16, 16))
ax = sns.heatmap(
    correlation, 
    vmin=-1, vmax=1, center=0,
    cmap=sns.diverging_palette(20, 220, n=200),
    square=True
)
ax.set_xticklabels(
    ax.get_xticklabels(),
    rotation=45,
    horizontalalignment='right'
)

 
# Number of each type of column
df.dtypes.value_counts().sort_values().plot(kind='barh',
                                            figsize=(22, 6),
                                            fontsize=16)

plt.title('Number of columns by data types', fontsize=18)
plt.xlabel('Number of columns', fontsize=16)
plt.ylabel('Data type', fontsize=16)

df.select_dtypes('object').apply(pd.Series.nunique, axis=0)

#########################################################3

df.drop(['customerID'],
        axis=1,
        inplace=True)

categorical = ['gender', 'SeniorCitizen', 'Partner', 'Dependents', 'tenure',
       'PhoneService', 'MultipleLines', 'InternetService', 'OnlineSecurity',
       'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV',
       'StreamingMovies', 'Contract', 'PaperlessBilling', 'PaymentMethod']
df = pd.get_dummies(df, columns = categorical)

##########################################################
for i in range(len(df)):
    #print(i,df.TotalCharges.iloc[i])
    if df.TotalCharges.iloc[i] != ' ':
         df.TotalCharges.iloc[i] = float(df.TotalCharges.iloc[i])
    else:
        df.TotalCharges.iloc[i] = 0
#     print(df.TotalCharges.iloc[i])

for i in range(len(df)):
    if type(X.TotalCharges[i]) != float:
        print(i, type(X.TotalCharges[i]))
        X.TotalCharges[i] = float(X.TotalCharges[i])
df = df.astype({'TotalCharges': 'float'})

y = df.target
X = df.drop(['target'], axis = 1)

# ++++++++++++++++++++++++++++++++++++++++++
# CHECKING FOR CORRELATION AND REMOVING IT                      
# ++++++++++++++++++++++++++++++++++++++++++
correlated_list = []
correlation_matrix = X.corr(method ='pearson')
for i in range(77):
    for j in range(i,77):
        if correlation_matrix.iloc[i,j]>0.5 and i!=j:
            #print(correlation_matrix.iloc[i,j])
            #print(correlation_matrix.columns[i],correlation_matrix.columns[j])
            if correlation_matrix.columns[i] not in correlated_list:
                correlated_list.append(correlation_matrix.columns[i])
                X = X.drop(correlation_matrix.columns[i], axis = 1)
pd.concat([X,y], axis = 1).to_csv(r"C:/Users/dkhurm/Downloads/nocorr_clf.csv")
# ____________________________________________________________________________________
# random forest feature classification
# ____________________________________________________________________________________

final_list = []
for i in X.columns:
    final_list.append(i)
# run feature selection using random forest
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
randomforest = RandomForestClassifier(random_state=7)
model = randomforest.fit(X,y)
from sklearn.feature_selection import SelectFromModel
threshold = 0.01
sfm = SelectFromModel(model, threshold = threshold)    
sfm.fit(X,y)
#for feature_list_index in sfm.get_support(indices = True):
    #print(X.columns[feature_list_index])
selected_featuress = pd.DataFrame(list(zip(X.columns,model.feature_importances_)), columns = ['predictor','Gini coefficient'])
selected_featuress = selected_featuress.sort_values(by = 'Gini coefficient', ascending = False)
selected_featuress.to_csv("image_features_ranked_clf.csv")

features_sorted_string = ""
for row in selected_featuress.values:
    features_sorted_string = features_sorted_string+""+str(row[0])+","
features_sorted_string = features_sorted_string[:-1]
features_sorted_arr = features_sorted_string.split(',')
filtered_features = selected_featuress[selected_featuress['Gini coefficient'] >= threshold]
filtered_features_sorted_string = ""
for row in filtered_features.values:
    filtered_features_sorted_string = filtered_features_sorted_string+""+str(row[0])+","
filtered_features_sorted_string = filtered_features_sorted_string[:-1]
filtered_features_sorted_arr = filtered_features_sorted_string.split(',')
# Create X with columns in decreasing order of their significance
X_filtered = X[filtered_features_sorted_arr]
X = X_filtered

pd.concat([X,y], axis = 1).to_csv(r"C:/Users/dkhurm/Downloads/rffs_clf.csv")

X_train, X_test, y_train, y_test = model_selection.train_test_split(
    X, y, test_size=0.3, random_state=42)


#  =============================================================================
#  Model Building
#  =============================================================================
#  1. K-NN 
#  =============================================================================
  from sklearn.neighbors import KNeighborsClassifier
  from sklearn.preprocessing import StandardScaler
  standardizer = StandardScaler()

  X_std = standardizer.fit_transform(X)
 
  # Separate the data
  from sklearn.model_selection import train_test_split
  X_train, X_test, y_train, y_test = train_test_split(X_std, y, test_size = 0.33, random_state = 5)
  
  # Run K-NN
  knn = KNeighborsClassifier(n_neighbors = 7)
  model = knn.fit(X_train, y_train)


  # Using the model to predict the results based on the test dataset
  y_test_pred = knn.predict(X_test)
  
  # Calculate the mean squared error of the prediction
  from sklearn.metrics import accuracy_score
  acc = accuracy_score(y_test, y_test_pred)
  print(acc) 
  # both 7 0.7711827956989248

  # =============================================================================
  # 2. Decision Tree 
  # =============================================================================
  from sklearn.tree import DecisionTreeClassifier
  from sklearn.model_selection import train_test_split
  X_train, X_test, y_train, y_test = train_test_split(X_std, y, test_size = 0.33, random_state = 5)
  
  # Run decision tree
  dtree = DecisionTreeClassifier(max_depth = 5)
  model = dtree.fit(X_train, y_train)
  
  # Using the model to predict the results based on the test dataset
  y_test_pred = dtree.predict(X_test)
  
  # Calculate the mean squared error of the prediction
  from sklearn.metrics import accuracy_score
  acc = accuracy_score(y_test, y_test_pred)
  print(acc) 
  #  0.7961290322580645

#  =============================================================================
#  3. Random Forest 
#  =============================================================================

  from sklearn.ensemble import RandomForestClassifier
  from sklearn.model_selection import train_test_split
  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.33, random_state = 5)
  
  # Run random forest
  rf = RandomForestClassifier(random_state = 0, n_estimators = 1000)
  model = rf.fit(X_train, y_train)
  
  
  # Using the model to predict the results based on the test dataset
  y_test_pred = rf.predict(X_test)
  
  # Calculate the mean squared error of the prediction
  from sklearn.metrics import accuracy_score
  acc = accuracy_score(y_test, y_test_pred)
  print(acc) 
  #  0.7931182795698924
 
 ################ GRID_SEARCH_CV ###########################################
 from sklearn.model_selection import GridSearchCV
 import numpy as np
 
 

base_model = RandomForestClassifier(n_estimators = 100, random_state = 0)
base_model.fit(X_train, y_train)
base_acc = accuracy_score(y_test, base_model.predict(X_test))

# Create the parameter grid based on the results of random search 
param_grid = {
    'bootstrap': [True],
    'max_depth': [80, 90, 100, 110],
    'max_features': [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1],
    'min_samples_leaf': [3, 4, 5],
    'min_samples_split': [8, 10, 12],
    'n_estimators': [100, 200, 300, 500, 700, 1000]
}
# Create a based model
rf = RandomForestClassifier()
# Instantiate the grid search model
grid_search = GridSearchCV(estimator = rf, param_grid = param_grid, 
                          cv = 3, n_jobs = -1, verbose = 2)
# Fit the grid search to the data
grid_search.fit(X, y)
grid_search.best_params_
#
#
best_grid = grid_search.best_estimator_
grid_acc = accuracy_score(y_test, best_grid.predict(X_test))
 
print('Improvement of {:0.2f}%.'.format( 100 * (grid_acc - base_acc) / base_acc))

#_____________________________________________________________________________

#
# 
  # =============================================================================
  # 4. Logistic
  # =============================================================================
  from sklearn.linear_model import LogisticRegression
  from sklearn.model_selection import train_test_split
  X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.33, random_state = 5)
  #random_state is like setting a random seed
  
  # Run linear regression
  lm3 = LogisticRegression()
  model3 = lm3.fit(X_train, y_train)
  
  # Using the model to predict the results based on the test dataset
  y_test_pred = lm3.predict(X_test)
  
  # Calculate the mean squared error of the prediction
  acc = accuracy_score(y_test,y_test_pred)
  print(acc) #0.8008602150537635

 # # =============================================================================
 # # 5. XG Boost
 # # =============================================================================
 
import xgboost as xgb

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.33, random_state = 5)

xgc = xgb.XGBClassifier()

xgc.fit(X_train,y_train)

preds = xgc.predict(X_test)

acc = (accuracy_score(y_test, preds))
print(acc)
# 0.8034408602150538

                
#  =============================================================================

