#!/usr/bin/env python
# coding: utf-8

# ## Covid-19 Zgierz Swiss-Polish ML Project  
# ### Positive Patients survival analysis with values at the admission to the hospital as features  
# The data for the project was obtained from a hospital in Zgierz, Poland during the first Covid-19 wave (March - June 2020).  
# The data include information on 515 subjects that are either SARS-CoV2 positive (n = 201), or negative (n = 314) with a Covid-19 like clinical manifestations.  
# The positive group is further subdivided into survived (n = 126) and deceased (n = 72) subjects.
# The dataset includes various variables:
# - laboratory test (26 features);
# - comorbidities (17 features);
#   
#   
# The analysis that is shown in this notebook concerns the survival of positive group with features at the baseline.  
# __Publication__: *Machine Learning Successfully Detects COVID-19 Patients Prior to PCR Results and Predicts Their Survival Based on Standard Laboratory Parameters*
# __Author of the notebook__: Damir Zhakparov, PhD Candidate, Swiss Institute of Allergy and Asthma Research  
# __Date__: September 2020 - June 2021

# The project consist of following parts:  
# 1. Preprocessing of the raw data files  
# 2. EDA
# 2. Analysis pipeline  
# 3. Feature Importance analysis  
# 4. (Extra) Performance of the best performer on different train/test splits  

# In[26]:


import warnings
import os
from datetime import date
warnings.filterwarnings("ignore", category=DeprecationWarning)

import pandas as pd 
from pandas import ExcelWriter 
import numpy as np
import plotly.graph_objs as go
from plotly.subplots import make_subplots
import missingno as msno

from sklearn.experimental import enable_iterative_imputer
from sksurv.datasets import get_x_y
from sklearn.preprocessing import RobustScaler 
from sklearn.impute import SimpleImputer

from sklearn.impute import IterativeImputer
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import BayesianRidge 
from sksurv.ensemble import RandomSurvivalForest
from sksurv.ensemble import GradientBoostingSurvivalAnalysis
from sksurv.linear_model import CoxPHSurvivalAnalysis
from sklearn.pipeline import Pipeline
from sklearn.utils import estimator_html_repr 
from sklearn.compose import ColumnTransformer
from sklearn import set_config

import eli5
from eli5.sklearn import PermutationImportance

set_config(display='diagram')


# In[4]:


#### import custom scripts 
import sys
sys.path.insert(0, '~/baseline-survival-analysis')
  
from estimator_helper import EstimatorSelectionHelper 
from preprocessing import CovidPreprocessor


# In[4]:


#os.system('jupyter nbconvert --to html final_analysis_draft.html')


# In[5]:


#### read the files ####
# laboratory values of the patients 
lab = pd.read_csv(os.path.join(r'~/Documents/Projects/Covid19_Poland/raw-data/','cov2_pos_lab.csv'),
                 nrows = 201)

# appendix containing additional laboratory measurements 
appdx = pd.read_csv(os.path.join(r'~/Documents/Projects/Covid19_Poland/raw-data/','cov2_appendix.csv'),
                   nrows = 201)

# information on comorbidites and premedications 
data  = pd.read_csv(os.path.join(r'~/Documents/Projects/Covid19_Poland/raw-data/','cov2_pos.csv'),
                    nrows = 201)

# new categories that come from collapsed smaller categories 
cats = pd.read_excel(os.path.join(r'~/Documents/Projects/Covid19_Poland/raw-data/','covidtable_categories_pat_MS_kb.xlsx'),
                     engine='openpyxl')


# In[6]:


# get the overview of the columns
for i in data.columns:
    print(i)


# ### 1. Preprcoessing of data files  
# Preprocessing procedure is done with `preprocessing.py`
# For the code and steps check the script itself

# In[7]:


# initialize preprocessing procedure 
data = CovidPreprocessor(lab_data = lab, cat_data = data, 
                         appendix = appdx, new_cats = cats)
# transform the lab values and categorical dar
lab_data = data.transform_lab_data()
cat_data = data.transform_cat_data()


# In[8]:


# combine into single dataframe 
complete_data = pd.concat([lab_data,cat_data],
                          axis = 1)


# In[9]:


complete_data.head()


# In[27]:


complete_data['outcome'].value_counts()


# ### 2. Exploratory Data Analysis 

# In[10]:


complete_data.info()


# In[11]:


f'Number of survived patients : {len(complete_data[complete_data.outcome == 1])} which is  {(126*100)/198}%'


# In[12]:


f'Number of deceased patients : {len(complete_data[complete_data.outcome == 0])} which is  {(72*100)/198}%'


# In[17]:


f'There are {len(list(cat_data.columns))} features in the comorbidities group: {list(cat_data.columns)}'


# In[18]:


f'There are {len(list(lab_data.columns))} features in the laboratory values group: {list(lab_data.columns)}'


# #### Plot missing values

# In[19]:


msno.matrix(complete_data)


# In[20]:


###### get the boxplots for every feature in the dataframe 
# plotly setup
plot_rows=5
plot_cols=5
fig = make_subplots(rows=plot_rows, cols=plot_cols,subplot_titles=lab_data.columns)

# add traces
x = 0
for i in range(1, plot_rows + 1):
    for j in range(1, plot_cols + 1):
        #print(str(i)+ ', ' + str(j))
        fig = fig.add_trace(go.Box(x=lab_data['outcome'], y=lab_data[lab_data.drop(labels = ['outcome'], axis =1).columns[x]].values, 
                                 name = lab_data.drop(labels = ['outcome'], axis =1).columns[x]), row=i, col=j)
        x=x+1
        
    for k in fig['layout']['annotations']:
        k['font'] = dict(size=8, color='black', family = 'sans-serif')

# Format and show fig
fig.update_layout(title = {'text' : 'Laboratory Features Boxplot',
                           'xanchor': 'center','yanchor': 'top'}, 
                           height=1200, width=1200, showlegend = True)
fig.show()


# In[21]:


# get the boxplots for every feature in the dataframe 
# plotly setup
plot_rows=4
plot_cols=4
fig = make_subplots(rows=plot_rows, cols=plot_cols,subplot_titles=cat_data.drop(labels = 'age', axis = 1).columns)

# add traces
x = 0
for i in range(1, plot_rows + 1):
    for j in range(1, plot_cols + 1):
        #print(str(i)+ ', ' + str(j))
        fig = fig.add_trace(go.Bar(x=lab_data['outcome'], y=cat_data.drop(labels = ['age'], axis = 1).iloc[:,x].value_counts(),
                                 name = cat_data.drop(labels = ['age'], axis =1).columns[x]), row=i, col=j)
        x=x+1
        
    for i in fig['layout']['annotations']:
        i['font'] = dict(size=8, color='black', family = 'sans-serif')

# Format and show fig
fig.update_layout(title = {'text' : 'Categorical Features Bar plot',
                           'xanchor': 'center','yanchor': 'top'}, 
                          height=1200, width=1200, showlegend = True)
fig.show()


# ### 2.1 Analysis pipeline 
# Steps:  
# - Do a nested 10-fold Cross Validation testing 3 estimators: `GradientBoostingSurvivalAnalysis`,`RandomForestSurvival`,`CoXPHAnalysis`;  
# - After every iteration, the best estimator is selected and then fitted on the train data and tested on the unseed test partition;  
# - The final scores are reported in the .csv file  

# In[ ]:


# define the data and the labels
X_rsf, y_rsf = get_x_y(complete_data, attr_labels = ['outcome','time'], pos_label = 0)

### define the outer CV
cv_outer = StratifiedKFold(n_splits=10)

### define dictionaries to store the results 
result_icv = [] #results of internal cross validation
result_ncv = [] #results of external cross validation

# feature names
numerical = list(complete_data.drop(labels = ['outcome','time'], axis = 1).columns)
categorical = list(cat_data.columns)

# define the imputer (similar to miceforest library)
BayesianRidge = IterativeImputer(estimator = BayesianRidge(), random_state = 1, verbose = 3, max_iter = 40, tol = 0.001)
robust_scaler = RobustScaler()


### initiate the outer loop of CV 
### splits the data into the train and test 
for train_ix, test_ix in cv_outer.split(X_rsf,y_rsf):
    
    # Outer cross-validation: split the data into train and holdout test set
    X_train, X_test = X_rsf.iloc[train_ix, :], X_rsf.iloc[test_ix, :]
    y_train, y_test = y_rsf[train_ix], y_rsf[test_ix]
    
    #### Define pipeline and parameter grid ####

    models = {
    'RandomSurvivalForest':RandomSurvivalForest(),
    'GradientBoostingSurvivalAnalysis':GradientBoostingSurvivalAnalysis(),
    'CoxPHSurvivalAnalysis':CoxPHSurvivalAnalysis()
    }

    params_1 = {
             'RandomSurvivalForest': [{'bootstrap':[True,False], 
              'n_estimators':np.arange(1,500,100).tolist(), 
              'max_depth':[None,1,3,5,9],
              'min_samples_split':np.arange(2,int(len(X_train)/10),20).tolist(),
              'min_samples_leaf':np.arange(1,int(len(X_train)/10),20).tolist()
                                        }],
        
              'GradientBoostingSurvivalAnalysis': [{'n_estimators':np.arange(1,500,100).tolist(), 'learning_rate': [0.1,0.2,0.5,1],  
                                                    'subsample': [0.5,1], 'loss':['coxph','squared', 'ipcwls'], 
                                                    'max_depth':[None,1,3,5], 'dropout_rate':[0.1, 0.5, 0.9]
               }], 
        
             'CoxPHSurvivalAnalysis':[{'alpha':np.arange(0.1,10,0.1).tolist(),
                                      'ties':['breslow','efron'],
                                      'n_iter': np.arange(100,1000,100).tolist()
                                      }]
              }
    
    numeric_transformation = Pipeline(steps = [('RobustScaler', robust_scaler), ('BayesianRidgeImputer',
                                                                                 BayesianRidge),])
    categorical_transformation =Pipeline(steps = [('SimpleImputer', SimpleImputer(strategy = 'most_frequent'))])
    
    preprocessing_pipeline = ColumnTransformer(transformers = [('numerical', numeric_transformation, numerical),
                                                               ('categorical', categorical_transformation, categorical)])

    # Internal cross-validation: the train set is further divided into train and validation
    full_pipeline = Pipeline(steps = [('preprocessing_pipeline', preprocessing_pipeline), 
                                      ('EstimatorSelectionHelper',EstimatorSelectionHelper(models, params_1))])
    
    
    
    
    full_pipeline.fit(X_train, y_train)
    grid_result = full_pipeline['EstimatorSelectionHelper'].score_summary()
    
    # get results of internal cross validation
    result_icv.append(grid_result)
    
    best_model = grid_result.iloc[[0]]
    best_list = best_model.to_dict(orient = 'records')
    best_model_dict  = best_list[0]

    ### Calculate feature importance for the best method
    if best_model_dict['estimator'] == 'RandomSurvivalForest':
        best_model_fit = eval(best_model_dict['estimator'])(bootstrap = best_model_dict['bootstrap'],
                                                           max_depth = best_model_dict['max_depth'],
                                                           min_samples_leaf = best_model_dict['min_samples_leaf'],
                                                           min_samples_split = best_model_dict['min_samples_split'],
                                                           n_estimators = best_model_dict['n_estimators']) 
        ##
        X_train = pd.DataFrame(preprocessing_pipeline.fit_transform(X_train), columns = X_train.columns)
        X_test = pd.DataFrame(preprocessing_pipeline.transform(X_test), columns = X_test.columns)
        best_model_fit.fit(X_train, y_train)  
        
        #fetch the score on the test data 
        y_score = best_model_fit.score(X_test,y_test)
        
        # Permutation Feature Importance calculation 
        perm_imp = PermutationImportance(best_model_fit, n_iter=5)
        perm_imp.fit(X_train, y_train, verbose = 0)
        perm_df = eli5.explain_weights(perm_imp, feature_names = list(X_train.columns))
        perm_feature_importance = eli5.format_as_dataframe(perm_df)
        best_model_dict['permutation_importance'] = perm_feature_importance.to_dict(orient = 'dict')
        
    elif best_model_dict['estimator'] == 'GradientBoostingSurvivalAnalysis': 
        best_model_fit = eval(best_model_dict['estimator'])(n_estimators = best_model_dict['n_estimators'],
                                                           learning_rate = best_model_dict['learning_rate'],
                                                           loss = best_model_dict['loss'],
                                                           max_depth = best_model_dict['max_depth'],
                                                           dropout_rate = best_model_dict['dropout_rate']
                                                           )   
        
        X_train = pd.DataFrame(preprocessing_pipeline.fit_transform(X_train), columns = X_train.columns)
        X_test = pd.DataFrame(preprocessing_pipeline.transform(X_test), columns = X_test.columns)
        best_model_fit.fit(X_train, y_train)  
        
        #fetch the score on the test data 
        y_score = best_model_fit.score(X_test,y_test)
        
        #inbuilt feature importance 
        fimp = pd.DataFrame(best_model_fit.feature_importances_, index = X_train.columns, columns = ['weight'])
        best_model_dict['important_features'] = fimp.to_dict(orient = 'dict')

    
        
        # Permutation Feature Importance calculation 
        perm_imp = PermutationImportance(best_model_fit, n_iter=5)
        perm_imp.fit(X_train, y_train, verbose = 0)
        perm_df = eli5.explain_weights(perm_imp, feature_names = list(X_train.columns))
        perm_feature_importance = eli5.format_as_dataframe(perm_df)
        best_model_dict['permutation_importance'] = perm_feature_importance.to_dict(orient = 'dict')
        
    elif best_model_dict['estimator'] == 'CoxPHSurvivalAnalysis':
        best_model_fit = eval(best_model_dict['estimator'])(alpha = best_model_dict['alpha'],
                                                           ties = best_model_dict['ties'],
                                                           n_iter = best_model_dict['n_iter'])
          
        X_train = pd.DataFrame(preprocessing_pipeline.fit_transform(X_train), columns = X_train.columns)
        X_test = pd.DataFrame(preprocessing_pipeline.transform(X_test), columns = X_test.columns)
        best_model_fit.fit(X_train, y_train)  
        
        #fetch the score on the test data 
        y_score = best_model_fit.score(X_test,y_test)
        
        # Permutation Feature Importance calculation 
        perm_imp = PermutationImportance(best_model_fit, n_iter=5)
        perm_imp.fit(X_train, y_train, verbose = 0)
        perm_df = eli5.explain_weights(perm_imp, feature_names = list(X_train.columns))
        perm_feature_importance = eli5.format_as_dataframe(perm_df)
        best_model_dict['permutation_importance'] = perm_feature_importance.to_dict(orient = 'dict')
    
    
    #append the score on the test set to the dataframe 
    best_model_dict['test_score'] = y_score 
    
    #fetch the samples ID for test and train splits 
    best_model_dict['train_split_sample_ID'] = list(X_rsf.iloc[train_ix, :].index)
    best_model_dict['test_split_sample_ID'] = list(X_rsf.iloc[test_ix, :].index)
    
    result_ncv.append(best_model_dict)


# In[ ]:


result_ncv = pd.DataFrame(result_ncv)

# save the results of the model selection
# create the directory if it doesn't exist
if not os.path.isdir('/results'):
    os.makedirs('/results')
# save the results of internal and external cross validations
today = str(date.today())
result_ncv.to_csv(os.path.join(r'~/results','survival-positive-patients/nested_cv_results_{}_ratios_only.csv'.format(today)))
with ExcelWriter(os.path.join('results','internal_cv_final_results_{}_ratios_only.xlsx'.format(today))) as writer:
    for n, df in enumerate(result_icv):
        df.to_excel(writer,'sheet%s' % n)
    writer.save()


# ### 3. Plotting the feature importance and metrics of the 10 best models  
# - Two feature importance plots are made:  
# 1. Inbuilt feature importance (in case of `GradientBoostingSurvivalAnalysis` | `CoxPHAnalysis`)  
# 2. Permutation feature importance for all the estimators  
# 
# 
# Additionally, ranking of features across 10 best models is done (median ranking score is taken for every estimator)  

# In[23]:


# load the csv file with the result if it's not in the session
result_ncv = pd.read_csv(os.path.join(r'~/results/survival-positive-patients','nested_cv_final_results_{}_ratios_only.csv'.format(today)))


# ### 3.1 Plot the model metrics during cross validation  
# - Blue bar indicates median score during corss validation  
# - Red bar indicates score on the unseen test data  
# The number after the estimator name in case when the estimators are the same (e.g. `GrdientBoostingSurvivalAnalysis_1` etc.) indicates a model with different set of hyperparameters  

# In[13]:


x = list(result_ncv.estimator)  
for i,j in enumerate(x):
    x[i] = j + '_' + str(i)
    
layout = ({"title": "Performance of 10 best models (with scaling)",
                       "yaxis": {"title":"Accuracy"},
                       "xaxis": {"title":"Model"},
                       "showlegend": True})

plot = go.Figure(data=[go.Bar( 
    name = 'Median score train', 
    x = x, 
    y = result_ncv.median_score,
    error_y=dict(type='data', array= result_ncv.std_score)
   ), 
                       go.Bar( 
    name = 'Test score', 
    x = x, 
    y = result_ncv.test_score 
   ) 
], layout = layout) 

plot.write_html(r"~/results/plot_{}_CV_barplot_ratios_only.html".format(today))
plot.show()


# ### 3.2 Plotting the feature importance for the inbuilt GradientBoostingSurvival Method  

# In[14]:


feature_importance = []
for i in range(0,10):
    if result_ncv.estimator.iloc[i] == 'GradientBoostingSurvivalAnalysis':
        feature_importance.append(pd.DataFrame(eval(result_ncv.important_features.iloc[i]))) 
    else:
        pass


feature_grouping = pd.concat(feature_importance, axis = 0)
feature_mean = feature_grouping.groupby(level = 0).mean()
feature_std = feature_grouping.groupby(level = 0).std()

feature_importance_df = pd.concat([feature_mean,feature_std], axis = 1)
feature_importance_df.reset_index(inplace = True)
feature_importance_df.columns = ['feature', 'weight_mean','weight_std']
print(feature_importance_df)

fimp_chart = [go.Bar( x = feature_importance_df.weight_mean, y = feature_importance_df.feature, orientation = 'h')]
my_layout = ({"title": "Average Feature Importance with inbuilt method for Gradient Boosting across 10 best models",
                       "yaxis": {"title":"Feature"},
                       "xaxis": {"title":"Weight"},
                       "showlegend": False})

fig = go.Figure(data = fimp_chart, layout = my_layout)

fig.update_layout(barmode='stack', yaxis={'categoryorder':'total ascending'})
fig.write_html(r"~/results/plot_feature_importance_inbuilt_GBoost_scaling_ratios_only_{}.html".format(today))
fig.show()


# In[25]:


perm_importance = []
for i in range(0,10):
    df = pd.DataFrame(eval(result_ncv.permutation_importance.iloc[i])).drop(labels = 'std', axis = 1)
    perm_importance.append(df)  



feature_grouping = pd.concat(perm_importance, axis = 0)
feature_grouping.set_index('feature', inplace = True)
feature_mean = feature_grouping.groupby(level = 0).mean()
feature_std = feature_grouping.groupby(level = 0).std()

perm_importance_df = pd.concat([feature_mean,feature_std], axis = 1)
perm_importance_df.reset_index(inplace = True)
perm_importance_df.columns = ['feature', 'weight_mean','weight_std']

# plotting 
fimp_chart = [go.Bar( x = perm_importance_df.weight_mean, y = perm_importance_df.feature, orientation = 'h')]
my_layout = ({"title": "Average Feature Importance with permutation method across 10 best models",
                       "yaxis": {"title":"Feature"},
                       "xaxis": {"title":"Weight"},
                       "showlegend": False})

fig = go.Figure(data = fimp_chart, layout = my_layout)

fig.update_layout(barmode='stack', yaxis={'categoryorder':'total ascending'})
fig.write_html("/results/plot_feature_importance_inbuilt_GBoost_scaling_ratios_only_{}.html".format(today))
fig.show()


# ### Plot the feature rank  
# This is done to alleviate the problem when a feature can have a very high weight, but not in all of the models, thus meaning that it's only locally important

# In[16]:


i = 1
feature_rank = []
for i in range(0,10): 
    if result_ncv.estimator.iloc[i] == 'GradientBoostingSurvivalAnalysis':
        df = pd.DataFrame(eval(result_ncv.important_features.iloc[i]))
        df.sort_values(by = 'weight', ascending = False, inplace = True)
        df.reset_index(inplace = True)
        df.rename(columns = {'index': 'feature'}, inplace = True)
        # adding one so the count starts from 1 instead of 0 
        # otherwise it's not possible to plot the feature that ranks the highest (1 in this case)
        df.index += 1 
        df['rank'] = df.index
        feature_rank.append(df)
    else: 
        pass 

# add iteration column to every dataframe 
for df in feature_rank: 
    df['iteration'] = i
    i+= 1 


rank_grouping = pd.concat(feature_rank, axis = 0)
rank_median = rank_grouping.groupby('feature').agg({'weight':'median', 'rank':'median'})

feature_rank_df = pd.concat([rank_median], axis = 1)
feature_rank_df.reset_index(inplace = True)
 

fimp_chart = [go.Bar( x = feature_rank_df['rank'], y = feature_rank_df.feature, orientation = 'h')]
my_layout = ({"title": "Median Feature Rank for algorithms across 10 best models",
                       "yaxis": {"title":"Feature"},
                       "xaxis": {"title":"Weight"},
                       "showlegend": False})

fig = go.Figure(data = fimp_chart, layout = my_layout)

fig.update_layout(barmode='stack', yaxis={'categoryorder':'total descending'})
fig.write_html("~/results/feature_rank_permutation_all_scaling_ratios_only_{}.html".format(today))
fig.show()


# In[48]:


k = 1
perm_rank = []
for i in range(0,10): 
    df = pd.DataFrame(eval(result_ncv.permutation_importance.iloc[i])).drop(columns = ['std'], axis = 1)
    df.sort_values(by = 'weight', ascending = False, inplace = True)
    #df.reset_index(inplace = True)
    df.rename(columns = {'index': 'feature'}, inplace = True)
    # adding one so the count starts from 1 instead of 0 
    # otherwise it's not possible to plot the feature that ranks the highest (1 in this case)
    df.index += 1 
    df['rank'] = df.index
    perm_rank.append(df)

    
### add iteration column to every data frame in the list

for df in perm_rank: 
    df['iteration'] = i
    k+= 1 

    
rank_grouping = pd.concat(perm_rank, axis = 0)
rank_median = rank_grouping.groupby('feature').agg({'weight':'median', 'rank':'median'})

perm_rank_df = pd.concat([rank_median], axis = 1)
perm_rank_df.reset_index(inplace = True)
 

fimp_chart = [go.Bar( x = perm_rank_df['rank'], y = perm_rank_df.feature, orientation = 'h')]
my_layout = ({"title": "Median Feature Rank for algorithms across 10 best models",
                       "yaxis": {"title":"Feature"},
                       "xaxis": {"title":"Weight"},
                       "showlegend": False})

fig = go.Figure(data = fimp_chart, layout = my_layout)

fig.update_layout(barmode='stack', yaxis={'categoryorder':'total descending'})
fig.write_html("/results/plot_feature_rank_permutation_all_scaling_ratios_only_{}.html".format(today))
fig.show()


# In[20]:


### save to csv file for further plotting in R 
#pd.concat(feature_rank, axis = 0).to_csv('feature_rank_iterations.csv')
#pd.concat(perm_rank, axis = 0).to_csv('perm_rank_iterations.csv')


# Bump chart and all other figures for the paper are done in R

# ### 4. (Extra) Evaluation of the best estimator on different test/train splits  
# - The best model is selected based on the highest median score of cross validation  
# - the model is fit and tested on different partitions of test/train sets  
# - feature importance is then retireved  
# 
# __This is not included in the final analysis interpretation, because it was decided to take the median across several algorithms instead__

# In[8]:


# picking the best model according to the median score 
result_ncv.iloc[3]


# In[60]:


# test the chosen model 1 different train and test splits 
# define the data and the labels
X_rsf, y_rsf = get_x_y(complete_data, attr_labels = ['outcome','time'], pos_label = 0)

### define the outer CV
cv_outer = StratifiedKFold(n_splits=10)

### define dictionaries to store the results 
best_model_metrics = []
counter = 1
# feature names
numerical = list(comp_df.drop(labels = ['outcome','time'], axis = 1).columns) 
categorical = list(new_cat.columns)

# define the miceforest imputer
BayesianRidge = IterativeImputer(estimator = BayesianRidge(), random_state = 1, verbose = 1, max_iter = 40, tol = 0.001)
robust_scaler = RobustScaler()
gbs_best_model = GradientBoostingSurvivalAnalysis(dropout_rate = 0.1, learning_rate = 0.5, loss = 'coxph', max_depth = 3, n_estimators = 101, subsample = 0.5)

### initiate the outer loop of CV 
### splits the data into the train and test 
for train_ix, test_ix in cv_outer.split(X_rsf,y_rsf):
    
    # split data
    X_train, X_test = X_rsf.iloc[train_ix, :], X_rsf.iloc[test_ix, :]
    y_train, y_test = y_rsf[train_ix], y_rsf[test_ix]
    
    print(train_ix)
    print(test_ix)
    row = {}
    
    numeric_transformation = Pipeline(steps = [('RobustScaler', robust_scaler), ('BayesianRidgeImputer', BayesianRidge),])
    categorical_transformation =Pipeline(steps = [('SimpleImputer', SimpleImputer(strategy = 'most_frequent'))])
    
    preprocessing_pipeline = ColumnTransformer(transformers = [('numerical', numeric_transformation, numerical),
                                                               ('categorical', categorical_transformation, categorical)])
    
    X_train = pd.DataFrame(preprocessing_pipeline.fit_transform(X_train), columns = X_train.columns)
    X_test = pd.DataFrame(preprocessing_pipeline.transform(X_test), columns = X_test.columns)
    gbs_best_model.fit(X_train, y_train)  

    #fetch the score on the test data
    y_train_score = gbs_best_model.score(X_train,y_train)
    y_hat_score = gbs_best_model.score(X_test,y_test)
    
    #best_model_metrics['train_score'] = y_train_score
    #best_model_metrics['test_score'] = y_hat_score
    
    print(y_train_score)
    print(y_hat_score)
    

    #inbuilt feature importance 
    fimp = pd.DataFrame(gbs_best_model.feature_importances_, index = X_train.columns, columns = ['weight'])
    #best_model_metrics['inbuilt_importance'] = fimp.to_dict(orient = 'dict')

    # Permutation Feature Importance calculation 
    perm_imp = PermutationImportance(gbs_best_model, n_iter= 10)
    perm_imp.fit(X_train, y_train, verbose = 0)
    perm_df = eli5.explain_weights(perm_imp, feature_names = list(X_train.columns))
    perm_feature_importance = eli5.format_as_dataframe(perm_df)
    #best_model_metrics['permutation_importance'] = perm_feature_importance.to_dict(orient = 'dict')
    
    counter += 1 
    #best_model_metrics['iter'] = counter
    row.update({'iter':counter, 
               'train_score':y_train_score,
               'test_score':y_hat_score,
               'inbuilt_importance':fimp.to_dict(orient = 'dict'),
               'permutation_importance':perm_feature_importance.to_dict(orient = 'dict'),
               'train_id':train_ix,
               'test_id':test_ix})
    
    best_model_metrics.append(row)

best_model_metrics = pd.DataFrame(best_model_metrics) 
best_model_metrics.to_csv("~/results/best_model_train_test_splits_{}.csv".format(today))


# In[44]:


split1_inbuilt = pd.DataFrame(best_model_metrics.inbuilt_importance.iloc[0]) 
fimp_chart_inbuilt = [go.Bar( x = split1_inbuilt.weight, y = split1_inbuilt.index, orientation = 'h')]
layout_inbuilt = ({"title": "Gradient Boosting Survival Feature Importance",
                       "yaxis": {"title":"Feature"},
                       "xaxis": {"title":"Weight"},
                       "showlegend": False})

fig = go.Figure(data = fimp_chart_inbuilt, layout = layout_inbuilt)

fig.update_layout(barmode='stack', yaxis={'categoryorder':'total ascending'})
fig.write_html("~/results/plot_feature_importance_inbuilt_GBoost_one_model_{}.html".format(today))
fig.show()


# In[61]:


split1_perm = pd.DataFrame(best_model_metrics.permutation_importance.iloc[0]) 
fimp_chart_perm = [go.Bar( x = split1_perm.weight, y = split1_perm.feature, orientation = 'h')]
layout_perm = ({"title": "Gradient Boosting Survival Permutation Feature Importance",
                       "yaxis": {"title":"Feature"},
                       "xaxis": {"title":"Weight"},
                       "showlegend": False})

fig = go.Figure(data = fimp_chart_perm, layout = layout_perm)

fig.update_layout(barmode='stack', yaxis={'categoryorder':'total ascending'})
fig.write_html("~/results/plot_feature_importance_permutation_GBoost_one_model_{}.html".format(today))
fig.show()

