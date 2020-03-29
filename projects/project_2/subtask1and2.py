#!/usr/bin/env python
# coding: utf-8

# # Notebook for experiments on modelling the second IML project data
# 
# As we go it may be good to keep this place tidy by implementing some of the suggestions made here: https://www.thoughtworks.com/insights/blog/coding-habits-data-scientist 

# ## TODO
# * Modelling
# * Tidy up code to make sure it runs in parallel where possible
# * Conform to guidelines provided in link above

# ## 1. Dependency management
# Install the provided environment.yml file in the root directory of the repository. 
# If this fails, run the commands below in a terminal window. This may lead to clashes with other packages!

# In[1]:


# !conda install -c conda-forge jupyterlab -y
# !conda install -c conda-forge imbalanced-learn -y
# !conda install -c conda-forge pandas-profiling -y
# !conda install -c conda-forge bokeh -y


# In[2]:


get_ipython().system('pip install pip install thundersvm')


# ## 2. Data Visualization & Analysis 

# In[3]:


import pandas_profiling
import multiprocessing
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import missingno as msno
from collections import Counter
from sklearn.preprocessing import FunctionTransformer
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import StandardScaler
from ipywidgets import widgets
from pandas_profiling import ProfileReport
from bokeh.layouts import gridplot
from bokeh.plotting import figure, output_file, show
from imblearn.over_sampling import ADASYN, SMOTE
from imblearn.under_sampling import ClusterCentroids
from sklearn.svm import LinearSVC,SVC
from sklearn.model_selection import GridSearchCV
from random import sample

get_ipython().run_line_magic('matplotlib', 'inline')


# In[4]:


# Load data
df_train = pd.read_csv("data/train_features.csv")
df_train_label = pd.read_csv("data/train_labels.csv")
df_test = pd.read_csv("data/test_features.csv")


# In[5]:


msno.matrix(df_train)


# In[6]:


msno.matrix(df_test)


# In[7]:


# The heatmap measures nullity correlation: how strongly the presence or absence of one variable affects the presence of another.
msno.heatmap(df_train)


# In[8]:


# Using hierarchichal clustering we can see how the nullity of a variable relates to that of another.
msno.dendrogram(df_train)


# Conclusions from above plots:
# * Data appears to be i.i.d.
# * Some columns appear to have many more measurements than others. 
# * When a column is null, other columns also tend to be null (some timepoints
# seem to contain more data than others, might be useful when doing imputation)
# 
# Let's have a look at the different types and other aspects of the dataframe:
# 

# In[9]:


df_train.head(20)


# In[10]:


df_train.dtypes


# In[11]:



df_train.describe()


# In[12]:


# Preliminary data analysis to see what we're dealing with here
# Expensive computation, output already on repository in separate HTML file
# profile_df_train = ProfileReport(df_train, title='Pandas Profiling Report of Training Features', html={'style':{'full_width':True}})
# profile_df_train.to_file(output_file="profile_report_training_features.html")


# In[13]:


plt.rcParams['figure.figsize'] = 30, 15


# In[14]:


df_train_label.hist()


# ## 3. Preprocessing
# ### Ideas for preprocessing
# Resolve class imbalance:
# * Use oversampling via SMOTE and ADASYN (https://medium.com/coinmonks/smote-and-adasyn-handling-imbalanced-data-set-34f5223e167)
# 
# Resolve missing data:
# * Use averaging methods
# * Use dynamic time warping and treat data as time series to get time series of equal lengths.
# * Use EM algorithm
# * Use algorithms that can deal with missing data

# In[15]:


# list of medical tests that we will have to predict, as well as vital signs (to delete for this task)
medical_tests = ["LABEL_BaseExcess", "LABEL_Fibrinogen", "LABEL_AST", "LABEL_Alkalinephos", "LABEL_Bilirubin_total", "LABEL_Lactate", "LABEL_TroponinI", "LABEL_SaO2", "LABEL_Bilirubin_direct", "LABEL_EtCO2"]
vital_signs = ["LABEL_RRate","LABEL_ABPm","LABEL_SpO2","LABEL_Heartrate"]
sepsis = ["LABEL_Sepsis"]


# In[16]:


# slower version - supports patient specific mean
def fill_na_with_average_patient_column(df_train):
    columns = list(df_train.columns)
    del columns[0:2]

    df_train_preprocessed = df_train

    for i,column in enumerate(columns):
        print("{} column of {} columns processed".format(i+1,len(columns)), end="\r") 
        # Fill na with patient average 
        df_train_preprocessed[[column]] = df_train_preprocessed.groupby(['pid'])[column]        .transform(lambda x: x.fillna(x.mean()))
        
    # Fill na with overall column average for lack of a better option for now
    df.fillna(df.mean())
    return df_train_preprocessed


# In[17]:


# quick version - does not support patient average
def fill_na_with_average_patient_column(df):
    return df.fillna(df.mean())


# In[18]:


# Would be useful to distribute/multithread this part
df_train_preprocessed = fill_na_with_average_patient_column(df_train)


# In[19]:


# preprocess testing data
df_test_preprocessed = fill_na_with_average_patient_column(df_test)


# In[20]:


# # There are a lot less missing values now.
# msno.bar(df_train_preprocessed)
# plt.show()


# In[21]:


# transform training labels for these tasks
df_train_label[medical_tests+vital_signs+sepsis] = df_train_label[medical_tests+vital_signs+sepsis].astype(int)
df_train_label_med = df_train_label.drop(columns=vital_signs+sepsis)
df_train_label_sepsis = df_train_label.drop(columns=vital_signs+medical_tests)


# In[22]:


# Merging pids to make sure they map correctly.
df_train_preprocessed_merged = pd.merge(df_train_preprocessed, df_train_label,  how='left', left_on='pid', right_on ='pid')


# In[23]:


# Transform to arrays for further processing
X_train = df_train_preprocessed_merged.drop(columns=medical_tests+sepsis+vital_signs).values

# get a different label for each medical test
y_train_set_med = []
for test in medical_tests:
    y_train_set_med.append(df_train_preprocessed_merged[test].values)
y_train_sepsis = df_train_preprocessed_merged['LABEL_Sepsis'].values


# Resampling dataset prior to model fitting

# In[24]:


def oversampling_strategies(X_train, y_train, strategy="adasyn"):
    # Oversampling methods
    if strategy=="adasyn":
        sampling_method = ADASYN()
    if strategy=="smote":
        sampling_method = SMOTE()
    
    # Undersampling methods
    if strategy=="clustercentroids":
        sampling_method = ClusterCentroids(random_state=42)
    if strategy=="random":
        sampling_method = RandomUnderSampler(random_state=0, replacement=True)
        
    X_train_resampled, y_train_resampled = sampling_method.fit_sample(X_train, y_train)
    
    print(sorted(Counter(y_train_resampled).items()))
    
    return X_train_resampled, y_train_resampled


# In[ ]:


# very long to compute
# compute resampled data for all medical tests 
X_train_resampled_set_med,y_train_resampled_set_med = [0]*len(y_train_set_med),[0]*len(y_train_set_med)
for i in range(len(y_train_set_med)):
    X_train_resampled_set_med[i], y_train_resampled_set_med[i] = oversampling_strategies(X_train, y_train_set_med[i], strategy="adasyn")


# In[ ]:


X_train_resampled_sepsis, y_train_resampled_sepsis = oversampling_strategies(X_train, y_train_sepsis, strategy="adasyn")


# ## 4. Modelling

# For now, use of Linear SVM, scales better to large datasets 
# 
# TODO: Try using regular SVR to be able to use kernels

# In[ ]:


# Linear
# regularization parameter
alphas = np.linspace(0.1,10,num=3)
# to perform either l1 or l2 regularization 
penalty = ["l1","l2"]

# for non linear
# kernel type
kernels = ["linear", "poly", "rbf", "sigmoid"]
# degree for poly kernel
degrees = range(1,5)
# gamma parameter for poly or rbf kernel
gamma_rbf = np.linspace(0.1,10,num=10)


# Random sampling of dataset for model testing

# In[ ]:


def get_random_sample(X_train_resampled_set,y_train_resampled_set,size=100):
    """Sample at random datapoints from the resampled datasets for each medical test
    
    Parameters: 
        X_train_resampled_set = np.array, set of size # of medical tests, with X_train for each
        y_train_resampled_set = np.array, set of size # of medical tests, with y_train for each
                                size = int, size of selected sample
    Returns:
        X_train_rd_set,y_train_rd_set : np.array, reduced sample sets where xxx_train_rd_set[i] is the reduced
                                            set for medical test i
    """
    X_train_rd_set,y_train_rd_set = [],[]
    for i,test in enumerate(medical_tests):
        ind = sample(range(len(X_train_resampled_set[i])),size)
        X_train_rd_set.append(X_train_resampled_set[i][ind])
        y_train_rd_set.append(y_train_resampled_set[i][ind])
    return np.array(X_train_rd_set),np.array(y_train_rd_set)


# In[ ]:


def get_models_medical_tests(X_train_resampled_set,y_train_resampled_set, alpha=10, param_grid={"C": [1,10]}, typ="naïve", reduced=True, size=100):
    """Function to obtain models for every set of medical test, either naïve or using CV Gridsearch
    
        Parameters: X_train_resampled_set = np.array, set of size # of medical tests, with X_train for each
                    y_train_resampled_set = np.array, set of size # of medical tests, with y_train for each
                    alpha = float (for naïve) regularization parameter, ignored if typ is not naive
                    param_grid = dict (for gridsearch), dictionary of parameters to search over, ignored if typ is not gridsearch
                    typ = str in ["naïve","gridsearch","naive_non_lin","gridsearch_non_lin"], default "naïve", how the task is performed
                    reduced = boolean, default True, if random sampling of dataset to test of smaller dataset
                    size = int, size of selected sample, ignored if reduced == False
        Returns:
                svr_models = list of Linear SVR models for each medical test, where svr_models[i] is the fitted 
                            model (best estimator in the case of gridsearch) for medical_test[i]
    """
    assert typ in ["naive","gridsearch","naive_non_lin","gridsearch_non_lin"], "typ must be in ['naive','gridsearch','naive_non_lin','gridsearch_non_lin']"
    if reduced:
        X_train_resampled_set, y_train_resampled_set = get_random_sample(X_train_resampled_set,y_train_resampled_set,size=size)
    svm_models = []
    for i,test in enumerate(medical_tests):
        print("Starting iteration for test {}".format(test))
        if typ=="naive":
            # setting dual to false because n_samples>n_features
            lin_svm = LinearSVC(C=alpha,dual=False)
            lin_svm.fit(X_train_resampled_set[i],y_train_resampled_set[i])
            svm_models.append(lin_svm)
        elif typ=="gridsearch":
            cores=multiprocessing.cpu_count()-2
            gs_svm = GridSearchCV(estimator=LinearSVC(dual=False),param_grid=param_grid,n_jobs=cores,scoring="roc_auc",verbose=2)
            gs_svm.fit(X_train_resampled_set[i],y_train_resampled_set[i])
            print("The estimated auc roc score for this estimator is {}, with alpha = {}".format(gs_svm.best_score_,gs_svm.best_params_))
            svm_models.append(gs_svm.best_estimator_)
        elif typ=="naive_non_lin":
            lin_svm = SVC(C=alpha)
            lin_svm.fit(X_train_resampled_set[i],y_train_resampled_set[i])
            svm_models.append(lin_svm)
        elif typ=="gridsearch_non_lin":
            cores=multiprocessing.cpu_count()-2
            gs_svm = GridSearchCV(estimator=SVC(),param_grid=param_grid,n_jobs=cores,scoring="roc_auc",verbose=2)
            gs_svm.fit(X_train_resampled_set[i],y_train_resampled_set[i])
            print("The estimated auc roc score for this estimator is {}, with alpha = {}".format(gs_svm.best_score_,gs_svm.best_params_))
            svm_models.append(gs_svm.best_estimator_)
    return svm_models


# In[ ]:


def get_model_sepsis(X_train_resampled,y_train_resampled, alpha=10, param_grid={"C": [1,10]}, typ="naïve", reduced=True, size=100):
    svm = LinearSVC()
    assert typ in ["naive","gridsearch","naive_non_lin","gridsearch_non_lin"], "typ must be in ['naive','gridsearch','naive_non_lin','gridsearch_non_lin']"
    if reduced:
        ind = sample(range(len(X_train_resampled)),size)
        X_train_resampled, y_train_resampled = X_train_resampled[ind],y_train_resampled[ind]
    if typ=="naive":
        # setting dual to false because n_samples>n_features
        svm = LinearSVC(C=alpha,dual=False)
        svm.fit(X_train_resampled,y_train_resampled)
    elif typ=="gridsearch":
        cores=multiprocessing.cpu_count()-2
        gs_svm = GridSearchCV(estimator=LinearSVC(dual=False),param_grid=param_grid,n_jobs=cores,scoring="roc_auc",verbose=2)
        gs_svm.fit(X_train_resampled,y_train_resampled)
        print("The estimated auc roc score for this estimator is {}, with alpha = {}".format(gs_svm.best_score_,gs_svm.best_params_))
        svm = gs_svm.best_estimator_
    elif typ=="naive_non_lin":
        svm = SVC(C=alpha)
        svm.fit(X_train_resampled,y_train_resampled)
    elif typ=="gridsearch_non_lin":
        cores=multiprocessing.cpu_count()-2
        gs_svm = GridSearchCV(estimator=SVC(),param_grid=param_grid,n_jobs=cores,scoring="roc_auc",verbose=2)
        gs_svm.fit(X_train_resampled,y_train_resampled)
        print("The estimated auc roc score for this estimator is {}, with alpha = {}".format(gs_svm.best_score_,gs_svm.best_params_))
        svm = gs_svm.best_estimator_
    return svm


# Naïve SVR for all medical tests

# In[ ]:


naive_svm_models = get_models_medical_tests(X_train_resampled_set_med, y_train_resampled_set_med, alpha=10, typ="naive", reduced=True, size=100)


# CV GridSearch with different regularization parameters

# In[ ]:


gridsearch_svm_models = get_models_medical_tests(X_train_resampled_set_med, y_train_resampled_set_med, param_grid = {"C": alphas, "penalty": penalty}, typ="gridsearch", reduced=False, size=100)


# In[ ]:


naive_non_lin_svm_models = get_models_medical_tests(X_train_resampled_set_med, y_train_resampled_set_med, alpha=10, typ="naive_non_lin", reduced=False, size=100)


# In[ ]:


# heavy computation, was too long to run on my machine
gridsearch_non_lin_svm_models = get_models_medical_tests(X_train_resampled_set_med, y_train_resampled_set_med, param_grid = {"C": alphas, "kernel": kernels, "degree": degrees}, typ="gridsearch_non_lin", reduced=False, size=20)


# In[ ]:


naive_sepsis_model = get_model_sepsis(X_train_resampled_sepsis,y_train_resampled_sepsis,alpha=5,typ="naive", reduced=False, size=100)


# In[ ]:


gridsearch_sepsis_model = get_model_sepsis(X_train_resampled_sepsis,y_train_resampled_sepsis, param_grid = {"C": alphas, "penalty": penalty}, typ="gridsearch", reduced=False, size=100)


# In[ ]:


non_lin_sepsis_model = get_model_sepsis(X_train_resampled_sepsis,y_train_resampled_sepsis, alpha=5, typ="naive_non_lin", reduced=False, size=100)


# In[ ]:


# heavy computation
non_lin_gridsearch_sepsis_model = get_model_sepsis(X_train_resampled_sepsis,y_train_resampled_sepsis, param_grid = {"C": alphas, "kernel": kernels, "degree": degrees}, typ="gridsearch_non_lin", reduced=False, size=20)


# ## 5. Performance assessment

# In[ ]:


# def of sigmoid
def sigmoid_f(x):
    return 1/(1 + np.exp(-x))


# In[ ]:


X_test = df_test_preprocessed.values


# In[ ]:


# get the unique test ids of patients
test_pids = np.unique(df_test_preprocessed[["pid"]].values)


# To get predictions as confidence level, the model predicts for all 12 sets of measures for each patient a distance to the hyperplane ; it is then transformed into a confidence level using the sigmoid function ; the confidence level reported is the mean of all confidence levels for a single patient 

# In[ ]:


def get_predictions(X_test,test_pids,svm_models,reduced=True,nb_patients=100):
    """Function to obtain predictions for every model, as a confidence level : the closer to 1 (resp 0), the more confidently)
            the sample belongs to class 1 (resp 0).
        Parameters: X_test = np.array, set of preprocessed test values
                    test_pids = np.array, unique set of patient ids in test set
                    svm_models = list, fitted svm models to training set 
                    reduced = boolean, default True, if random sampling of dataset to test of smaller dataset
                    nb_patients = int, size of number of patients selected, ignored if reduced == False
        Returns:
                df_pred = pd.DataFrame, dataframe containing for each patient id the predicted label as a confidence level
    """
    df_pred = pd.DataFrame()
    if reduced:
        # sample at random nb_patients patients 
        test_pids = sample(list(test_pids),nb_patients)
        X_test = X_test[test_pids]
    for i,test in enumerate(medical_tests):
        # decision_function returns the distance to the hyperplane 
        y_conf = svm_models[i].decision_function(X_test)
        # compute the predictions as confidence levels, ie using sigmoid function instead of sign function
        y_pred = [sigmoid_f(y_conf[i]) for i in range(len(y_conf))]
        # use the mean of the computation for each patient as overall confidence level 
        y_mean = [np.mean(y_pred[i:i+12]) for i in range(len(test_pids))]
        df = pd.DataFrame({test: y_mean},index=test_pids)
        df_pred = pd.concat([df_pred,df], axis=1)
    return df_pred


# In[ ]:


def get_sepsis_predictions(X_test,test_pids,svm,reduced=True,nb_patients=100):
    """Function to obtain predictions for every model, as a confidence level : the closer to 1 (resp 0), the more confidently)
            the sample belongs to class 1 (resp 0).
        Parameters: X_test = np.array, set of preprocessed test values
                    test_pids = np.array, unique set of patient ids in test set
                    svm_models = list, fitted svm models to training set 
                    reduced = boolean, default True, if random sampling of dataset to test of smaller dataset
                    nb_patients = int, size of number of patients selected, ignored if reduced == False
        Returns:
                df_pred = pd.DataFrame, dataframe containing for each patient id the predicted label as a confidence level
    """
    if reduced:
        # sample at random nb_patients patients 
        test_pids = sample(list(test_pids),nb_patients)
        X_test = X_test[test_pids]
    # decision_function returns the distance to the hyperplane 
    y_conf = svm.decision_function(X_test)
    # compute the predictions as confidence levels, ie using sigmoid function instead of sign function
    y_pred = [sigmoid_f(y_conf[i]) for i in range(len(y_conf))]
    # use the mean of the computation for each patient as overall confidence level 
    y_mean = [np.mean(y_pred[i:i+12]) for i in range(len(test_pids))]
    df = pd.DataFrame({sepsis[0]: y_mean},index=test_pids)
    return df


# In[ ]:


naive_predictions = get_predictions(X_test,test_pids,naive_svm_models,reduced=False,nb_patients=100)


# In[ ]:


gridsearch_predictions = get_predictions(X_test,test_pids,gridsearch_svm_models,reduced=False,nb_patients=100)


# In[ ]:


# for some reason, the prediction function with SVC (non linear) is constant... need to look it up more
naive_non_lin_predictions = get_predictions(X_test,test_pids,naive_non_lin_svm_models,reduced=False,nb_patients=100)


# In[ ]:


naive_sepsis_predictions = get_sepsis_predictions(X_test,test_pids,naive_sepsis_model,reduced=False,nb_patients=100)


# In[ ]:


gridsearch_sepsis_predictions = get_sepsis_predictions(X_test,test_pids,gridsearch_sepsis_model,reduced=False,nb_patients=100)


# In[ ]:


naive_predictions.head()


# In[ ]:


gridsearch_predictions.head()


# In[ ]:


naive_non_lin_predictions.head()


# In[ ]:


naive_sepsis_predictions.head()


# In[ ]:


gridsearch_sepsis_predictions.head()


# In[ ]:


# suppose df is a pandas dataframe containing the result
# df.to_csv('prediction.zip', index=False, float_format='%.3f', compression='zip')

