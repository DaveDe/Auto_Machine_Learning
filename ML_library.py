#add boosting algorithms
#add feature selection methods

import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import ElasticNetCV
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingRegressor
import getModelPrediction as gmp
import handleInput as hi
import FormatDataset as fd


with open('input.txt', 'r') as f:
    chosenAlgorithms, perfCovariateShift, nominal_features_labels, trainData, testData, outputPredictions = hi.returnInputFileInfo(f)

X_train,X_test,Y_train,Y_test = fd.convertNominalToDummy(trainData,testData,nominal_features_labels)

perfLinearRegression = chosenAlgorithms[0]
perfPolynomialRegression = chosenAlgorithms[1]
perfANN = chosenAlgorithms[2]
perfElasticNet = chosenAlgorithms[3]
perfRandomForest = chosenAlgorithms[4]
perfGBM = chosenAlgorithms[5]
##################### Feature Selection #####################

#transform X_train and X_test


################# Drop features with high covariate drift ############
if(perfCovariateShift):
    print("Checking columns with high covariate drift...")

    #combine subsets of train and test data into one dataset,
    #add column which identifies which set it came from.

    #go through each feature, and if it can be easily predicted from that feature
    #whether each row belongs to training or test (it has a high ROC), then you
    #know that feature has significant covariate drift.

    X_train['origin'] = 0
    X_test['origin'] = 1

    #assume test data is about 30% the size of training
    X_train_sample = X_train.sample(frac = 0.33)
    X_test_sample = X_test.sample(frac = 0.33)

    X_train.drop('origin', axis=1, inplace = True)
    X_test.drop('origin', axis=1, inplace = True)

    ## combining random samples
    combined = X_train_sample.append(X_test_sample)
    y_combined = combined['origin']
    combined.drop('origin',axis=1,inplace=True)

    ## modelling
    model = RandomForestClassifier(n_estimators = 50, max_depth = 5,min_samples_leaf = 5)
    drop_list = []
    for i in combined.columns:
        #score is array of scores from both runs (cv = 2)
        score = cross_val_score(model,pd.DataFrame(combined[i]),y_combined,cv=2,scoring='roc_auc')
        if (np.mean(score) > 0.8):
            drop_list.append(i)

    #drop features with high drift
    X_train.drop(drop_list, axis=1, inplace = True)
    X_test.drop(drop_list, axis=1, inplace = True)

    if(len(drop_list) > 0):
        print("Dropping columns with high drift:",drop_list)
    else:
        print("No columns found with high drift")

################### Begin ML Regression Algorithms ###################
print("Running algorithms...")

#Linear Regression
if(perfLinearRegression):
    model = LinearRegression(random_state=10)
    r_squared,mse = gmp.getModelPredictions(model,X_train,Y_train,X_test,Y_test,outputPredictions,"Linear_Regression")
    print("\nLinear Regression R_squared:", r_squared)
    print("Linear Regression MSE:", mse)

#Polynomial Regression
if(perfPolynomialRegression):
    poly = PolynomialFeatures(degree=2)
    poly_X_train = poly.fit_transform(X_train)
    poly_X_test = poly.fit_transform(X_test)
    r_squared,mse = gmp.getModelPredictions(model,poly_X_train,Y_train,poly_X_test,Y_test,outputPredictions,"Polynomial_Regression")
    print("\nPolynomial Regression R_squared:", r_squared)
    print("Polynomial Regression MSE:", mse)

#ANN
if(perfANN):
    model = MLPRegressor(hidden_layer_sizes=(100,),solver="lbfgs",activation="relu",random_state=10)
    r_squared,mse = gmp.getModelPredictions(model,X_train,Y_train,X_test,Y_test,outputPredictions,"ANN")
    print("\nANN Regression R_squared:", r_squared)
    print("ANN Regression MSE:", mse)

#Elastic Net
if(perfElasticNet):
    model = ElasticNetCV(l1_ratio=[.1, .5, .7, .9, .95, .99, .995, 1], eps=0.001, n_alphas=100, fit_intercept=True,
                            normalize=True, precompute='auto', max_iter=2000, tol=0.0001, cv=5,
                            copy_X=True, verbose=0, n_jobs=-1, positive=False, random_state=10, selection='cyclic')
    r_squared,mse = gmp.getModelPredictions(model,X_train,Y_train,X_test,Y_test,outputPredictions,"Elastic_Net")
    print("\nElastic Net Regression R_squared:", r_squared)
    print("Elastic Net Regression MSE:", mse)

#Random Forest
if(perfRandomForest):
    model = RandomForestRegressor(random_state=10)
    r_squared,mse = gmp.getModelPredictions(model,X_train,Y_train,X_test,Y_test,outputPredictions,"Random_Forest")
    print("\nRandom Forest Regression R_squared:", r_squared)
    print("Random Forest Regression MSE:", mse)

#Gradient Boosting Machine
if(perfGBM):
    
    #tune number of estimators (decision trees)
    param_test = {'n_estimators':range(20,101,10)}
    min_samples_split = int(0.01*len(Y_train))

    ############################### SHOULD I BEGIN WITH ALL DEFAULTS?? ##############################
    gsearch = GridSearchCV(estimator = GradientBoostingRegressor(learning_rate=0.1, min_samples_split=min_samples_split,min_samples_leaf=50,
                                                                    max_features='sqrt',subsample=0.8,random_state=10),
                                        param_grid = param_test, n_jobs=4, iid=False, cv=5)
    gsearch.fit(X_train,Y_train.values.ravel())
    n_estimators = gsearch.best_params_['n_estimators']

    #check if n_estimators is 20 or 80, if so check more lower or higher values in another grid search
    if(n_estimators == 20):
        param_test = {'n_estimators':range(5,20,2)}
        gsearch = GridSearchCV(estimator = GradientBoostingRegressor(learning_rate=0.1, min_samples_split=min_samples_split,min_samples_leaf=50,
                                                                    max_features='sqrt',subsample=0.8,random_state=10),
                                        param_grid = param_test, n_jobs=4, iid=False, cv=5)
        gsearch.fit(X_train,Y_train.values.ravel())
        n_estimators = gsearch.best_params_['n_estimators']

    if(n_estimators == 100):
        param_test = {'n_estimators':range(100,161,10)}
        gsearch = GridSearchCV(estimator = GradientBoostingRegressor(learning_rate=0.1, min_samples_split=min_samples_split,min_samples_leaf=50,
                                                                    max_features='sqrt',subsample=0.8,random_state=10),
                                        param_grid = param_test, n_jobs=4, iid=False, cv=5)
        gsearch.fit(X_train,Y_train.values.ravel())
        n_estimators = gsearch.best_params_['n_estimators']

    #tune min number of samples needed to split
    upper_limit = min_samples_split
    step_size = int(upper_limit-2/10)
    param_test = {'min_samples_split':range(2,upper_limit+1,step_size)}
    gsearch = GridSearchCV(estimator = GradientBoostingRegressor(learning_rate=0.1, n_estimators=n_estimators, max_features='sqrt',
                                                                                min_samples_leaf=50,subsample=0.8, random_state=10),
                                                                param_grid = param_test, n_jobs=4, iid=False, cv=5)
    gsearch.fit(X_train,Y_train.values.ravel())
    min_samples_split = gsearch.best_params_['min_samples_split']

    #check if min_samples_split is an extreme value. If so tune on a more accurate range
    if(min_samples_split == upper_limit):
        step_size = int(upper_limit/2)
        param_test = {'min_samples_split':range(upper_limit, (upper_limit*10)+1,step_size)}
        gsearch = GridSearchCV(estimator = GradientBoostingRegressor(learning_rate=0.1, n_estimators=n_estimators, max_features='sqrt',
                                                                            min_samples_leaf=50,subsample=0.8, random_state=10),
                                                                    param_grid = param_test, n_jobs=4, iid=False, cv=5)
        gsearch.fit(X_train,Y_train.values.ravel())
        min_samples_split = gsearch.best_params_['min_samples_split']

    #tune max_depth
    param_test = {'max_depth':range(3,14,2)}
    gsearch = GridSearchCV(estimator = GradientBoostingRegressor(learning_rate=0.1, n_estimators=n_estimators,min_samples_split=min_samples_split,
                                                                max_features='sqrt', min_samples_leaf=50,subsample=0.8, random_state=10),
                                                                param_grid = param_test, n_jobs=4, iid=False, cv=5)
    gsearch.fit(X_train,Y_train.values.ravel())
    max_depth = gsearch.best_params_['max_depth']
    #check if max_depth is an extreme value. If so tune on a more accurate range
    if(max_depth == 13):
        param_test = {'max_depth':range(13,20,1)}
        gsearch = GridSearchCV(estimator = GradientBoostingRegressor(learning_rate=0.1, n_estimators=n_estimators,min_samples_split=min_samples_split,
                                                                max_features='sqrt', min_samples_leaf=50,subsample=0.8, random_state=10),
                                                                param_grid = param_test, n_jobs=4, iid=False, cv=5)
        gsearch.fit(X_train,Y_train.values.ravel())
        max_depth = gsearch.best_params_['max_depth']

    #tune min_samples_leaf
    param_test = {'min_samples_leaf':range(1,81,10)}
    gsearch = GridSearchCV(estimator = GradientBoostingRegressor(learning_rate=0.1, n_estimators=n_estimators, min_samples_split=min_samples_split,
                                                                max_depth=max_depth,max_features='sqrt', subsample=0.8, random_state=10),
                                        param_grid = param_test, n_jobs=4, iid=False, cv=5)
    gsearch.fit(X_train,Y_train.values.ravel())
    min_samples_leaf = gsearch.best_params_['min_samples_leaf']

    #check if min_samples_leaf is an extreme value. If so tune on a more accurate range
    if(min_samples_leaf == 80):
        param_test = {'min_samples_leaf':range(81,161,10)}
        gsearch = GridSearchCV(estimator = GradientBoostingRegressor(learning_rate=0.1, n_estimators=n_estimators, min_samples_split=min_samples_split,
                                                                max_depth=max_depth,max_features='sqrt', subsample=0.8, random_state=10),
                                        param_grid = param_test, n_jobs=4, iid=False, cv=5)
        gsearch.fit(X_train,Y_train.values.ravel())
        min_samples_leaf = gsearch.best_params_['min_samples_leaf']

    #tune max_features
    upper_bound = len(X_train.columns)
    lower_bound = int(upper_bound/2)
    step_size = int((upper_bound-lower_bound)/10)
    param_test = {'max_features':range(lower_bound,upper_bound+1,step_size)}#use more accurate range
    gsearch = GridSearchCV(estimator = GradientBoostingRegressor(learning_rate=0.1, n_estimators=n_estimators, min_samples_split=min_samples_split,
                                                                min_samples_leaf=min_samples_leaf,max_depth=max_depth, subsample=0.8, random_state=10),
                                        param_grid = param_test, n_jobs=4, iid=False, cv=5)
    gsearch.fit(X_train,Y_train.values.ravel())
    max_features = gsearch.best_params_['max_features']

    #check if max_features is an extreme value. If so tune on a more accurate range
    if(max_features == lower_bound):
        step_size = int(lower_bound/10)
        param_test = {'max_features':range(2,lower_bound,step_size)}#use more accurate range
        gsearch = GridSearchCV(estimator = GradientBoostingRegressor(learning_rate=0.1, n_estimators=n_estimators, min_samples_split=min_samples_split,
                                                                    min_samples_leaf=min_samples_leaf,max_depth=max_depth, subsample=0.8, random_state=10),
                                            param_grid = param_test, n_jobs=4, iid=False, cv=5)
        gsearch.fit(X_train,Y_train.values.ravel())
        max_features = gsearch.best_params_['max_features']

    #tune subsample
    param_test = {'subsample':[0.6,0.7,0.75,0.8,0.85,0.9,0.95,1]}
    gsearch = GridSearchCV(estimator = GradientBoostingRegressor(learning_rate=0.1, n_estimators=n_estimators, min_samples_split=min_samples_split,
                                                                min_samples_leaf=min_samples_leaf,max_depth=max_depth,
                                                                 max_features=max_features, random_state=10),
                                        param_grid = param_test, n_jobs=4, iid=False, cv=5)
    gsearch.fit(X_train,Y_train.values.ravel())
    subsample = gsearch.best_params_['subsample']

    print("n_estimators:",n_estimators)
    print("min_samples_split:",min_samples_split)
    print("max_depth:",max_depth)
    print("min_samples_leaf:",min_samples_leaf)
    print("max_features:",max_features)
    print("subsample:",subsample)

    #proportionally decrease learning rate and increase # of estimators. Pick th bst combo
    model = GradientBoostingRegressor(learning_rate=0.01, n_estimators=n_estimators*10, min_samples_split=min_samples_split,
                min_samples_leaf=min_samples_leaf,max_depth=max_depth,max_features=max_features,subsample=subsample, random_state=10)
    model.fit(X_train,Y_train.values.ravel())
    score = model.score(X_test,Y_test)

    model2 = GradientBoostingRegressor(learning_rate=0.05, n_estimators=n_estimators*2, min_samples_split=min_samples_split,
                min_samples_leaf=min_samples_leaf,max_depth=max_depth,max_features=max_features,subsample=subsample, random_state=10)
    model2.fit(X_train,Y_train.values.ravel())
    score2 = model2.score(X_test,Y_test)
    if(score2 > score):
        score = score2
        model = model2


    model3 = GradientBoostingRegressor(learning_rate=0.1, n_estimators=n_estimators, min_samples_split=min_samples_split,
                min_samples_leaf=min_samples_leaf,max_depth=max_depth,max_features=max_features,subsample=subsample, random_state=10)
    model3.fit(X_train,Y_train.values.ravel())
    score3 = model3.score(X_test,Y_test)
    if(score3 > score):
        score = score3
        model = model3

    model4 = GradientBoostingRegressor(learning_rate=0.2, n_estimators=int(n_estimators/2), min_samples_split=min_samples_split,
                min_samples_leaf=min_samples_leaf,max_depth=max_depth,max_features=max_features,subsample=subsample, random_state=10)
    model4.fit(X_train,Y_train.values.ravel())
    score4 = model4.score(X_test,Y_test)
    if(score4 > score):
        model = model4

    #model is the tuned and trained GBM now
    r_squared = gmp.getModelPredictions(model,X_train,Y_train,X_test,Y_test,outputPredictions,"GBM")
    print("\nTuned GBM R_squared:", r_squared)
    
    #not tuned GBM
    model = GradientBoostingRegressor(learning_rate=0.01,n_estimators=1000,max_depth=7,random_state=10)
    r_squared,mse = gmp.getModelPredictions(model,X_train,Y_train,X_test,Y_test,outputPredictions,"Test")
    print("\nNot Tuned GBM R_squared:", r_squared)

    #0.510173664085 with learning_rate=0.01,n_estimators=1000
    #gets better with tuned n_estimators
    #gets WORSE with tuned max_depth
    #gets better with tuned max_features
    #gets worse with tuned subsample