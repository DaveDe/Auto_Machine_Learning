#add boosting algorithms
#add feature selection methods (try pca and zca)

import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import scale
from sklearn.decomposition import PCA
from sklearn.linear_model import ElasticNetCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingRegressor
import xgboost as xgb
from xgboost.sklearn import XGBRegressor
import getModelPrediction as gmp
import handleInput as hi
import FormatDataset as fd
import tuneGBMParams as tgbm
import tuneXGBoostParams as txgb

with open('input.txt', 'r') as f:
    chosenAlgorithms,PCANum,perfCovariateShift, nominal_features_labels, trainData, testData, outputPredictions = hi.returnInputFileInfo(f)

X_train,X_test,Y_train,Y_test = fd.convertNominalToDummy(trainData,testData,nominal_features_labels)

#standardize data
X_train = pd.DataFrame(scale(X_train))
X_test = pd.DataFrame(scale(X_test))

perfLinearRegression = chosenAlgorithms[0]
perfPolynomialRegression = chosenAlgorithms[1]
perfANN = chosenAlgorithms[2]
perfElasticNet = chosenAlgorithms[3]
perfRandomForest = chosenAlgorithms[4]
perfGBM = chosenAlgorithms[5]
perfXGBoost = chosenAlgorithms[6]

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

##################### Feature Selection #####################

#transform X_train and X_test
if(PCANum != 0):
    print("Performing PCA...")
    pca = PCA(n_components=PCANum)
    pca.fit(X_train)
    X_train = pd.DataFrame(pca.transform(X_train))
    pca.fit(X_test)
    X_test = pd.DataFrame(pca.transform(X_test))

################### Begin ML Regression Algorithms ###################
print("Running algorithms...")

#Linear Regression
if(perfLinearRegression):
    model = LinearRegression()
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

    tunedGBMModel = tgbm.getTunedGBMModel(X_train,Y_train,X_test,Y_test)

    #model is the tuned and trained GBM now
    r_squared,mse = gmp.getModelPredictions(tunedGBMModel,X_train,Y_train,X_test,Y_test,outputPredictions,"GBM_tuned")
    print("\nTuned GBM R_squared:", r_squared)
    print("Tuned GBM MSE:", mse)

    #not tuned GBM
    model = GradientBoostingRegressor(random_state=10)
    r_squared,mse = gmp.getModelPredictions(model,X_train,Y_train,X_test,Y_test,outputPredictions,"GBM_untuned")
    print("\nNot Tuned GBM R_squared:", r_squared)
    print("Not Tuned GBM MSE:", mse)

if(perfXGBoost):

    tunedXGBoostModel = txgb.getTunedXGBoostModel(X_train,Y_train,X_test,Y_test)

    r_squared,mse = gmp.getModelPredictions(model,X_train,Y_train,X_test,Y_test,outputPredictions,"XGBoost")
    print("\nTuned XGBoost R_squared:", r_squared)
    print("Tuned XGBoost MSE:", mse)

    model = XGBRegressor(random_state=10)
    r_squared,mse = gmp.getModelPredictions(model,X_train,Y_train,X_test,Y_test,outputPredictions,"XGBoost")
    print("\nNot Tuned XGBoost R_squared:", r_squared)
    print("Not Tuned XGBoost MSE:", mse)