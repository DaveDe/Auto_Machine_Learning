import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import ElasticNetCV
from sklearn.ensemble import RandomForestRegressor
import getModelPrediction as gmp
import handleInput as hi

################### Read data and convert nominal features #################


with open('input.txt', 'r') as f:
    chosenAlgorithms, nominal_features_labels, data = hi.returnInputFileInfo(f)

featureLabels = [x for x in data[0,:] if x != "Y"]
data = pd.DataFrame(data)
X = data[featureLabels]
Y = data['Y']

#convert nominal features into dummy matricies
if(len(nominal_features_labels) > 0):
    for f in nominal_features_labels:
        dummy_matrix = pd.get_dummies(X[f])
        X = pd.concat([X,dummy_matrix], axis=1)
    X.drop(nominal_features_labels, axis=1, inplace = True)


##################### Begin Feature Selection #####################



################## Split data into train and test ###################

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.33)

################### Begin ML Regression Algorithms ###################

#Linear Regression
if(perfLinearRegression):
    model = LinearRegression()
    r_squared = gmp.getModelPredictions(model,X_train,Y_train,X_test,Y_test)
    print("\nLinear Regression R_squared:", r_squared)

#Polynomial Regression
if(perfPolynomialRegression):
    poly = PolynomialFeatures(degree=2)
    poly_X_train = poly.fit_transform(X_train)
    poly_X_test = poly.fit_transform(X_test)
    r_squared = gmp.getModelPredictions(model,poly_X_train,Y_train,poly_X_test,Y_test)
    print("\nPolynomial Regression R_squared:", r_squared)

#ANN
if(perfANN):
    model = MLPRegressor(hidden_layer_sizes=(100,),solver="lbfgs",activation="relu")
    r_squared = gmp.getModelPredictions(model,X_train,Y_train,X_test,Y_test)
    print("\nANN Regression R_squared:", r_squared)

#Elastic Net
if(perfElasticNet):
    model = ElasticNetCV(l1_ratio=[.1, .5, .7, .9, .95, .99, .995, 1], eps=0.001, n_alphas=100, fit_intercept=True,
                            normalize=True, precompute='auto', max_iter=2000, tol=0.0001, cv=5,
                            copy_X=True, verbose=0, n_jobs=-1, positive=False, random_state=None, selection='cyclic')
    r_squared = gmp.getModelPredictions(model,X_train,Y_train,X_test,Y_test)
    print("\nElastic Net Regression R_squared:", r_squared)

#Random Forest
if(perfRandomForest):
    model = RandomForestRegressor()
    r_squared = gmp.getModelPredictions(model,X_train,Y_train,X_test,Y_test)
    print("\nRandom Forest Regression R_squared:", r_squared)
