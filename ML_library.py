import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import ElasticNet
from sklearn.ensemble import RandomForestRegressor
import getModelPrediction as gmp

################### Read data and convert nominal features #################

#read input file if it's not empty. If it is then there are no nominal features
with open('input.txt', 'r') as f:
    dataset = f.readline().replace("\n","")
    nominal_features_labels = f.readline()
    if(len(nominal_features_labels) > 0):
        nominal_features_labels = nominal_features_labels.split(",")
        nominal_features_labels[-1] = nominal_features_labels[-1].replace("\n","")


#read data
data = pd.read_csv(dataset)
numberOfFeatures = len(data.T)-1
featureLabels = ["X"+str(i) for i in range(1,numberOfFeatures+1)]

#split into features and labels
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
model = LinearRegression()
r_squared = gmp.getModelPredictions(model,X_train,Y_train,X_test,Y_test)
print("\nLinear Regression R_squared:", r_squared)

#Polynomial Regression
poly = PolynomialFeatures(degree=2)
poly_X_train = poly.fit_transform(X_train)
poly_X_test = poly.fit_transform(X_test)
r_squared = gmp.getModelPredictions(model,poly_X_train,Y_train,poly_X_test,Y_test)
print("\nPolynomial Regression R_squared:", r_squared)

#ANN
model = MLPRegressor(hidden_layer_sizes=(100,),solver="lbfgs",activation="relu")
r_squared = gmp.getModelPredictions(model,X_train,Y_train,X_test,Y_test)
print("\nANN Regression R_squared:", r_squared)

#Elastic Net
model = ElasticNet(alpha=1, l1_ratio=0.7)
r_squared = gmp.getModelPredictions(model,X_train,Y_train,X_test,Y_test)
print("\nElastic Net Regression R_squared:", r_squared)

#Random Forest
model = RandomForestRegressor()
r_squared = gmp.getModelPredictions(model,X_train,Y_train,X_test,Y_test)
print("\nRandom Forest Regression R_squared:", r_squared)
