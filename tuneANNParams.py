from sklearn.model_selection import GridSearchCV
from sklearn.neural_network import MLPRegressor

def getTunedANNModel(X_train,Y_train,X_test,Y_test):

    #choose weight optimization solver
    solver = "adam"
    if(X_train.shape[0] < 1000):
        solver = "lbfgs"

    #tune number of nodes per layer, and number of layers
    param_test = {'hidden_layer_sizes':[(100,),(100,50),(50,),(100,100),(200,200),(200,100)]}
    gsearch = GridSearchCV(estimator = MLPRegressor(solver=solver,random_state=10),
                                        param_grid = param_test,iid=False, cv=5)
    gsearch.fit(X_train,Y_train.values.ravel())
    hidden_layer_sizes = gsearch.best_params_['hidden_layer_sizes']

    #tune activation function
    param_test = {'activation':["identity", "logistic", "tanh", "relu"]}
    gsearch = GridSearchCV(estimator = MLPRegressor(hidden_layer_sizes=hidden_layer_sizes,
                                                    solver=solver,
                                                    random_state=10),
                                        param_grid = param_test,iid=False, cv=5)
    gsearch.fit(X_train,Y_train.values.ravel())
    activation = gsearch.best_params_['activation']

    #tune alpha (L2 penalty)
    param_test = {'alpha':[0,0.00001,0.0001,0.001,0.01,0.1,0.5,1]}
    gsearch = GridSearchCV(estimator = MLPRegressor(hidden_layer_sizes=hidden_layer_sizes,
                                                    activation=activation,
                                                    solver=solver,
                                                    random_state=10),
                                        param_grid = param_test,iid=False, cv=5)
    gsearch.fit(X_train,Y_train.values.ravel())
    alpha = gsearch.best_params_['alpha']

    print("\n----------------------------------")
    print("Tuned ANN params:")
    print("hidden_layer_sizes:",hidden_layer_sizes)
    print("activation:",activation)
    print("solver:",solver)
    print("alpha:",alpha)
    print("----------------------------------\n")

    model = MLPRegressor(hidden_layer_sizes=hidden_layer_sizes,
                        activation=activation,
                        solver=solver,
                        alpha=alpha,
                        random_state=10)

    return model
