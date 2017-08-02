from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestRegressor

def getTunedRandomForestModel(X_train,Y_train,X_test,Y_test):

    #tune number of trees
    param_test = {'n_estimators':range(10,201,10)}
    gsearch = GridSearchCV(estimator = RandomForestRegressor(random_state=10),
                                        param_grid = param_test,iid=False, cv=5)
    gsearch.fit(X_train,Y_train.values.ravel())
    n_estimators = gsearch.best_params_['n_estimators']

    #tune function to measure quality of split. Currently causes killed: 9 error
    """param_test = {'criterion':["mse", "mae"]}
    gsearch = GridSearchCV(estimator = RandomForestRegressor(n_estimators=n_estimators,
                                                            random_state=10),
                                        param_grid = param_test,iid=False, cv=5)
    gsearch.fit(X_train,Y_train.values.ravel())
    criterion = gsearch.best_params_['criterion']"""

    #max_features
    upper_bound = len(X_train.columns)
    lower_bound = int(upper_bound/2)
    step_size = int((upper_bound-lower_bound)/10)
    if((upper_bound - lower_bound) < 10):
        step_size = 1
    param_test = {'max_features':range(lower_bound,upper_bound+1,step_size)}
    gsearch = GridSearchCV(estimator = RandomForestRegressor(n_estimators=n_estimators,
                                                            random_state=10),
                                        param_grid = param_test, n_jobs=4, iid=False, cv=5)
    gsearch.fit(X_train,Y_train.values.ravel())
    max_features = gsearch.best_params_['max_features']

    #tune max_depth
    param_test = {'max_depth':range(1,30,2)}
    gsearch = GridSearchCV(estimator = RandomForestRegressor(n_estimators=n_estimators,
                                                            max_features=max_features,
                                                            random_state=10),
                                    param_grid = param_test, n_jobs=4, iid=False, cv=5)
    gsearch.fit(X_train,Y_train.values.ravel())
    max_depth = gsearch.best_params_['max_depth']

    #tune min_samples_split
    upper_limit = int(0.01*len(Y_train))
    step_size = int(upper_limit-2/10)
    param_test = {'min_samples_split':range(2,upper_limit+1,step_size)}
    gsearch = GridSearchCV(estimator = RandomForestRegressor(n_estimators=n_estimators,
                                                            max_features=max_features,
                                                            max_depth=max_depth,
                                                            random_state=10),
                                    param_grid = param_test, n_jobs=4, iid=False, cv=5)
    gsearch.fit(X_train,Y_train.values.ravel())
    min_samples_split = gsearch.best_params_['min_samples_split']


    print("\n----------------------------------")
    print("Tuned Random Forest params:")
    print("n_estimators:",n_estimators)
    print("max_features:",max_features)
    print("max_depth:",max_depth)
    print("min_samples_split:",min_samples_split)
    print("----------------------------------\n")

    model = RandomForestRegressor(
                        n_estimators=n_estimators,
                        max_features=max_features,
                        max_depth=max_depth,
                        min_samples_split=min_samples_split,
                        random_state=10)

    return model
