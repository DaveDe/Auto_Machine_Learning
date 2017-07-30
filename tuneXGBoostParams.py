from sklearn.model_selection import GridSearchCV
import xgboost as xgb
from xgboost.sklearn import XGBRegressor

def getTunedXGBoostModel(X_train,Y_train,X_test,Y_test):
    #tune number of estimators (decision trees)
    param_test = {'n_estimators':range(20,101,10)}

    gsearch = GridSearchCV(estimator = XGBRegressor(learning_rate=0.1, random_state=10),
                                        param_grid = param_test, n_jobs=4, iid=False, cv=5)
    gsearch.fit(X_train,Y_train.values.ravel())
    n_estimators = gsearch.best_params_['n_estimators']

    #tune min_child_weight
    param_test = {'min_child_weight':range(1,12,1)}

    gsearch = GridSearchCV(estimator = XGBRegressor(learning_rate=0.1,n_estimators=n_estimators,random_state=10),
                                        param_grid = param_test, n_jobs=4, iid=False, cv=5)
    gsearch.fit(X_train,Y_train.values.ravel())
    min_child_weight = gsearch.best_params_['min_child_weight']
    
    #tune max_depth
    param_test = {'max_depth':range(3,12,1)}

    gsearch = GridSearchCV(estimator = XGBRegressor(learning_rate=0.1,n_estimators=n_estimators,
                                                    min_child_weight=min_child_weight,random_state=10),
                                        param_grid = param_test, n_jobs=4, iid=False, cv=5)
    gsearch.fit(X_train,Y_train.values.ravel())
    max_depth = gsearch.best_params_['max_depth']

    #tune gamma
    param_test = {'gamma':[0.0,0.1,0.2,0.3,0.4,0.5]}
    gsearch = GridSearchCV(estimator = XGBRegressor(learning_rate=0.1,
                                                    n_estimators=n_estimators,
                                                    min_child_weight=min_child_weight,
                                                    max_depth=max_depth,
                                                    random_state=10),
                                        param_grid = param_test, n_jobs=4, iid=False, cv=5)
    gsearch.fit(X_train,Y_train.values.ravel())
    gamma = gsearch.best_params_['gamma']

    #tune subsample
    param_test = {'subsample':[0.6,0.7,0.75,0.8,0.85,0.9,0.95,1.0]}
    gsearch = GridSearchCV(estimator = XGBRegressor(learning_rate=0.1,
                                                    n_estimators=n_estimators,
                                                    min_child_weight=min_child_weight,
                                                    max_depth=max_depth,
                                                    gamma=gamma,
                                                    random_state=10),
                                        param_grid = param_test, n_jobs=4, iid=False, cv=5)
    gsearch.fit(X_train,Y_train.values.ravel())
    subsample = gsearch.best_params_['subsample']

    #tune colsample_bytree
    param_test = {'colsample_bytree':[0.6,0.65,0.7,0.75,0.8,0.85,0.9,0.95,1]}
    gsearch = GridSearchCV(estimator = XGBRegressor(learning_rate=0.1,
                                                    n_estimators=n_estimators,
                                                    min_child_weight=min_child_weight,
                                                    max_depth=max_depth,
                                                    gamma=gamma,
                                                    subsample=subsample,
                                                    random_state=10),
                                        param_grid = param_test, n_jobs=4, iid=False, cv=5)
    gsearch.fit(X_train,Y_train.values.ravel())
    colsample_bytree = gsearch.best_params_['colsample_bytree']

    param_test = {'reg_alpha':[1e-5, 1e-2, 0.1, 1, 100]}
    gsearch = GridSearchCV(estimator = XGBRegressor(learning_rate=0.1,
                                                    n_estimators=n_estimators,
                                                    min_child_weight=min_child_weight,
                                                    max_depth=max_depth,
                                                    gamma=gamma,
                                                    subsample=subsample,
                                                    colsample_bytree=colsample_bytree,
                                                    random_state=10),
                                        param_grid = param_test, n_jobs=4, iid=False, cv=5)
    gsearch.fit(X_train,Y_train.values.ravel())
    reg_alpha = gsearch.best_params_['reg_alpha']

    param_test = {'reg_lambda':[1e-5, 1e-2, 0.1, 1, 100]}
    gsearch = GridSearchCV(estimator = XGBRegressor(learning_rate=0.1,
                                                    n_estimators=n_estimators,
                                                    min_child_weight=min_child_weight,
                                                    max_depth=max_depth,
                                                    gamma=gamma,
                                                    subsample=subsample,
                                                    colsample_bytree=colsample_bytree,
                                                    reg_alpha=reg_alpha,
                                                    random_state=10),
                                        param_grid = param_test, n_jobs=4, iid=False, cv=5)
    gsearch.fit(X_train,Y_train.values.ravel())
    reg_lambda = gsearch.best_params_['reg_lambda']

    print("\n----------------------------------")
    print("Tuned XGBoost params:")
    print("n_estimators:",n_estimators)
    print("min_child_weight:",min_child_weight)
    print("max_depth:",max_depth)
    print("gamma:",gamma)
    print("subsample:",subsample)
    print("colsample_bytree:",colsample_bytree)
    print("reg_alpha",reg_alpha)
    print("reg_lambda",reg_lambda)
    print("----------------------------------\n")


    #proportionally decrease learning rate and increase # of estimators. Pick th bst combo
    model = XGBRegressor(learning_rate=0.01, 
                        n_estimators=n_estimators*10, 
                        min_child_weight=min_child_weight,
                        max_depth=max_depth,
                        gamma=gamma,
                        subsample=subsample,
                        colsample_bytree=colsample_bytree,
                        reg_alpha=reg_alpha,
                        reg_lambda=reg_lambda,
                        random_state=10)
    model.fit(X_train,Y_train.values.ravel())
    score = model.score(X_test,Y_test)

    model2 = XGBRegressor(learning_rate=0.05, 
                        n_estimators=n_estimators*2, 
                        min_child_weight=min_child_weight,
                        max_depth=max_depth,
                        gamma=gamma,
                        subsample=subsample,
                        colsample_bytree=colsample_bytree,
                        reg_alpha=reg_alpha,
                        reg_lambda=reg_lambda,
                        random_state=10)
    model2.fit(X_train,Y_train.values.ravel())
    score2 = model2.score(X_test,Y_test)
    if(score2 > score):
        score = score2
        model = model2


    model3 = XGBRegressor(learning_rate=0.1, 
                        n_estimators=n_estimators, 
                        min_child_weight=min_child_weight,
                        max_depth=max_depth,
                        gamma=gamma,
                        subsample=subsample,
                        colsample_bytree=colsample_bytree,
                        reg_alpha=reg_alpha,
                        reg_lambda=reg_lambda,
                        random_state=10)
    model3.fit(X_train,Y_train.values.ravel())
    score3 = model3.score(X_test,Y_test)
    if(score3 > score):
        score = score3
        model = model3

    model4 = XGBRegressor(learning_rate=0.2, 
                        n_estimators=int(n_estimators/2), 
                        min_child_weight=min_child_weight,
                        max_depth=max_depth,
                        gamma=gamma,
                        subsample=subsample,
                        colsample_bytree=colsample_bytree,
                        reg_alpha=reg_alpha,
                        reg_lambda=reg_lambda,
                        random_state=10)
    model4.fit(X_train,Y_train.values.ravel())
    score4 = model4.score(X_test,Y_test)
    if(score4 > score):
        model = model4

    return model