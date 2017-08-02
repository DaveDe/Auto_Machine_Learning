from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import GradientBoostingRegressor

def getTunedGBMModel(X_train,Y_train,X_test,Y_test):
    
    #tune number of estimators (decision trees)
    param_test = {'n_estimators':range(20,101,10)}

    gsearch = GridSearchCV(estimator = GradientBoostingRegressor(learning_rate=0.1, random_state=10),
                                        param_grid = param_test, n_jobs=4, iid=False, cv=5)
    gsearch.fit(X_train,Y_train.values.ravel())
    n_estimators = gsearch.best_params_['n_estimators']

    #check if n_estimators is 20 or 80, if so check more lower or higher values in another grid search
    if(n_estimators == 20):
        param_test = {'n_estimators':range(5,20,2)}
        gsearch = GridSearchCV(estimator = GradientBoostingRegressor(learning_rate=0.1, random_state=10),
                                        param_grid = param_test, n_jobs=4, iid=False, cv=5)
        gsearch.fit(X_train,Y_train.values.ravel())
        n_estimators = gsearch.best_params_['n_estimators']

    if(n_estimators == 100):
        param_test = {'n_estimators':range(100,161,10)}
        gsearch = GridSearchCV(estimator = GradientBoostingRegressor(learning_rate=0.1, random_state=10),
                                        param_grid = param_test, n_jobs=4, iid=False, cv=5)
        gsearch.fit(X_train,Y_train.values.ravel())
        n_estimators = gsearch.best_params_['n_estimators']

    #tune min number of samples needed to split
    upper_limit = int(0.01*len(Y_train))
    step_size = int(upper_limit-2/10)
    param_test = {'min_samples_split':range(2,upper_limit+1,step_size)}
    gsearch = GridSearchCV(estimator = GradientBoostingRegressor(learning_rate=0.1, n_estimators=n_estimators, random_state=10),
                                                                param_grid = param_test, n_jobs=4, iid=False, cv=5)
    gsearch.fit(X_train,Y_train.values.ravel())
    min_samples_split = gsearch.best_params_['min_samples_split']

    #check if min_samples_split is an extreme value. If so tune on a more accurate range
    if(min_samples_split == upper_limit):
        step_size = int(upper_limit/2)
        param_test = {'min_samples_split':range(upper_limit, (upper_limit*10)+1,step_size)}
        gsearch = GridSearchCV(estimator = GradientBoostingRegressor(learning_rate=0.1, n_estimators=n_estimators,random_state=10),
                                                                    param_grid = param_test, n_jobs=4, iid=False, cv=5)
        gsearch.fit(X_train,Y_train.values.ravel())
        min_samples_split = gsearch.best_params_['min_samples_split']

    #tune max_depth
    param_test = {'max_depth':range(3,14,2)}
    gsearch = GridSearchCV(estimator = GradientBoostingRegressor(learning_rate=0.1, n_estimators=n_estimators,min_samples_split=min_samples_split,
                                                                random_state=10),
                                                                param_grid = param_test, n_jobs=4, iid=False, cv=5)
    gsearch.fit(X_train,Y_train.values.ravel())
    max_depth = gsearch.best_params_['max_depth']
    #check if max_depth is an extreme value. If so tune on a more accurate range
    if(max_depth == 13):
        param_test = {'max_depth':range(13,20,1)}
        gsearch = GridSearchCV(estimator = GradientBoostingRegressor(learning_rate=0.1, n_estimators=n_estimators,min_samples_split=min_samples_split,
                                                                random_state=10),
                                                                param_grid = param_test, n_jobs=4, iid=False, cv=5)
        gsearch.fit(X_train,Y_train.values.ravel())
        max_depth = gsearch.best_params_['max_depth']

    #tune min_samples_leaf
    param_test = {'min_samples_leaf':range(1,81,10)}
    gsearch = GridSearchCV(estimator = GradientBoostingRegressor(learning_rate=0.1, n_estimators=n_estimators, min_samples_split=min_samples_split,
                                                                max_depth=max_depth,random_state=10),
                                        param_grid = param_test, n_jobs=4, iid=False, cv=5)
    gsearch.fit(X_train,Y_train.values.ravel())
    min_samples_leaf = gsearch.best_params_['min_samples_leaf']

    #check if min_samples_leaf is an extreme value. If so tune on a more accurate range
    if(min_samples_leaf == 80):
        param_test = {'min_samples_leaf':range(81,161,10)}
        gsearch = GridSearchCV(estimator = GradientBoostingRegressor(learning_rate=0.1, n_estimators=n_estimators, min_samples_split=min_samples_split,
                                                                max_depth=max_depth, random_state=10),
                                        param_grid = param_test, n_jobs=4, iid=False, cv=5)
        gsearch.fit(X_train,Y_train.values.ravel())
        min_samples_leaf = gsearch.best_params_['min_samples_leaf']

    #tune max_features
    upper_bound = len(X_train.columns)
    lower_bound = int(upper_bound/2)
    step_size = int((upper_bound-lower_bound)/10)
    if((upper_bound - lower_bound) < 10):
        step_size = 1
    param_test = {'max_features':range(lower_bound,upper_bound+1,step_size)}#use more accurate range
    gsearch = GridSearchCV(estimator = GradientBoostingRegressor(learning_rate=0.1, n_estimators=n_estimators, min_samples_split=min_samples_split,
                                                                min_samples_leaf=min_samples_leaf,max_depth=max_depth, random_state=10),
                                        param_grid = param_test, n_jobs=4, iid=False, cv=5)
    gsearch.fit(X_train,Y_train.values.ravel())
    max_features = gsearch.best_params_['max_features']

    #check if max_features is an extreme value. If so tune on a more accurate range
    if(max_features == lower_bound):
        step_size = int(lower_bound/10)
        param_test = {'max_features':range(2,lower_bound,step_size)}#use more accurate range
        gsearch = GridSearchCV(estimator = GradientBoostingRegressor(learning_rate=0.1, n_estimators=n_estimators, min_samples_split=min_samples_split,
                                                                    min_samples_leaf=min_samples_leaf,max_depth=max_depth, random_state=10),
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

    print("\n----------------------------------")
    print("Tuned GBM params:")
    print("n_estimators:",n_estimators)
    print("min_samples_split:",min_samples_split)
    print("max_depth:",max_depth)
    print("min_samples_leaf:",min_samples_leaf)
    print("max_features:",max_features)
    print("subsample:",subsample)
    print("----------------------------------\n")


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

    return model
