from sklearn.metrics import r2_score

def getModelPredictions(model, X_train, Y_train, X_test, Y_test):
    model.fit(X_train,Y_train.values.ravel())
    prediction = model.predict(X_test)
    r_squared = r2_score(Y_test,prediction)
    return r_squared
