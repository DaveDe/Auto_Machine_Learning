from sklearn.metrics import r2_score
import FormatDataset as fd

def getModelPredictions(model, X_train, Y_train, X_test, Y_test, outputPredictions, modelName):
    model.fit(X_train,Y_train.values.ravel())
    prediction = model.predict(X_test)
    r_squared = r2_score(Y_test,prediction)
    if(outputPredictions):
    	filename = modelName + "_Predictions"
    	fd.writeCSV(prediction,filename)
    return r_squared
