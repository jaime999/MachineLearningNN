import keras_tuner

def getBayesianTuner(hypermodel, objective, seed, max_trials, project_name):
    return keras_tuner.BayesianOptimization(hypermodel,
                                             objective=objective,
                                             seed=seed,
                                             max_trials=max_trials,
                                             directory='my_dir',
                                             project_name=project_name)

def getHyperbandTuner(hypermodel, objective, max_epochs, factor, project_name):
    return keras_tuner.Hyperband(hypermodel,
                          objective=objective,
                          max_epochs=max_epochs,
                          factor=factor,
                          directory='my_dir',
                          project_name=project_name)