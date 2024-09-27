import tensorflow as tf
from tensorflow.keras.models import Sequential
import keras_tuner
from tensorflow.keras.layers import Dense, Input, Dropout, Concatenate
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.models import load_model, Model
from sklearn.metrics import r2_score
import sklearn
import tuners
import os

os.environ["CUDA_VISIBLE_DEVICES"]=""

# Configuración para utilizar la CPU
tf.config.set_visible_devices([], 'GPU')


class BaseModel(keras_tuner.HyperModel):
    def build(self):
        raise NotImplementedError("Subclass must implement build method")

    def fit(self, hp, model, *args, **kwargs):
        return model.fit(
            *args,
            batch_size=hp.Choice("batch_size", [16, 32, 64]),
            **kwargs,
        )


class MyHyperModel(BaseModel):
    def __init__(self, inputShape, activationHiddenLayers, activationOutputLayer, loss, metrics, outputShape):
        self.inputShape = inputShape
        self.activationHiddenLayers = activationHiddenLayers
        self.activationOutputLayer = activationOutputLayer
        self.loss = loss
        self.metrics = metrics
        self.outputShape = outputShape

    def build(self, hp):
        # Tune the number of layers.
        modelLayers = []
        for i in range(hp.Int("num_layers", 1, 3)):
            modelLayers.append(Dense(
                # Tune number of units separately.
                units=hp.Int(f"units_{i}", min_value=32,
                             max_value=512, step=32),
                activation=hp.Choice(
                    f"activation_{i}", self.activationHiddenLayers),
                input_shape=[self.inputShape]
            ))

            if hp.Boolean(f"dropout_{i}"):
                modelLayers.append(Dropout(rate=hp.Float(
                    f'dropout_rate_{i}', min_value=0.1, max_value=0.25, step=0.05)))

        modelLayers.append(
            Dense(self.outputShape, activation=self.activationOutputLayer))
        model = Sequential(modelLayers)
        learning_rate = hp.Float(
            "lr", min_value=0.001, max_value=0.1, sampling="log")
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
            loss=self.loss,
            metrics=self.metrics

        )
        return model  
    
class MyHyperModelConcatenated(BaseModel):
    def __init__(self, inputShape, activationHiddenLayers, activationOutputLayer,
                 loss, metrics, outputShape, extraInputs, modifyInputPos):
        self.inputShape = inputShape
        self.activationHiddenLayers = activationHiddenLayers
        self.activationOutputLayer = activationOutputLayer
        self.loss = loss
        self.metrics = metrics
        self.outputShape = outputShape
        self.extraInputs = extraInputs
        self.modifyInputPos = modifyInputPos

    def build(self, hp):
        # Entradas
        text_input_aux = Input(shape=(512,), name='layer')
        latitude_input = Input(shape=(1,), name='latitude')
        longitude_input = Input(shape=(1,), name='longitude')
        model_input = Input(shape=(1,), name='model')
        
        # latitude_branch = Dense(
        #     units=hp.Int("units_latitude", min_value=32,
        #                  max_value=512, step=32),
        #     activation=hp.Choice("activation_latitude",
        #                          self.activationHiddenLayers)
        # )(latitude_input)

        # longitude_branch = Dense(
        #     units=hp.Int("units_longitude", min_value=32,
        #                  max_value=512, step=32),
        #     activation=hp.Choice("activation_longitude",
        #                          self.activationHiddenLayers)
        # )(longitude_input)
        
        inputs = [text_input_aux, latitude_input, longitude_input, model_input]
        # inputsConcatenated = [text_input_aux, latitude_branch, longitude_branch, model_input]
        for actualInput in self.extraInputs:
            inputs.append(actualInput)
            
        inputsConcatenated = []
        for index, actualInput in enumerate(inputs):
            if index == self.modifyInputPos:
                input_branch = Dense(
                    units=hp.Int("units_input", min_value=32,
                                  max_value=512, step=32),
                    activation=hp.Choice("activation_input",
                                          self.activationHiddenLayers)
                )(inputs[index])
                inputsConcatenated.append(input_branch)
                
            else:
                inputsConcatenated.append(actualInput)            
            
        concatenated = Concatenate()(inputsConcatenated)

        for i in range(hp.Int("num_layers", 1, 3)):
            concatenated = Dense(
                # Tune number of units separately.
                units=hp.Int(f"units_{i}", min_value=32,
                             max_value=512, step=32),
                activation=hp.Choice(
                    f"activation_{i}", self.activationHiddenLayers),
                input_shape=[self.inputShape]
            )(concatenated)

            if hp.Boolean(f"dropout_{i}"):
                concatenated = Dropout(rate=hp.Float(
                    f'dropout_rate_{i}', min_value=0.1, max_value=0.25, step=0.05))(concatenated)

        output_layer = Dense(self.outputShape, activation=self.activationOutputLayer,
                             name='output')(concatenated)
        model = Model(inputs=inputs, outputs=output_layer)
        learning_rate = hp.Float(
            "lr", min_value=0.001, max_value=0.1, sampling="log")
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
            loss=self.loss,
            metrics=self.metrics

        )
        return model


class NeuralNetworkProcess:
    def fitTunerModel(tuner, bestModelName, hypermodel, monitor, X_train, y_train):
        models = tuner.get_best_models(num_models=2)
        best_model = models[0]
        best_model.summary()

        checkpointer = ModelCheckpoint(filepath=bestModelName, monitor=monitor,
                                       verbose=1, save_best_only=True)

        # Get the top 2 hyperparameters.
        best_hp = tuner.get_best_hyperparameters()[0]
        print(best_hp.values)
        # Build the model with the best hp.
        model = hypermodel.build(best_hp)
        hypermodel.fit(best_hp, model, X_train, y_train,
                       validation_split=0.2, callbacks=[checkpointer], epochs=50)

    def fitAndEvaluateHyperparameters(tuner, objective, hypermodel, X_train, y_train, projectName, bestModelName):
        tuner.search_space_summary()
        stop_early = tf.keras.callbacks.EarlyStopping(
            monitor=objective, patience=5, restore_best_weights=True)
        tuner.search(X_train, y_train, epochs=50,
                     validation_split=0.2, callbacks=[stop_early])
        NeuralNetworkProcess.fitTunerModel(tuner, bestModelName, hypermodel, objective,
                                           X_train, y_train)

    def fitAndEvaluateConcatenatedModels(objective, hypermodel, projectName, X_train, y_train, bestModelName):
        tuner = tuners.getBayesianTuner(hypermodel, objective, 42, 10, projectName)

        NeuralNetworkProcess.fitAndEvaluateHyperparameters(tuner, objective, hypermodel,
                                                           X_train, y_train, projectName, bestModelName)

    def getNumRingsConcatenatedTuner(embeddingArray, latitud_estandarizada, longitud_estandarizada,
                                     modelEncoded, excelNumRings, projectName):
        (X_layer_train, X_layer_test, X_latitude_train, X_latitude_test, X_longitude_train,
         X_longitude_test, X_model_train, X_model_test, y_train, y_test) = train_test_split(
            embeddingArray, latitud_estandarizada, longitud_estandarizada, modelEncoded, excelNumRings, test_size=0.2, random_state=42)
        bestModelName = f'NNModel/best_model_{projectName}.keras'
        hypermodel = MyHyperModelConcatenated(inputShape=515, activationHiddenLayers=["relu", "sigmoid", "softmax"], activationOutputLayer='softmax',
                                  loss='sparse_categorical_crossentropy', metrics=['accuracy'], outputShape=10, extraInputs=[], modifyInputPos=-1)
        
        NeuralNetworkProcess.fitAndEvaluateConcatenatedModels('val_accuracy', hypermodel, projectName, [X_layer_train, X_latitude_train,
                                                                                   X_longitude_train, X_model_train], y_train, bestModelName)
        
        return NeuralNetworkProcess.evaluateClassificationModel(
                    bestModelName, 
      [X_layer_test, X_latitude_test, X_longitude_test, X_model_test], y_test)

    def getWidthsConcatenatedTuner(embeddingArray, latitud_estandarizada, longitud_estandarizada,
                                   modelEncoded, excelNumRings, excelWidths, projectName):
        (X_layer_train, X_layer_test, X_latitude_train, X_latitude_test, X_longitude_train,
         X_longitude_test, X_model_train, X_model_test, X_numRings_train, X_numRings_test, y_train, y_test) = train_test_split(
            embeddingArray, latitud_estandarizada, longitud_estandarizada, modelEncoded, excelNumRings, excelWidths, test_size=0.2, random_state=42)
        bestModelName = f'NNModel/best_model_{projectName}.keras'
        extraInputs = [Input(shape=(1,), name='numRings')]
        hypermodel = MyHyperModelConcatenated(inputShape=516, activationHiddenLayers=["relu", "sigmoid", "linear"], activationOutputLayer='linear',
                                  loss='mean_squared_error', metrics=['mean_absolute_error'], outputShape=5, extraInputs=extraInputs, modifyInputPos=4)
        
        NeuralNetworkProcess.fitAndEvaluateConcatenatedModels('val_loss', hypermodel, projectName, [X_layer_train, X_latitude_train,
                                                                                   X_longitude_train, X_model_train, X_numRings_train], y_train, bestModelName)
        
        return NeuralNetworkProcess.evaluateRegressionModel(
                    bestModelName, 
      [X_layer_test, X_latitude_test, X_longitude_test, X_model_test, X_numRings_test], y_test)

    def getResistancesConcatenatedTuner(embeddingArray, latitud_estandarizada, longitud_estandarizada,
                                        modelEncoded, excelNumRings, excelWidths, excelResistances, projectName):
        (X_layer_train, X_layer_test, X_latitude_train, X_latitude_test, X_longitude_train,
         X_longitude_test, X_model_train, X_model_test, X_numRings_train, X_numRings_test, X_widths_train, X_widths_test, y_train, y_test) = train_test_split(
            embeddingArray, latitud_estandarizada, longitud_estandarizada, modelEncoded, excelNumRings, excelWidths, excelResistances, test_size=0.2, random_state=42)
        bestModelName = f'NNModel/best_model_{projectName}.keras'
        extraInputs = [Input(shape=(1,), name='numRings'), Input(shape=(5,), name='widths')]
        hypermodel = MyHyperModelConcatenated(inputShape=521, activationHiddenLayers=["relu", "sigmoid"], activationOutputLayer='linear',
                                       loss='mean_squared_error', metrics=['mean_absolute_error'], outputShape=5, extraInputs=extraInputs, modifyInputPos=-1)
        NeuralNetworkProcess.fitAndEvaluateConcatenatedModels('val_loss', hypermodel, projectName, [X_layer_train, X_latitude_train,
                                                                                   X_longitude_train, X_model_train, X_numRings_train, X_widths_train], y_train, bestModelName)
        
        return NeuralNetworkProcess.evaluateRegressionModel(
                    bestModelName, 
      [X_layer_test, X_latitude_test, X_longitude_test, X_model_test, X_numRings_test, X_widths_test], y_test)

    def getNumRingsTuner(X_train, y_train, X_test, y_test, projectName):
        bestModelName = f'NNModel/best_model_{projectName}.keras'
        objective = 'val_accuracy'
        hypermodel = MyHyperModel(inputShape=515, activationHiddenLayers=["relu", "sigmoid", "softmax"], activationOutputLayer='softmax',
                                  loss='sparse_categorical_crossentropy', metrics=['accuracy'], outputShape=10)

        tuner = tuners.getBayesianTuner(
            hypermodel, objective, 42, 10, projectName)

        NeuralNetworkProcess.fitAndEvaluateHyperparameters(tuner, objective, hypermodel,
                                                           X_train, y_train, projectName, bestModelName)
        return NeuralNetworkProcess.evaluateClassificationModel(
            bestModelName, X_test, y_test)

    def getWidthsTuner(X_train, y_train, X_test, y_test, projectName):
        bestModelName = f'NNModel/best_model_{projectName}.keras'
        objective = 'val_loss'
        hypermodel = MyHyperModel(inputShape=516, activationHiddenLayers=["relu", "sigmoid", "linear"], activationOutputLayer='linear',
                                  loss='mean_squared_error', metrics=['mean_absolute_error'], outputShape=5)

        tuner = tuners.getBayesianTuner(
            hypermodel, objective, 42, 10, projectName)

        NeuralNetworkProcess.fitAndEvaluateHyperparameters(tuner, objective, hypermodel,
                                                           X_train, y_train, projectName, bestModelName)
        return NeuralNetworkProcess.evaluateRegressionModel(
            bestModelName, X_test, y_test)

    def getResistancesTuner(X_train, X_test, y_train, y_test, projectName):
        hypermodel = MyHyperModel(inputShape=521, activationHiddenLayers=["relu", "sigmoid"], activationOutputLayer='linear',
                                  loss='mean_squared_error', metrics=['mean_absolute_error'], outputShape=5)

        bestModelName = f'NNModel/best_model_{projectName}.keras'
        objective = 'val_loss'
        tuner = tuners.getBayesianTuner(
            hypermodel, objective, 42, 10, projectName)

        NeuralNetworkProcess.fitAndEvaluateHyperparameters(tuner, objective, hypermodel,
                                                           X_train, y_train, projectName, bestModelName)
        return NeuralNetworkProcess.evaluateRegressionModel(
            bestModelName, X_test, y_test)

    def evaluateClassificationModel(bestModelName, X_test, y_test):
        best_model = load_model(bestModelName)
        test_results = best_model.evaluate(X_test, y_test, verbose=1)

        y_pred = best_model.predict(X_test)
        # Obtener la clase con la probabilidad más alta como predicción final
        predictedNumRings = np.argmax(y_pred, axis=1)
        print(
            f'F1 score micro: {sklearn.metrics.f1_score(y_test,predictedNumRings,average="micro")}')
        print(
            f'F1 score macro: {sklearn.metrics.f1_score(y_test,predictedNumRings,average="macro")}')
        print(
            f'F1 score weighted: {sklearn.metrics.f1_score(y_test,predictedNumRings,average="weighted")}')

        return predictedNumRings

    def evaluateRegressionModel(bestModelName, X_test, y_test):
        best_model = load_model(bestModelName)
        test_results = best_model.evaluate(X_test, y_test, verbose=1)

        y_pred = best_model.predict(X_test)
        print(f'R score total: {r2_score(y_test,y_pred)}')
        for index in range(y_pred.shape[1]):
            y_test_index = (y_test[:, index]).reshape(-1, 1)
            y_pred_index = (y_pred[:, index]).reshape(-1, 1)
            print(
                f'R score Ring {index}: {r2_score(y_test_index,y_pred_index)}')

        return y_pred
