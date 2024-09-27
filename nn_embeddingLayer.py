import numpy as np
import tensorflow_text
import tensorflow as tf
from tensorflow.keras.preprocessing.text import one_hot
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Flatten, Embedding, Input, Dropout, Concatenate
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.callbacks import ModelCheckpoint
from sklearn.model_selection import train_test_split
import os
import tensorflow_hub as hub
import pandas as pd
import keras_tuner
from tensorflow.keras.models import load_model 
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder, StandardScaler
import matplotlib.pyplot as plt
import sklearn.metrics, math
import random
import seaborn as sns
os.environ["CUDA_VISIBLE_DEVICES"]=""

# Configuración para utilizar la CPU
tf.config.set_visible_devices([], 'GPU')

def getNormalization(values, minValue, maxValue):
    return (values - minValue) / (maxValue - minValue)

def standarizeValues(values):
    # Calcular la media y la desviación estándar
    meanValue = np.mean(values)
    deviationValue = np.std(values)
    
    return (values - meanValue) / deviationValue

def getRingsResistances(excelFile):
    rings = pd.DataFrame()
    for column in excelFile.columns:
        if column.startswith("Ring"):
            rings[column] = excelFile[column]
            
    return np.array(rings)

excelFile = pd.read_excel('ScenariosLayers_2.xlsx')
excelLayers = np.array(excelFile['Layer'])
excelResistances = np.array(excelFile['Ring0'])
excelLongitud = np.array(excelFile['Longitud'])
excelLatitud = np.array(excelFile['Latitud'])
excelModel = np.array(excelFile['Model'])
excelNumRings = np.array(excelFile['NumRings'])
excelResistances = getRingsResistances(excelFile)

embed_normal = hub.KerasLayer("https://www.kaggle.com/models/google/universal-sentence-encoder/frameworks/TensorFlow2/variations/multilingual/versions/2",
                 input_shape=[],
                 output_shape=[512],
                 #dtype=tf.string,
                 name = 'universal_sentence_encoder_multi')

embed_normal_train = hub.KerasLayer("https://www.kaggle.com/models/google/universal-sentence-encoder/frameworks/TensorFlow2/variations/multilingual/versions/2",
                 input_shape=[],
                 dtype=tf.string,
                 trainable=True,                   
                 name = 'universal_sentence_encoder_multi')

embed_large = hub.KerasLayer("https://www.kaggle.com/models/google/universal-sentence-encoder/frameworks/TensorFlow2/variations/multilingual-large/versions/2",
                 input_shape=[],
                 dtype=tf.string,
                 name = 'universal_sentence_encoder_multi_large')

embed_qa = hub.KerasLayer("https://www.kaggle.com/models/google/universal-sentence-encoder/frameworks/TensorFlow2/variations/multilingual-qa/versions/2",
                 input_shape=[],
                 dtype=tf.string,
                 name = 'universal_sentence_encoder_multi_qa')

# Normalización min-max
latitud_normalizada = getNormalization(excelLatitud, excelLatitud.min(), excelLatitud.max())
longitud_normalizada = getNormalization(excelLongitud, excelLongitud.min(), excelLongitud.max())

# Aplicar la estandarización
latitud_estandarizada = standarizeValues(excelLatitud)
longitud_estandarizada = standarizeValues(excelLongitud)

# Operaciones con el modelo de pathfinder
le = LabelEncoder()
le.fit(excelModel)
modelEncoded = le.transform(excelModel)

scaler = StandardScaler()
scaler.fit(modelEncoded.reshape(-1,1))
modelStandarized = scaler.transform(modelEncoded.reshape(-1,1)).flatten()

# Operaciones con el número de anillos
numRings_normalizado = getNormalization(excelNumRings, excelNumRings.min(), excelNumRings.max())
numRings_estandarizado = standarizeValues(excelNumRings)

#X_train, X_test, y_train, y_test = train_test_split(excelLayers, excelResistances, test_size=0.2, random_state=42)
(X_layer_train, X_layer_test, X_latitude_train, X_latitude_test, X_longitude_train,
 X_longitude_test, X_model_train, X_model_test, y_train, y_test) = train_test_split(
    excelLayers, latitud_estandarizada, longitud_estandarizada, modelEncoded, excelResistances, test_size=0.2, random_state=42)

class MyHyperModel(keras_tuner.HyperModel):
    def build(self, hp):    
        # Entradas
        text_input_aux = Input(shape=(1,), dtype=tf.float32, name='layer')
        latitude_input = Input(shape=(1,), name='latitude')
        longitude_input = Input(shape=(1,), name='longitude')
        model_input = Input(shape=(1,), name='model')
        
        text_input = tf.squeeze(text_input_aux, axis=1)

        # embed_normal = hub.KerasLayer("https://www.kaggle.com/models/google/universal-sentence-encoder/frameworks/TensorFlow2/variations/multilingual/versions/2",
        #                  input_shape=[],
        #                  output_shape=[512],
        #                  #dtype=tf.string,
        #                  name = 'universal_sentence_encoder_multi')(text_input)
        embed_large = hub.KerasLayer("https://www.kaggle.com/models/google/universal-sentence-encoder/frameworks/TensorFlow2/variations/multilingual-large/versions/2",
                         input_shape=[],
                         output_shape=[512],
                         #dtype=tf.string,
                         name = 'universal_sentence_encoder_multi_large')(text_input)
        # Capa de embedding preentrenada

        flattened_embedding_layer = Flatten()(embed_large)
        # Concatenar las entradas de texto con los valores float
        concat_layer = Concatenate()([flattened_embedding_layer, latitude_input, longitude_input,
                                      model_input])
        # modelLayers.append(tf.keras.layers.Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=(512, 1)))
        # modelLayers.append(tf.keras.layers.Flatten())
        # if hp.Boolean("flatten"):
        #     modelLayers.append(tf.keras.layers.Flatten(input_shape=[512]))
        for i in range(hp.Int("num_layers", 1, 3)):
            concat_layer = Dense(
                    # Tune number of units separately.
                    units=hp.Int(f"units_{i}", min_value=32, max_value=512, step=32),
                    activation=hp.Choice(f"activation_{i}", ["relu", "sigmoid"]),
                    input_shape=[515]
                )(concat_layer)
            
            if hp.Boolean(f"dropout_{i}"):
                concat_layer = Dropout(rate=hp.Float(f'dropout_rate_{i}', min_value=0.1, max_value=0.25,step=0.05))(concat_layer)
        
        output_layer = Dense(10, activation="linear", name='output')(concat_layer)
        #modelLayers.append(Dense(1, activation=hp.Choice("activation_out", ["relu", "linear"])))
        #modelLayers.append(Dense(1, activation='sigmoid'))
        #model = Sequential(modelLayers)
        model = Model(inputs=[text_input, latitude_input, longitude_input, model_input], outputs=output_layer)
        learning_rate = hp.Float("lr", min_value=0.001, max_value=0.1, sampling="log")
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
            loss="mean_squared_error",
            metrics=['mean_absolute_error'],
        )
        return model
    
    def fit(self, hp, model, *args, **kwargs):
        return model.fit(
            *args,
            batch_size=hp.Choice("batch_size", [16, 32, 64]),
            **kwargs,
        )
    

def savePredictionsToExcel(X_layer_test, X_latitude_test, X_longitude_test, X_model_test, y_pred, y_test, excelTitle):
    # Crear un DataFrame con las predicciones y las entradas
    data = {'Layer': X_layer_test, 'Latitude': X_latitude_test, 'Longitude': X_longitude_test.flatten(),
            'Model': X_model_test}
    for ringIndex in range(10):
        data[f'RingPrediction{ringIndex}'] = y_pred[:, ringIndex]
        data[f'RingActual{ringIndex}'] = y_test[:, ringIndex]
    
    df = pd.DataFrame(data)

    # Escribir el DataFrame en un archivo de Excel
    df.to_excel(excelTitle, index=False)


# tuner = keras_tuner.Hyperband(MyHyperModel(),
#                      #objective=keras_tuner.Objective("val_r_square", direction="max"),
#                      objective='val_loss',
#                      max_epochs=10,
#                      factor=3,
#                      directory='my_dir',
#                      project_name='hyperband_embeddingLayer_rings')

tuner = keras_tuner.BayesianOptimization(MyHyperModel(),
                     #objective=keras_tuner.Objective("val_r_square", direction="max"),
                     objective='val_loss',
                     seed=42,
                     max_trials=10,
                     directory='my_dir',
                     project_name='bayesian_embeddingLayer_rings_modelLarge')

stop_early = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)


# tuner = keras_tuner.RandomSearch(
#     hypermodel=build_model,
#     objective="val_loss",
#     #objective=keras_tuner.Objective("val_mean_absolute_error", direction="min"),
#     max_trials=10,
#     executions_per_trial=2,
#     overwrite=True,
#     directory="my_dir",

#     project_name="embeddingInput",
# )
tuner.search_space_summary()

#X_expanded = tf.expand_dims(X_train, axis=-1)
tuner.search([X_layer_train, X_latitude_train, X_longitude_train, X_model_train], y_train, epochs=10, validation_split=0.2, callbacks=[stop_early])

models = tuner.get_best_models(num_models=2)
best_model = models[0]
best_model.summary()

bestModelName = 'best_model_bayesian_embeddingLayer_rings.hdf5'
checkpointer = ModelCheckpoint(filepath=bestModelName, monitor='val_loss',
                               verbose=1, save_best_only=True)

# Get the top 2 hyperparameters.
best_hps = tuner.get_best_hyperparameters(5)
best_hp = tuner.get_best_hyperparameters()[0]
print(best_hp.values)
# Build the model with the best hp.
hypermodel = MyHyperModel()
model = hypermodel.build(best_hp)
hypermodel.fit(best_hp, model, [X_layer_train, X_latitude_train, X_longitude_train, X_model_train], y_train, validation_split=0.2, callbacks=[checkpointer], epochs=20)

best_model = load_model(bestModelName,
       custom_objects={'KerasLayer':hub.KerasLayer})
test_results = best_model.evaluate([X_layer_test, X_latitude_test, X_longitude_test, X_model_test], y_test, verbose=1)
y_pred = best_model.predict([X_layer_test, X_latitude_test, X_longitude_test, X_model_test])
print(f'R score total: {sklearn.metrics.r2_score(y_test,y_pred)}')
for index in range(10):
    y_test_index = (y_test[:, index]).reshape(-1,1)
    y_pred_index = (y_pred[:, index]).reshape(-1,1)
    print(f'R score Ring {index}: {sklearn.metrics.r2_score(y_test_index,y_pred_index)}')

#y_pred = best_model.predict(X_test)
indices = random.sample(range(len(y_test)), 200)
def getLinearRegressionPlot(ring):    
    #y_test_sample = (y_test[indices][:, ring]).reshape(-1,1)
    #y_pred_sample = (y_pred[indices][:, ring]).reshape(-1,1)
    y_test_sample = (y_test[:, ring]).reshape(-1,1)
    y_pred_sample = (y_pred[:, ring]).reshape(-1,1)
    regressor = LinearRegression()  
    regressor.fit(y_test_sample, y_pred_sample)  
    y_fit_sample = regressor.predict(y_pred_sample)
    
    p1 = max(max(y_pred_sample), max(y_test_sample))
    p2 = min(min(y_pred_sample), min(y_test_sample))
    plt.plot([p1, p2], [p1, p2], 'b-', color='red', label= 'Linear regression')
    plt.scatter(y_test_sample, y_pred_sample, label='data')
    #plt.plot(y_pred_sample, y_fit_sample, color='red', linewidth=2, label = 'Linear regression') 
    plt.title('Linear Regression')
    plt.legend()
    plt.xlabel('observed')
    plt.ylabel('predicted')
    plt.show()
    
    print("\n")
    print("Mean absolute error (MAE):      %f" % sklearn.metrics.mean_absolute_error(y_test_sample,y_pred_sample))
    print("Mean squared error (MSE):       %f" % sklearn.metrics.mean_squared_error(y_test_sample,y_pred_sample))
    print("Root mean squared error (RMSE): %f" % math.sqrt(sklearn.metrics.mean_squared_error(y_test_sample,y_pred_sample)))
    print("R square (R^2):                 %f" % sklearn.metrics.r2_score(y_test_sample,y_pred_sample))

bordes_bins = [-0.25, 0, 0.25, 0.5, 0.75, 1, 1.25, 1.5, 1.75, 2, 2.25]
#bordes_bins = [-0.2, 0, 0.2, 0.4, 0.6, 0.8, 1, 1.2]
# Crear subplots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))

indices = random.sample(range(len(y_test)), 100)
y_test_sample = y_test[indices].reshape(-1,1)
y_pred_sample = y_pred[indices]
# Histograma de predicciones
ax1.hist(y_pred_sample, bins=bordes_bins, color='blue', alpha=0.7)
ax1.set_xlabel('Predicciones')
ax1.set_ylabel('Frecuencia')
ax1.set_title('Histograma de Predicciones')

# Histograma de valores reales
ax2.hist(y_test_sample, bins=bordes_bins, color='green', alpha=0.7)
ax2.set_xlabel('Valores Reales')
ax2.set_ylabel('Frecuencia')
ax2.set_title('Histograma de Valores Reales')

# Mostrar los histogramas
plt.tight_layout()
plt.show()

# Crear gráfico de densidad para predicciones
sns.kdeplot(y_pred_sample, color='blue', label='Predictions')

y_test_sample = y_test[indices]
# Crear gráfico de densidad para valores reales
sns.kdeplot(y_test_sample, color='green', label='Actual')

# Agregar etiquetas y título
plt.xlabel('Value')
plt.ylabel('Density')
plt.title('Comparison between predicted and actual values')

# Mostrar el gráfico
plt.legend()
plt.show()

text_input_aux = Input(shape=(1,), dtype=tf.string, name='layer')
latitude_input = Input(shape=(1,), name='latitude')
longitude_input = Input(shape=(1,), name='longitude')
model_input = Input(shape=(1,), name='model')

text_input = tf.squeeze(text_input_aux, axis=1)

# embed_normal = hub.KerasLayer("https://www.kaggle.com/models/google/universal-sentence-encoder/frameworks/TensorFlow2/variations/multilingual/versions/2",
#                  input_shape=[],
#                  output_shape=[512],
#                  #dtype=tf.string,
#                  name = 'universal_sentence_encoder_multi')(text_input)
embed_large = hub.KerasLayer("https://www.kaggle.com/models/google/universal-sentence-encoder/frameworks/TensorFlow2/variations/multilingual-large/versions/2",
                 input_shape=[],
                 output_shape=[512],
                 #dtype=tf.string,
                 name = 'universal_sentence_encoder_multi_large')(text_input)
# Capa de embedding preentrenada

flattened_embedding_layer = Flatten()(embed_large)
# Concatenar las entradas de texto con los valores float
concat_layer = Concatenate()([flattened_embedding_layer, latitude_input, longitude_input,
                              model_input])
# modelLayers.append(tf.keras.layers.Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=(512, 1)))
# modelLayers.append(tf.keras.layers.Flatten())
# if hp.Boolean("flatten"):
#     modelLayers.append(tf.keras.layers.Flatten(input_shape=[512]))
concat_layer = Dense(units=288,activation='sigmoid', input_shape=[515])(concat_layer)
concat_layer = Dense(units=192,activation='relu', input_shape=[515])(concat_layer)
concat_layer = Dense(units=512,activation='sigmoid', input_shape=[515])(concat_layer)
concat_layer = Dropout(rate=0.1)(concat_layer)
output_layer = Dense(10, activation="linear", name='output')(concat_layer)

model = Model(inputs=[text_input, latitude_input, longitude_input, model_input], outputs=output_layer)
model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.002),
            loss="mean_squared_error",
            metrics=['mean_absolute_error'],
)

bestModelName = 'bayesian_embeddingLayer_rings_modelLarge.hdf5'
checkpointer = ModelCheckpoint(filepath=bestModelName, monitor='val_loss',
                               verbose=1, save_best_only=True)

model.fit([X_layer_train, X_latitude_train, X_longitude_train, X_model_train], y_train, epochs=20, validation_split=0.2, batch_size=32, callbacks=[checkpointer])
# modelLayer.fit(X_train, y_train, epochs=2, validation_split=0.2, callbacks=[checkpointer])

# #best_model = load_model('best_model_layer.hdf5')
# test_results = modelLayer.evaluate(X_test, y_test, verbose=1)

# y_pred = modelLayer.predict(X_test)

# print the linear regression and display datapoints

regressor = LinearRegression()  
regressor.fit(y_test.reshape(-1,1), y_pred)  
y_fit = regressor.predict(y_pred) 

reg_intercept = round(regressor.intercept_[0],4)
reg_coef = round(regressor.coef_.flatten()[0],4)
reg_label = "y = " + str(reg_intercept) + "*x +" + str(reg_coef)

plt.scatter(y_test, y_pred, color='blue', label= 'data')
plt.plot(y_pred, y_fit, color='red', linewidth=2, label = 'Linear regression\n'+reg_label) 
plt.title('Linear Regression')
plt.legend()
plt.xlabel('observed')
plt.ylabel('predicted')
plt.show()

import sklearn.metrics, math
print("\n")
print("Mean absolute error (MAE):      %f" % sklearn.metrics.mean_absolute_error(y_test,y_pred))
print("Mean squared error (MSE):       %f" % sklearn.metrics.mean_squared_error(y_test,y_pred))
print("Root mean squared error (RMSE): %f" % math.sqrt(sklearn.metrics.mean_squared_error(y_test,y_pred)))
print("R square (R^2):                 %f" % sklearn.metrics.r2_score(y_test,y_pred))

def build_model(hp):    
    #model.add(layers.Flatten())        
    # Tune the number of layers.
    modelLayers = [embed_normal]
    # modelLayers.append(tf.keras.layers.Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=(512, 1)))
    # modelLayers.append(tf.keras.layers.Flatten())
    # if hp.Boolean("flatten"):
    #     modelLayers.append(tf.keras.layers.Flatten(input_shape=[512]))
    for i in range(hp.Int("num_layers", 1, 3)):
        modelLayers.append(tf.keras.layers.Dense(
                # Tune number of units separately.
                units=hp.Int(f"units_{i}", min_value=32, max_value=512, step=32),
                activation=hp.Choice("activation", ["relu", "tanh", "sigmoid"]),
                #input_shape=[512]
            ))
        
    if hp.Boolean("dropout"):
        modelLayers.append(tf.keras.layers.Dropout(rate=0.25))
    modelLayers.append(tf.keras.layers.Dense(1, activation=hp.Choice("activation", ["relu", "sigmoid"])))
    model = Sequential(modelLayers)
    learning_rate = hp.Float("lr", min_value=0.001, max_value=1, sampling="log")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss="mean_squared_error",
        metrics=['accuracy'],
    )
    return model

tuner = keras_tuner.RandomSearch(
    hypermodel=build_model,
    objective="val_loss",
    #objective=keras_tuner.Objective("val_mean_absolute_error", direction="min"),
    max_trials=3,
    executions_per_trial=2,
    overwrite=True,
    directory="my_dir",
    project_name="embeddingLayer",
)
tuner.search_space_summary()

# tuner.search(X_train, y_train, epochs=2, validation_split=0.33)

models = tuner.get_best_models(num_models=2)
best_model = models[0]
best_model.summary()

checkpointer = ModelCheckpoint(filepath='best_model_layer.hdf5', monitor='val_loss',
                               verbose=1, save_best_only=True)

# Get the top 2 hyperparameters.
best_hps = tuner.get_best_hyperparameters(5)
# Build the model with the best hp.
model = build_model(best_hps[0])
model.summary()
# model.fit(X_train, y_train, epochs=10, validation_split=0.33, callbacks=[checkpointer])

# #best_model = load_model('best_model.hdf5')

# print(f"Test evaluation: {model.evaluate(X_test, y_test)}")

# sentence_encoder_model_normal = tf.keras.Sequential([
#     embed_normal,
#     #Dense(128),
#     Dense(1)
# ])

# sentence_encoder_model_normal.compile(loss="mean_squared_error",
#                        optimizer=tf.keras.optimizers.Adam(),
#                        metrics=["accuracy"])

# sentence_encoder_model_normal.summary()

# sentence_encoder_model_normal_split = tf.keras.Sequential([
#     embed_normal,
#     #Dense(128),
#     Dense(1)
# ])

# sentence_encoder_model_normal_split.compile(loss="mean_squared_error",
#                        optimizer=tf.keras.optimizers.Adam(),
#                        metrics=["accuracy"])

# sentence_encoder_model_normal_split.summary()

# # Para que devuelva un valor positivo
# sentence_encoder_model_normal_relu_output = tf.keras.Sequential([
#     embed_normal,
#     #Dense(128),
#     Dense(1, activation='relu')
# ])

# sentence_encoder_model_normal_relu_output.compile(loss="mean_squared_error",
#                        optimizer=tf.keras.optimizers.Adam(),
#                        metrics=["accuracy"])

# sentence_encoder_model_normal_relu_output.summary()

# sentence_encoder_model_normal_2 = tf.keras.Sequential([
#     embed_normal,
#     Dense(128),
#     Dense(1)
# ])

# sentence_encoder_model_normal_2.compile(loss="mean_squared_error",
#                        optimizer=tf.keras.optimizers.Adam(),
#                        metrics=["accuracy"])

# sentence_encoder_model_normal_2.summary()

# sentence_encoder_model_normal_3 = tf.keras.Sequential([
#     embed_normal,
#     Dense(128, activation='relu'),
#     Dense(1)
# ])

# sentence_encoder_model_normal_3.compile(loss="mean_squared_error",
#                        optimizer=tf.keras.optimizers.Adam(),
#                        metrics=["accuracy"])

# sentence_encoder_model_normal_3.summary()

# sentence_encoder_history_normal = sentence_encoder_model_normal.fit(excelLayers, excelResistances_2, epochs=10, verbose=1)
