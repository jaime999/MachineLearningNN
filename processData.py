import numpy as np
import pandas as pd
from joblib import dump, load
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder, StandardScaler


def getMeanAndDeviation(values):
    # Calcular la media y la desviación estándar
    return np.mean(values), np.std(values)

def getNormalization(values, minValue, maxValue):
    return (values - minValue) / (maxValue - minValue)

def standarizeValues(values):
    meanValue, deviationValue = getMeanAndDeviation(values)
    
    return (values - meanValue) / deviationValue

def scikitStandarization(values, standarizationPath):
    scaler = StandardScaler()
    standarizedValues = scaler.fit_transform(values)
    
    # Guardar el scaler para usarlo en inferencia
    dump(scaler, f'{standarizationPath}.pkl')
    
    return standarizedValues

def loadPreprocessingValues(values, preprocessingTransformersPath):
    # Cargar el fichero de preprocesamiento
    preprocessingTransformer = load(f'{preprocessingTransformersPath}.pkl')
    
    return preprocessingTransformer.transform(values)

def getRingsResistances(excelFile):
    rings = pd.DataFrame()
    for column in excelFile.columns:
        if column.startswith("Ring"):
            rings[column] = excelFile[column]
            
    ringsArray = np.array(rings)
       
    return ringsArray[:, :5]

def getRingsWidth(excelFile):
    widths = pd.DataFrame()
    for column in excelFile.columns:
        if column.startswith("width"):
            widths[column] = excelFile[column] 
            
    widthsArray = np.array(widths)
       
    return widthsArray[:, :5]


def encodeText(text, encoderSavedPath):
    le = OrdinalEncoder(handle_unknown='use_encoded_value',
                                 unknown_value=-1)
    textEncoded = le.fit_transform(text)
    
    # Guardar el scaler para usarlo en inferencia
    dump(le, f'{encoderSavedPath}.pkl')
    
    return textEncoded

def oneHotEncoding(obj):
    onehotEncoderCountries = OneHotEncoder(sparse_output=False, categories='auto')
    obj = obj.reshape(len(obj), 1)
    
    return onehotEncoderCountries.fit_transform(obj)

def addColumnToArray(array, column):
    return np.c_[array, column]

def getArray(objToConvert):
    return np.array(objToConvert)

def createFunctionalInput(embeddingLayers, latitude, longitude, MCDAModel):
    return [processInputs(embeddingLayers, latitude, longitude, MCDAModel)]

def createSequentialInput(embeddingLayers, latitude, longitude, MCDAModel):
    embeddingArray, latitud_estandarizada, longitud_estandarizada, modelEncoded = processInputs(embeddingLayers, latitude,
                                                                                                longitude, MCDAModel)
    embeddingArray = addColumnToArray(embeddingArray, longitud_estandarizada)
    embeddingArray = addColumnToArray(embeddingArray, latitud_estandarizada)
    embeddingArray = addColumnToArray(embeddingArray, modelEncoded)
    
    return embeddingArray

def processInputs(embeddingLayers, latitude, longitude, MCDAModel):
    embeddingArray = np.array(embeddingLayers)

    # Aplicar la estandarización
    latitud_estandarizada = loadPreprocessingValues(latitude.reshape(-1, 1), 'scalerLatitude')
    longitud_estandarizada = loadPreprocessingValues(longitude.reshape(-1, 1), 'scalerLongitude')

    # Operaciones con el modelo de pathfinder
    modelEncoded = encodeText(MCDAModel, 'ordinalEncoderMCDAModel')
    
    return [embeddingArray, latitud_estandarizada, longitud_estandarizada, modelEncoded]
    
    