import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import processData
import tensorflow_text
import embeddingModels
import hypermodelSelection

def main():
    embeddingArray, latitud_estandarizada, longitud_estandarizada, modelEncoded, excelNumRings = loadExcelData()
    
    hypermodelSelection.getNumRingsConcatenatedTuner(embeddingArray, latitud_estandarizada, longitud_estandarizada,
                                     modelEncoded, excelNumRings, 'bayesian_numRings_coordinatesWeighted_kerasLastVersion')    

def loadExcelData():
    excelFile = pd.read_excel('Excels/ScenariosLayers_minus1000Width.xlsx')
    excelLayers = np.array(excelFile['Layer'])
    excelLongitud = np.array(excelFile['Longitud'])
    excelLatitud = np.array(excelFile['Latitud'])
    excelWidths = processData.getRingsWidth(excelFile)
    excelResistances = processData.getRingsResistances(excelFile)
    excelModel = np.array(excelFile['Model'])
    excelNumRings = np.array(excelFile['NumRings'])
    
    multilingual_result = embeddingModels.loadEmbeddingModel(modelName='EmbeddingModel/model_large_minus1000Width.pt')
    embeddingArray = processData.getArray(multilingual_result)

    # Aplicar la estandarización
    latitud_estandarizada = processData.loadStandarization(excelLatitud.reshape(-1, 1), 'scalerLatitude')
    longitud_estandarizada = processData.loadStandarization(excelLongitud.reshape(-1, 1), 'scalerLongitude')

    # Operaciones con el modelo de pathfinder
    modelEncoded = processData.encodeText(excelModel)

    #embeddingArray = processData.addColumnToArray(embeddingArray, longitud_estandarizada)
    #embeddingArray = processData.addColumnToArray(embeddingArray, latitud_estandarizada)
    #embeddingArray = processData.addColumnToArray(embeddingArray, modelEncoded)
        
    return embeddingArray, latitud_estandarizada, longitud_estandarizada, modelEncoded, excelNumRings
