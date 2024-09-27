from tensorflow.keras.models import load_model
import numpy as np
import embeddingModels
import processData
import tensorflow_hub as hub

class UltimatePathfinderSuggestions:
    def __init__(self):
        self.embeddingModel = 'https://www.kaggle.com/models/google/universal-sentence-encoder/frameworks/TensorFlow2/variations/multilingual-large/versions/2'
        self.numRingsModel = 'NNModel/best_model_bayesian_numRings_2models_large_excelWidthsMinus1000.hdf5'
        self.bufferSizesModel = 'NNModel/best_model_bayesian_widths_2models_large_minus1000Width_5ringsMax.hdf5'
        self.resistancesModel = 'NNModel/best_model_bayesian_rings_2models_large_widthsStandarized_5ringsMax.hdf5'

    def BufferSuggestor(self, layers, latitude, longitude, MCDAModel):
        embeddingModel = hub.load(self.embeddingModel)
        numRingsModel = load_model(self.numRingsModel)
        embeddingLayers = embeddingModels.embed_text(layers, embeddingModel)
        modelInput = processData.createInputs(embeddingLayers, latitude, longitude, MCDAModel)
        numRingsOutput = numRingsModel.predict(modelInput)
        
        predictedNumRings = np.argmax(numRingsOutput, axis=1)
        modelInput = processData.addColumnToArray(modelInput, predictedNumRings)
        
        bufferSizesModel = load_model(self.bufferSizesModel)        
        bufferSizesOutput = bufferSizesModel.predict(modelInput)
        bufferSizesOutput = processData.addColumnToArray(bufferSizesOutput, predictedNumRings)
        
        return bufferSizesOutput
    
    def ResistanceSuggestor(self, layers, latitude, longitude, MCDAModel, widthsPredicted):
        embeddingModel = hub.load(self.embeddingModel)
        numRingsModel = load_model(self.numRingsModel)
        embeddingLayers = embeddingModels.embed_text(layers, embeddingModel)
        modelInput = processData.createInputs(embeddingLayers, latitude, longitude, MCDAModel)
        numRingsOutput = numRingsModel.predict(modelInput)
        
        # widths_standarized = processData.scikitStandarization(widthsPredicted, 'scalerWidths')
        widths_standarized = processData.loadStandarization(widthsPredicted, 'scalerWidths')
        modelInput = processData.addColumnToArray(modelInput, widths_standarized)
        
        predictedNumRings = np.argmax(numRingsOutput, axis=1)
        modelInput = processData.addColumnToArray(modelInput, predictedNumRings)
        
        resistancesModel = load_model(self.resistancesModel)        
        resistancesOutput = resistancesModel.predict(modelInput)
        resistancesOutput = processData.addColumnToArray(resistancesOutput, predictedNumRings)
        
        return resistancesOutput
        
        

