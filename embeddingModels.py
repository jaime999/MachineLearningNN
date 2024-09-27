from sentence_transformers import SentenceTransformer
import tensorflow_hub as hub
import numpy as np
import torch

SentenceTransformerModels = ['nli-distilroberta-base-v2', 'distiluse-base-multilingual-cased-v1',
                             'paraphrase-multilingual-MiniLM-L12-v2', 'paraphrase-multilingual-mpnet-base-v2']

TensorflowUrls = ['multilingual/versions/2', 'multilingual-large/versions/2', 'multilingual-qa/versions/2',
                  'cmlm-multilingual-base/versions/1']

# El embedding siempre debe ir en la carpeta EmbeddingModel
def saveTensorflowEmbeddingModel(text, tensorflowModelUrl='multilingual-large/versions/2', modelName='EmbeddingModel/model_large_noOw3.pt'):
    model = hub.load(f'https://www.kaggle.com/models/google/universal-sentence-encoder/frameworks/TensorFlow2/variations/{tensorflowModelUrl}')
    # Se separa el texto para que no haya problemas con la memoria
    chunkSize = 40000
    layerEmbedded = model(text[:chunkSize])
    if len(text) > chunkSize:
        for i in range(1, len(text), chunkSize):
            layerEmbedded2 = model(text[i*chunkSize:])
            layerEmbedded = np.concatenate((layerEmbedded, layerEmbedded2), axis=0)
       
    torch.save(layerEmbedded, modelName)
    
def saveSentenceTransformerEmbeddingModel(sentenceTransformerModel, text, modelName='EmbeddingModel/model_large_noOw3.pt'):
    model = SentenceTransformer(sentenceTransformerModel)
    sentence_transformers_embeddings = model.encode(text)
    torch.save(sentence_transformers_embeddings, modelName)
    
def loadEmbeddingModel(modelName='EmbeddingModel/model_large_noOw3.pt'):
    return torch.load(modelName)
    