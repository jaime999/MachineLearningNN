import json
import pandas as pd
import os
import glob
import re


def getResistancesValues(vectorLayers, layersWidths):
    resistances = []
    for layer in vectorLayers:        
        layer['name'] = processLayerName(layer['name'])
        if layer['name'] in layersWidths.keys():
            if layer['rmode'] == 'AV':
                resistances.append(layer['resistance_value'])
    
            elif layer['rmode'] == 'PR':
                layerResistances = layer['rvalues']['resistance_values']
                layerName = layer['name']
                actualLayerWidth = layersWidths[layerName]
                # Se comprueba el tamaño de la lista de tamaños y de resistencias,
                # y se itera sobre la más corta            
                if len(actualLayerWidth) <= len(layerResistances):
                    for index, width in enumerate(actualLayerWidth):
                        resistances.append(layerResistances[index])
                        
                else:      
                    for index, resistance in enumerate(layerResistances):
                        resistances.append(resistance)

    return resistances, vectorLayers


def normalizeValue(value, minimum, rangeResistances):
    return (value - minimum) / rangeResistances


def removeBeforeString(text):
    colon = text.find(':')
    wordArcgis = 'arcgis pro'
    arcgis = text.find(wordArcgis)
    if colon == -1 and arcgis == -1:
        return text

    newText = text
    if colon != -1:
        newText = text[(colon+1):]

    if arcgis != -1:
        newText = text[(arcgis+len(wordArcgis)+1):]

    return newText


def layerToRemove(word):
    wordSplit = word.split()
    removeLayers = ['bufferedpath', 'optimalsiting', 'viewshed', 'raster', 'suitabilityclasses',
                    'clipped_corridor', 'ridgelines', '.geojson', '.gpkg', 'buffered_path']

    return word.replace(' ', '') in removeLayers or 'dem' in wordSplit or 'slope' in word.split()


def processLayerName(layer):
    layerName = layer.lstrip('0123456789.- ').lower()
    layerName = layerName.replace("_", " ")
    layerName = removeBeforeString(layerName)

    return layerName


def loadLayers(jsonPath, rasterLayers, layersWidths):
    jsonFile = open(jsonPath)
    scenariosDict = json.load(jsonFile)
    layerResistances = pd.DataFrame()
    model = scenariosDict['config']['mcda_class_model']
    if model == 'TernaMCDA' or model == 'SwissMCDA':
        return layerResistances

    vectorLayers = [layer for layer in scenariosDict['layers'] if layer['name'] not in rasterLayers and
                    (layer['rmode'] == 'PR' or layer['rmode'] == 'FB' or
                    (layer['rmode'] == 'AV' and layer['resistance_value'] != 0))]
    if len(vectorLayers) == 0:
        return layerResistances

    #print(jsonPath)
    resistances, vectorLayers = getResistancesValues(vectorLayers, layersWidths)
    if len(resistances) == 0:
        return layerResistances

    maxResistances, minResistance = max(resistances), min(resistances)
    rangeResistances = maxResistances - minResistance
    if rangeResistances == 0:
        return layerResistances

    for layer in vectorLayers:
        layerName = layer['name']
        if layerName in layersWidths:
            rvalues = layer['rvalues']
            ringsResistances = rvalues['resistance_values']
            # layerName = processLayerName(layer['name'])
            layerWidths = layersWidths[layerName]
            if len(layerName) > 0 and not layerToRemove(layerName):
                newLayer = {'Layer': layerName, 'Model': model}
                if layer['rmode'] == 'PR':                
                    # Se comprueba el tamaño de la lista de tamaños y de resistencias,
                    # y se itera sobre la más corta
                    listToIterate = layerWidths
                    if len(layerWidths) > len(ringsResistances):
                        listToIterate = ringsResistances
                            
                    newLayer['NumRings'] = len(listToIterate) - 1
                    for index, value in enumerate(listToIterate):
                        ringColumn = f'Ring{index}'
                        if rvalues['forbidden'][index]:
                            newLayer[ringColumn] = 2
    
                        else:
                            newLayer[ringColumn] = normalizeValue(
                                ringsResistances[index], minResistance, rangeResistances)
                            
                        newLayer[f'width{ringColumn}'] = layerWidths[index]
    
                else:
                    newLayer['NumRings'] = 0
                    newLayer['widthRing0'] = layerWidths[0]
                    if layer['rmode'] == 'FB':
                        newLayer['Ring0'] = 2
                    else:
                        newLayer['Ring0'] = normalizeValue(
                            layer['resistance_value'], minResistance, rangeResistances)
    
                layerResistances = pd.concat(
                    [layerResistances, pd.DataFrame([newLayer])], ignore_index=True)

    return layerResistances


def searchProjectLayers(layers):
    if not layers:
        return None, None
    rasterLayers = []
    layersWidth = {}
    # Se buscan las capas que tengan el tipo "RST", que indica que son raster
    for key, value in layers.items():
        if value['type'] == 'RST':
            rasterLayers.append(key)

        else:
            ringWidths = value['rings']['ring_widths']
            # Se comprueba que ningun tamaño supere los 1000
            excessSize = any(width > 1000 for width in ringWidths)
            if not excessSize:
                layerName = processLayerName(key)
                layersWidth[layerName] = value['rings']['ring_widths']

    return rasterLayers, layersWidth


def searchCoordinatesAndProjectLayers(actualPath, fileName):
    # Obtener la ruta de la carpeta padre
    pathFolder = os.path.dirname(actualPath)
    parentFolder = os.path.dirname(pathFolder)
    # Construir la ruta completa al archivo
    filePath = os.path.join(f'{parentFolder}/', fileName)
    # Verificar si el archivo existe
    if os.path.isfile(filePath):
        jsonFile = open(filePath)
        projectDict = json.load(jsonFile)
        geographicArea = projectDict['geographic_area']
        if not geographicArea:
            return None, None, None
        # Las coordenadas se encuentran con este patrón
        coordinatesPattern = r'\(\((.*?)\)\)'
        # Encontrar todas las coincidencias
        coincidences = re.findall(coordinatesPattern, geographicArea)
        # Se obtienen las 4 coordenadas en una lista
        coordinatesList = coincidences[0].split(',')
        longitudes = []
        latitudes = []
        for coordinates in coordinatesList:
            longitud, latitud = coordinates.split()
            longitudes.append(float(longitud))
            latitudes.append(float(latitud))

        coordinatesMean = sum(longitudes) / \
            len(longitudes), sum(latitudes) / len(latitudes)
        rasterLayers, layersWidths = searchProjectLayers(projectDict['layers'])

        return coordinatesMean, rasterLayers, layersWidths


directorio_raiz = '../pathfi_export_25_04_2024'
archivos_json = glob.glob(os.path.join(
    directorio_raiz, '**/scen_info.json'), recursive=True)
layerResistances = pd.DataFrame()
pathfinderCoordinates = pd.read_excel('Excels/PathfinderCoordinates.xlsx')
for jsonFile in archivos_json:
    #print(jsonFile)
    coordinates, rasterLayers, layersWidths = searchCoordinatesAndProjectLayers(
        jsonFile, 'proj_ctx.json')
    if coordinates:
        # Se calculan todas las resistencias del fichero, usando solo capas AV, PR o FB, y quitando
        # las que no sean necesarias
        layerResistance = loadLayers(jsonFile, rasterLayers, layersWidths)
        if not layerResistance.empty:
            layerResistance['Longitud'] = coordinates[0]
            layerResistance['Latitud'] = coordinates[1]
            # Se busca el país en el fichero de coordenadas, donde se encuentra toda su información
            country = pathfinderCoordinates.loc[(pathfinderCoordinates['Latitude'] == round(
                coordinates[1], 4)) & (pathfinderCoordinates['Longitude'] == round(coordinates[0], 4))]['Country']
            if len(country) > 1:
                raise Exception(
                    'Una coordenada no puede pertencer a más de 1 país', coordinates[1])

            #print(coordinates[1], coordinates[0])
            #layerResistance['Country'] = country.iloc[0]
            layerResistances = pd.concat(
                [layerResistances, layerResistance], ignore_index=True)

# Rellenar NaN con -1, ya que indica que no hay valor en ese anillo
layerResistances.fillna(-1, inplace=True)
print(layerResistances)


def saveExcel():
    layerResistances.to_excel('Excels/ScenariosLayers_minus1000Width.xlsx', index=False)
