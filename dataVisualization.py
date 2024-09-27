import numpy as np
import pandas as pd
import bokeh
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, mean_squared_error
from sklearn.linear_model import LinearRegression  
from scipy.stats import gaussian_kde
import sklearn
from math import sqrt
import seaborn as sns

# Visualizar la similitud entre distintos textos
def visualize_similarity(embeddings_1, embeddings_2, labels_1, labels_2,
                         plot_title,
                         plot_width=1200, plot_height=600,
                         xaxis_font_size='12pt', yaxis_font_size='12pt'):

    assert len(embeddings_1) == len(labels_1)
    assert len(embeddings_2) == len(labels_2)

    # arccos based text similarity (Yang et al. 2019; Cer et al. 2019)
    sim = 1 - np.arccos(cosine_similarity(embeddings_1,
                                          embeddings_2))/np.pi

    # Comprobar valor de gamma (3er parámetro, por defecto 1.0/n_features)
    '''sim = sklearn.metrics.pairwise.laplacian_kernel(embeddings_1,
                                                   embeddings_2)'''

    embeddings_1_col, embeddings_2_col, sim_col = [], [], []
    for i in range(len(embeddings_1)):
        for j in range(len(embeddings_2)):
            embeddings_1_col.append(labels_1[i])
            embeddings_2_col.append(labels_2[j])
            sim_col.append(sim[i][j])
    df = pd.DataFrame(zip(embeddings_1_col, embeddings_2_col, sim_col),
                      columns=['embeddings_1', 'embeddings_2', 'sim'])

    mapper = bokeh.models.LinearColorMapper(
        palette=[*reversed(bokeh.palettes.YlOrRd[9])], low=df.sim.min(),
        high=df.sim.max())

    p = bokeh.plotting.figure(title=plot_title, x_range=labels_1,
                              x_axis_location="above",
                              y_range=[*reversed(labels_2)],
                              plot_width=plot_width, plot_height=plot_height,
                              tools="save", toolbar_location='below', tooltips=[
                                  ('pair', '@embeddings_1 ||| @embeddings_2'),
                                  ('sim', '@sim')])
    p.rect(x="embeddings_1", y="embeddings_2", width=1, height=1, source=df,
           fill_color={'field': 'sim', 'transform': mapper}, line_color=None)

    p.title.text_font_size = '12pt'
    p.axis.axis_line_color = None
    p.axis.major_tick_line_color = None
    p.axis.major_label_standoff = 16
    p.xaxis.major_label_text_font_size = xaxis_font_size
    p.xaxis.major_label_orientation = 0.25 * np.pi
    p.yaxis.major_label_text_font_size = yaxis_font_size
    p.min_border_right = 300

    bokeh.io.output_notebook()
    bokeh.io.show(p)


def savePredictionsToExcel(data, predictions, test, excelTitle):
    for ringIndex in range(10):
        data[f'RingPrediction{ringIndex}'] = predictions[:, ringIndex]
        data[f'RingActual{ringIndex}'] = test[:, ringIndex]

    df = pd.DataFrame(data)

    # Escribir el DataFrame en un archivo de Excel
    df.to_excel(f'Excels/{excelTitle}.xlsx', index=False)
    
def getConfusionMatrix(y_test, predictedNumRings):
    cm = confusion_matrix(y_test, predictedNumRings)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot()
    plt.show()
    
def getLinearRegressionPlot(y_test, y_pred, ring):   
    y_test_sample = (y_test[:, ring])
    y_pred_sample = (y_pred[:, ring])
    
    # Filtrar los valores mayores que -1 en array1 y obtener los índices donde se encuentran
    indexNoMinus1 = np.where(y_test_sample > -1)
    y_test_noMinus1 = y_test_sample[indexNoMinus1]
    
    # Obtener los valores correspondientes en array2
    y_pred_noMinus1 = y_pred_sample[indexNoMinus1]

    regressor = LinearRegression()  
    regressor.fit(y_test_noMinus1.reshape(-1,1), y_pred_noMinus1.reshape(-1,1))  
    y_fit_sample = regressor.predict(y_pred_noMinus1.reshape(-1,1))
    
    p1 = max(max(y_pred_noMinus1.reshape(-1,1)), max(y_test_noMinus1.reshape(-1,1)))
    p2 = min(min(y_pred_noMinus1.reshape(-1,1)), min(y_test_noMinus1.reshape(-1,1)))
    plt.plot([p1, p2], [p1, p2], 'b-', color='red', label= 'Linear regression')
    # Calculate the point density
    xy = np.vstack([y_test_noMinus1,y_pred_noMinus1])
    print(xy)
    z = gaussian_kde(xy)(xy)
    
    # Sort the points by density, so that the densest points are plotted last
    idx = z.argsort()
    x, y, z = y_test_noMinus1[idx], y_pred_noMinus1[idx], z[idx]
    plt.scatter(x, y, c=z, s=50, label='data')
    plt.title(f'Linear Regression ring {ring}')
    plt.legend()
    plt.xlabel('observed')
    plt.ylabel('predicted')
    plt.show()
    
    print("\n")
    print("Mean absolute error (MAE):      %f" % sklearn.metrics.mean_absolute_error(y_test_sample,y_pred_sample))
    print("Mean squared error (MSE):       %f" % mean_squared_error(y_test_sample,y_pred_sample))
    print("Root mean squared error (RMSE): %f" % sqrt(mean_squared_error(y_test_sample,y_pred_sample)))
    print("R square (R^2):                 %f" % sklearn.metrics.r2_score(y_test_sample,y_pred_sample))

def createHistogram(y_test, y_pred, ring):
    bordes_bins = [-0.25, 0, 0.25, 0.5, 0.75, 1, 1.25, 1.5, 1.75, 2, 2.25]
    # Crear subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))

    # Histograma de predicciones
    ax1.hist(y_pred[:, ring], bins=bordes_bins, color='blue', alpha=0.7)
    ax1.set_xlabel('Predicted values')
    ax1.set_ylabel('Frecuency')
    ax1.set_title('Predictions histogram')

    # Histograma de valores reales
    ax2.hist(y_test[:, ring].reshape(-1,1), bins=bordes_bins, color='green', alpha=0.7)
    ax2.set_xlabel('Actual values')
    ax2.set_ylabel('Frecuency')
    ax2.set_title('Actual values Histogram')

    # Mostrar los histogramas
    plt.tight_layout()
    plt.show()
    
def createDensityPlot(y_test, y_pred, ring):
    # Crear gráfico de densidad para predicciones
    sns.kdeplot(y_pred[:, ring], color='blue', label='Predictions')

    # Crear gráfico de densidad para valores reales
    sns.kdeplot(y_test[:, ring], color='green', label='Actual')

    # Agregar etiquetas y título
    plt.xlabel('Value')
    plt.ylabel('Density')
    plt.title('Comparison between predicted and actual values')

    # Mostrar el gráfico
    plt.legend()
    plt.show()
