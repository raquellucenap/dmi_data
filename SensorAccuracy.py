import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
from itertools import combinations
from sklearn.metrics import accuracy_score, classification_report
import numpy as np
import random
# Obtén la ruta actual de los csv
ruta = os.getcwd()+"/""Instrument2/"
# Lista todos los archivos en la carpeta actual
archivos_csv = [archivo for archivo in os.listdir(ruta) if archivo.endswith('.csv') and archivo != 'V.csv']
# Inicializa listas para almacenar las accuracies de cada método
svm_accuracies = []
knn_accuracies = []
dt_accuracies = []
mlp_accuracies = []
lr_accuracies = []
rf_accuracies = []
# Recorre cada archivo CSV
seed = 109
random.seed(seed)
k=3
for archivo in archivos_csv:
    ruta_archivo = os.path.join(ruta, archivo)

    # Lee el archivo CSV usando pandas
    try:
        df = pd.read_csv(ruta_archivo)
        df = df[df.apply(lambda row: len(row.dropna()) == 13, axis=1)]      
        # Obtiene las dimensiones del DataFrame
        dimension = df.shape
        filas, columnas = dimension       
        # Imprime la información
        print('---------------------------------------------------------')
        print(f"Archivo: {archivo}, Dimensiones: {filas} filas x {columnas} columnas")
        # Dividir los datos en características (x) y la variable objetivo (y)
        x = df.drop(columns=['CLASE'])
        y = df['CLASE']
        # Dividir el conjunto de datos en entrenamiento y prueba
        # Identificar y eliminar outliers utilizando el rango intercuartílico (IQR)
        Q1 = x.quantile(0.1)
        Q3 = x.quantile(0.9)
        IQR = Q3 - Q1
        outlier_mask = ((x < (Q1 - 1.5 * IQR)) | (x > (Q3 + 1.5 * IQR))).any(axis=1)
        x = x[~outlier_mask]
        y = y[~outlier_mask]
        #Correlation to extract important features
        correlation_matrix = df.corr()
        high_corr_features = correlation_matrix[correlation_matrix['CLASE'].abs() > 0.9]['CLASE'].drop('CLASE').index.tolist()
        if not high_corr_features:
            high_corr_features = correlation_matrix.abs().nlargest(k+1, 'CLASE')['CLASE'].drop('CLASE').index.tolist()
        if len(high_corr_features) == 1:
            high_corr_features = correlation_matrix.abs().nlargest(3, 'CLASE')['CLASE'].drop('CLASE').index.tolist()

        X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=109)
        # SVM
        svm_model = SVC(kernel='rbf')
        svm_model.fit(X_train, y_train)
        svm_predictions = svm_model.predict(X_test)
        svm_accuracy = accuracy_score(y_test, svm_predictions)
        svm_accuracies.append(svm_accuracy) 
        print(f'SVM Accuracy: {svm_accuracy}')
        # KNN
        knn_model = KNeighborsClassifier()
        knn_model.fit(X_train, y_train)
        knn_predictions = knn_model.predict(X_test)
        knn_accuracy = accuracy_score(y_test, knn_predictions)
        knn_accuracies.append(knn_accuracy)
        print(f'KNN Accuracy: {knn_accuracy}')
        # Decision Tree
        dt_model = DecisionTreeClassifier()
        dt_model.fit(X_train, y_train)
        dt_predictions = dt_model.predict(X_test)
        dt_accuracy = accuracy_score(y_test, dt_predictions)
        dt_accuracies.append(dt_accuracy)
        print(f'Decision Tree Accuracy: {dt_accuracy}')
        # Neural Network
        mlp_model = MLPClassifier()
        mlp_model.fit(X_train, y_train)
        mlp_predictions = mlp_model.predict(X_test)
        mlp_accuracy = accuracy_score(y_test, mlp_predictions)
        mlp_accuracies.append(mlp_accuracy)
        print(f'Neural Network Accuracy: {mlp_accuracy}')
        # Logistic Regression
        lr_model = LogisticRegression()
        lr_model.fit(X_train, y_train)
        lr_predictions = lr_model.predict(X_test)
        lr_accuracy = accuracy_score(y_test, lr_predictions)
        lr_accuracies.append(lr_accuracy)
        print(f'Logistic Regression Accuracy: {lr_accuracy}')
        #Random Forest
        rf_model = RandomForestClassifier()
        rf_model.fit(X_train, y_train)
        rf_predictions = rf_model.predict(X_test)
        rf_accuracy = accuracy_score(y_test, rf_predictions)
        rf_accuracies.append(rf_accuracy)
        print(f'Random Forest Accuracy: {rf_accuracy}')   
    except Exception as e:
        print(f"Error al leer el archivo {archivo}: {e}")
# Calcula la media de las accuracies para cada método
svm_media = sum(svm_accuracies) / len(svm_accuracies)
knn_media = sum(knn_accuracies) / len(knn_accuracies)
dt_media = sum(dt_accuracies) / len(dt_accuracies)
mlp_media = sum(mlp_accuracies) / len(mlp_accuracies)
lr_media = sum(lr_accuracies) / len(lr_accuracies)
rf_media = sum(rf_accuracies) / len(rf_accuracies)

# Imprime las medias
print('\nMedia de Accuracies:')
print(f'SVM: {svm_media}')
print(f'KNN: {knn_media}')
print(f'Decision Tree: {dt_media}')
print(f'Neural Network: {mlp_media}')
print(f'Logistic Regression: {lr_media}')
print(f'Random Forest: {rf_media}')
print(f'Maxim Global Accuracy: {max(svm_media,knn_media,dt_media,mlp_media,lr_media,rf_media)}')
print(f'Minim Global Accuracy: {min(svm_media,knn_media,dt_media,mlp_media,lr_media,rf_media)}')
