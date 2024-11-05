import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
import joblib

# Cargar datos
df = pd.read_csv('data/entrada_salida_pasajeros.csv')

# Preprocesamiento de datos
df.drop('Unnamed: 5', axis=1, inplace=True)
df['ENTRADAS'] = df['ENTRADAS'].str.replace(',', '').astype(float).astype(int)
df['SALIDAS'] = df['SALIDAS'].str.replace(',', '').astype(float).astype(int)
df['AEROPUERTOS'] = df['AEROPUERTOS'].replace('LAS AMERICAS (JFPG)', 'LAS AMERICAS')

# Asignación de niveles de temporada
temporada_alta = ['DICIEMBRE', 'JULIO', 'AGOSTO', 'ENERO']
temporada_media = ['NOVIEMBRE', 'JUNIO', 'SEPTIEMBRE']
def clasificacion_temporada(mes):
    if mes in temporada_alta:
        return 2
    elif mes in temporada_media:
        return 1
    return 0

df['NIVEL_TEMPORADA'] = df['MES'].apply(clasificacion_temporada)

# Codificar aeropuertos
df_encoded = pd.get_dummies(df, columns=['AEROPUERTOS'], drop_first=True)

# Dividir en características (X) y variable objetivo (y)
X = df_encoded[['AÑO', 'NIVEL_TEMPORADA'] + [col for col in df_encoded.columns if 'AEROPUERTOS_' in col]]
y = df_encoded['ENTRADAS']

# División de datos
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Definir y entrenar el modelo RandomForest
modelo_rf = RandomForestRegressor(n_estimators=100, random_state=42)
modelo_rf.fit(X_train, y_train)

# Evaluación inicial
y_pred = modelo_rf.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
print(f'MAE inicial: {mae}\nMSE inicial: {mse}')

# Optimización de Hiperparámetros
param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [10, 30, None],
    'min_samples_split': [2, 5],
    'min_samples_leaf': [1, 2],
    'bootstrap': [True]
}
grid_search = GridSearchCV(estimator=RandomForestRegressor(random_state=42),
                           param_grid=param_grid,
                           cv=3,
                           scoring='neg_mean_absolute_error')
grid_search.fit(X_train, y_train)
mejores_parametros = grid_search.best_params_

# Reentrenar modelo optimizado
modelo_optimizado = RandomForestRegressor(**mejores_parametros, random_state=42)
modelo_optimizado.fit(X_train, y_train)
y_pred_optimizado = modelo_optimizado.predict(X_test)
mae_optimizado = mean_absolute_error(y_test, y_pred_optimizado)
mse_optimizado = mean_squared_error(y_test, y_pred_optimizado)
print(f'MAE optimizado: {mae_optimizado}\nMSE optimizado: {mse_optimizado}')

# Guardar el modelo
joblib.dump(modelo_optimizado, 'models/modelo_optimizado.pkl')

# Visualización de resultados
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred_optimizado, alpha=0.6)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
plt.xlabel('Valores Reales')
plt.ylabel('Predicciones')
plt.title('Valores Reales vs. Predicciones (Modelo Optimizado)')
plt.grid(True)
plt.show()