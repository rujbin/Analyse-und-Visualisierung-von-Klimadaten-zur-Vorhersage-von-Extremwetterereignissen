import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# 1. Datenvorbereitung und -bereinigung

# CSV-Datei einlesen
df = pd.read_csv('klimadaten.csv')

# Überblick über die Daten
print("Datenübersicht:")
print(df.head())
print(df.info())

# Datumsspalte in Datetime-Format umwandeln
df['Datum'] = pd.to_datetime(df['Datum'])
df.set_index('Datum', inplace=True)

# Fehlende Werte überprüfen
print("\nFehlende Werte:")
print(df.isnull().sum())

# Fehlende Werte behandeln (hier wird ffill verwendet)
df.ffill(inplace=True)  # Ersetze fillna(method='ffill', inplace=True) mit ffill(inplace=True)

# 2. Zeitreihe analysieren

# Diagramm für Temperatur_Max
plt.figure(figsize=(12, 6))
df['Temperatur_Max'].plot(label='Maximale Temperatur', color='blue')
plt.title('Maximale Temperatur über die Zeit')
plt.xlabel('Datum')
plt.ylabel('Temperatur (°C)')
plt.legend()
plt.show()

# 3. Extremwetterereignisse identifizieren

# Hitzewellen definieren (95. Perzentil als Schwelle) für Temperatur_Max
heatwave_threshold = df['Temperatur_Max'].quantile(0.95)
heatwaves = df[df['Temperatur_Max'] > heatwave_threshold]

print(f"\nHitzewellen-Schwelle (Temperatur_Max): {heatwave_threshold}")
print("\nIdentifizierte Hitzewellen:")
print(heatwaves)

# 4. Modellierung und Vorhersage

# Merkmale hinzufügen (Monat und Jahr)
df['Monat'] = df.index.month
df['Jahr'] = df.index.year

# Features und Zielvariable festlegen
features = df[['Monat', 'Jahr', 'Temperatur_Min', 'Niederschlag', 'Luftfeuchtigkeit', 'Windgeschwindigkeit']]
target = df['Temperatur_Max']

# Daten aufteilen
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

# Modell erstellen und trainieren
model = RandomForestRegressor(n_estimators=100)
model.fit(X_train, y_train)

# Vorhersagen treffen
predictions = model.predict(X_test)

# Modellbewertung
mse = mean_squared_error(y_test, predictions)
print(f"\nMean Squared Error: {mse}")

# 5. Visualisierung der Vorhersagen

# Vorhersagen für Testdaten in DataFrame speichern (Index für X_test muss übereinstimmen)
X_test_with_index = X_test.copy()
X_test_with_index['Predictions'] = predictions

# Plot der Vorhersagen
plt.figure(figsize=(14, 7))
plt.plot(df.index, df['Temperatur_Max'], label='Echte Werte', color='blue')
plt.scatter(X_test_with_index.index, X_test_with_index['Predictions'], color='red', label='Vorhersagen', alpha=0.5)
plt.title('Temperaturvorhersagen')
plt.xlabel('Datum')
plt.ylabel('Temperatur (°C)')
plt.legend()
plt.show()
