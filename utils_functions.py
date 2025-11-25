"""
Utility functions imported from data_read.ipynb
"""
import pandas as pd
import numpy as np


def to_float_array(arr):
    """
    Konvertiert ein beliebiges NumPy-Array in float-Typ.
    Unterstützt auch Listen oder Tupel als Eingabe.
    """
    try:
        # Falls Eingabe keine NumPy-Array ist, umwandeln
        if not isinstance(arr, np.ndarray):
            arr = np.array(arr)

        # Konvertieren in float64 (Standard-Double-Precision)
        float_arr = arr.astype(np.float64)

        return float_arr

    except (ValueError, TypeError) as e:
        print(f"Fehler bei der Umwandlung: {e}")
        return None


def df_d0(x):
    """
    Berechnet die relative Änderung gegenüber dem ersten Wert.
    Formel: (x[i] - x[0]) / x[0]
    """
    df_d0 = []
    for i in range(0, len(x)):
        df_d0.append((x[i] - x[0]) / x[0])
    return df_d0


def dataframe1(df1, x, y, z, w):
    """
    Erstellt einen DataFrame mit Mittelwerten und Standardabweichungen
    aus einem CSV-DataFrame und vier Zeilenindizes.
    
    Parameter:
    - df1: Eingabe-DataFrame (aus CSV)
    - x, y, z, w: Zeilenindizes für Zeit und die drei Messungen
    
    Rückgabe:
    - DataFrame mit Spalten: Time, 1.Messung, 2.Messung, 3.Messung, 
      Mittelwerte, Standardabweichung
    """
    Kontrolle = pd.DataFrame()
    Kontrolle["Time"] = to_float_array(np.array(df1.iloc[x])[1:])
    
    Kontrolle["1.Messung"] = df_d0(to_float_array(np.array(df1.iloc[y])[1:]))
    Kontrolle["2.Messung"] = df_d0(to_float_array(np.array(df1.iloc[z])[1:]))
    Kontrolle["3.Messung"] = df_d0(to_float_array(np.array(df1.iloc[w])[1:]))
    
    # Berechne Mittelwert über die ersten drei Messung-Spalten
    mittelwerte = Kontrolle[["1.Messung", "2.Messung", "3.Messung"]].mean(axis=1).tolist()
    Kontrolle["Mittelwerte"] = mittelwerte
    
    # Berechne Standardabweichung
    standardabweichung = Kontrolle[["1.Messung", "2.Messung", "3.Messung"]].std(axis=1).tolist()
    Kontrolle["Standardabweichung"] = standardabweichung
    
    return Kontrolle
