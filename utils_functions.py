"""
Utility functions imported from data_read.ipynb
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def to_float_array(arr):
    """
    Konvertiert ein beliebiges NumPy-Array in float-Typ.
    Unterstützt auch Listen oder Tupel als Eingabe.
    Behandelt Kommas als Dezimaltrennzeichen.
    """
    try:
        # Falls Eingabe keine NumPy-Array ist, umwandeln
        if not isinstance(arr, np.ndarray):
            arr = np.array(arr)
        
        # Wenn es Strings sind, ersetze Kommas durch Punkte
        if arr.dtype.kind in ['U', 'S', 'O']:  # Unicode, Byte-String oder Object
            arr = np.array([str(x).replace(',', '.') if isinstance(x, str) else x for x in arr])

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

def df_d0_absolut(x):
    """
    Berechnet die relative Änderung gegenüber dem ersten Wert.
    Formel: (x[i] - x[0]) / x[0]
    """
    df_d0 = []
    for i in range(0, len(x)):
        df_d0.append((x[i] - x[0]))
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

def dataframe1_absolut(df1, x, y, z, w):
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
    
    Kontrolle["1.Messung"] = df_d0_absolut(to_float_array(np.array(df1.iloc[y])[1:]))
    Kontrolle["2.Messung"] = df_d0_absolut(to_float_array(np.array(df1.iloc[z])[1:]))
    Kontrolle["3.Messung"] = df_d0_absolut(to_float_array(np.array(df1.iloc[w])[1:]))
    
    # Berechne Mittelwert über die ersten drei Messung-Spalten
    mittelwerte = Kontrolle[["1.Messung", "2.Messung", "3.Messung"]].mean(axis=1).tolist()
    Kontrolle["Mittelwerte"] = mittelwerte
    
    # Berechne Standardabweichung
    standardabweichung = Kontrolle[["1.Messung", "2.Messung", "3.Messung"]].std(axis=1).tolist()
    Kontrolle["Standardabweichung"] = standardabweichung
    
    return Kontrolle

# Funktion 1: Exponentieller Fit der Daten
def fit_exponential_decay(data_frames):
    """
    Fittet Daten mit exponentieller Zerfallsfunktion: y = A*exp(-k*t) + C
    
    Parameters:
    -----------
    data_frames : list of tuples
        Liste mit (name, marker, dataframe) für jede Messung
        
    Returns:
    --------
    results : dict
        Dictionary mit Namen als Keys und dict mit Fit-Parametern als Values
        Format: {'name': {'A': val, 'A_err': err, 'k': val, 'k_err': err, 
                         'C': val, 'C_err': err, 'flux': val, 'flux_err': err,
                         'times': array, 'means': array, 'stds': array}}
    """
    from scipy.optimize import curve_fit
    
    def exp_decay(t, A, k, C):
        """Exponentielle Zerfallsfunktion: y = A*exp(-k*t) + C"""
        return A * np.exp(-k * t) + C
    
    results = {}
    
    for name, marker, df in data_frames:
        times = df["Time"].values
        means = df["Mittelwerte"].values
        stds = df["Standardabweichung"].values
        
        # Entferne NaN-Werte
        mask = ~(np.isnan(times) | np.isnan(means))
        times_clean = times[mask]
        means_clean = means[mask]
        stds_clean = stds[mask]
        
        try:
            # Startparameter schätzen
            A_guess = means_clean[0] - means_clean[-1]
            C_guess = means_clean[-1]
            k_guess = 0.01
            
            popt, pcov = curve_fit(exp_decay, times_clean, means_clean, 
                                  p0=[A_guess, k_guess, C_guess],
                                  maxfev=10000)
            
            A_fit, k_fit, C_fit = popt
            
            # Berechne Unsicherheiten
            A_err = np.sqrt(pcov[0,0])
            k_err = np.sqrt(pcov[1,1])
            C_err = np.sqrt(pcov[2,2])
            
            # Berechne initialen Fluss: Flux = k * A
            flux = k_fit * A_fit
            flux_err = np.sqrt((k_fit * A_err)**2 + (A_fit * k_err)**2)
            
            # Speichere Ergebnisse
            results[name] = {
                'A': A_fit,
                'A_err': A_err,
                'k': k_fit,
                'k_err': k_err,
                'C': C_fit,
                'C_err': C_err,
                'flux': flux,
                'flux_err': flux_err,
                'times': times_clean,
                'means': means_clean,
                'stds': stds_clean,
                'marker': marker
            }
            
        except Exception as e:
            print(f'Fit für {name} fehlgeschlagen: {e}')
            results[name] = None
    
    return results


# Funktion 2: Plot mit Daten und Fit erstellen
def plot_data_with_fits(results, title='Exponential Fits', save_path=None, uniform_axes=True):
    """
    Erstellt Subplot-Grid mit Daten und Fits
    
    Parameters:
    -----------
    results : dict
        Dictionary aus fit_exponential_decay()
    title : str
        Titel für die gesamte Figur
    save_path : str, optional
        Pfad zum Speichern der Figur
    uniform_axes : bool
        Ob alle Subplots gleiche Achsenskalierung haben sollen
        
    Returns:
    --------
    fig, axes : matplotlib Figure und Axes
    """
    def exp_decay(t, A, k, C):
        return A * np.exp(-k * t) + C
    
    n = len(results)
    cols = 3
    rows = (n + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(18, 5*rows))
    if n == 1:
        axes = [axes]
    else:
        axes = axes.flatten()
    
    # Sammle Daten für uniforme Skalierung
    all_times = []
    all_values = []
    
    for idx, (name, data) in enumerate(results.items()):
        if idx >= len(axes):
            break
            
        ax = axes[idx]
        
        if data is None:
            ax.text(0.5, 0.5, f'{name}\nFit fehlgeschlagen', 
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title(name, fontsize=10)
            continue
        
        times = data['times']
        means = data['means']
        stds = data['stds']
        marker = data['marker']
        
        all_times.extend(times)
        all_values.extend(means)
        
        # Plotte Originaldaten
        ax.errorbar(times, means, yerr=stds, 
                   marker=marker, linestyle='', label='Data', capsize=4, 
                   markerfacecolor='white', markeredgewidth=1.5, markersize=6, 
                   alpha=0.7, color='C0')
        
        # Plotte Fit
        t_fit = np.linspace(times.min(), times.max(), 200)
        y_fit = exp_decay(t_fit, data['A'], data['k'], data['C'])
        ax.plot(t_fit, y_fit, '-', linewidth=2, color='C1', label='Fit')
        
        # Füge Parameter als Text hinzu
        textstr = f"A = {data['A']:.3f} ± {data['A_err']:.3f}\n"
        textstr += f"k = {data['k']:.4f} ± {data['k_err']:.4f}\n"
        textstr += f"C = {data['C']:.3f} ± {data['C_err']:.3f}"
        ax.text(0.95, 0.95, textstr, transform=ax.transAxes, 
               fontsize=8, verticalalignment='top', horizontalalignment='right',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        ax.set_title(name, fontsize=10)
        ax.set_xlabel('Time [s]')
        ax.set_ylabel(r'$\Delta F/F_0$', rotation=0, labelpad=20)
        ax.yaxis.set_label_coords(-0.15, 0.5)
        ax.grid(alpha=0.3)
        ax.legend(loc='upper left', fontsize=8)
    
    # Verstecke überschüssige Subplots
    for idx in range(len(results), len(axes)):
        axes[idx].axis('off')
    
    # Setze uniforme Achsenskalierung
    if uniform_axes and all_times and all_values:
        x_min, x_max = min(all_times), max(all_times)
        y_min, y_max = min(all_values), max(all_values)
        x_margin = (x_max - x_min) * 0.05
        y_margin = (y_max - y_min) * 0.1
        
        for idx in range(len(results)):
            axes[idx].set_xlim(x_min - x_margin, x_max + x_margin)
            axes[idx].set_ylim(y_min - y_margin, y_max + y_margin)
    
    plt.suptitle(title, fontsize=14, y=0.995)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f'Saved plot: {save_path}')
    
    return fig, axes


# Funktion 3: Histogramme für Parameter-Vergleich
def plot_parameter_histograms(results, save_path=None, figsize=(20, 6)):
    """
    Erstellt drei Histogramme: k, A und Flux (k*A)
    
    Parameters:
    -----------
    results : dict
        Dictionary aus fit_exponential_decay()
    save_path : str, optional
        Pfad zum Speichern der Figur
    figsize : tuple
        Größe der Figur (width, height)
        
    Returns:
    --------
    fig, axes : matplotlib Figure und Axes
    """
    # Extrahiere Daten
    names = []
    k_values = []
    k_errors = []
    A_values = []
    A_errors = []
    flux_values = []
    flux_errors = []
    
    for name, data in results.items():
        if data is None:
            continue
        names.append(name)
        k_values.append(data['k'])
        k_errors.append(data['k_err'])
        A_values.append(data['A'])
        A_errors.append(data['A_err'])
        flux_values.append(data['flux'])
        flux_errors.append(data['flux_err'])
    
    # Erstelle drei Histogramme
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=figsize)
    
    x_pos = np.arange(len(names))
    
    # Histogramm 1: Transportrate k
    bars1 = ax1.bar(x_pos, k_values, yerr=k_errors, capsize=5, 
                   color='steelblue', edgecolor='black', linewidth=1.5, alpha=0.7)
    ax1.set_xlabel('Messung', fontsize=12)
    ax1.set_ylabel('k [1/s]', fontsize=12)
    ax1.set_title('Transportrate (k)', fontsize=14, fontweight='bold')
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(names, rotation=45, ha='right', fontsize=9)
    ax1.grid(axis='y', alpha=0.3)
    
    for i, (bar, k_val, k_err) in enumerate(zip(bars1, k_values, k_errors)):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + k_err,
                f'{k_val:.4f}', ha='center', va='bottom', fontsize=8)
    
    # Histogramm 2: Vorfaktor A
    bars2 = ax2.bar(x_pos, A_values, yerr=A_errors, capsize=5,
                   color='coral', edgecolor='black', linewidth=1.5, alpha=0.7)
    ax2.set_xlabel('Messung', fontsize=12)
    ax2.set_ylabel('A', fontsize=12)
    ax2.set_title('Amplitude (A)', fontsize=14, fontweight='bold')
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(names, rotation=45, ha='right', fontsize=9)
    ax2.grid(axis='y', alpha=0.3)
    
    for i, (bar, A_val, A_err) in enumerate(zip(bars2, A_values, A_errors)):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + A_err,
                f'{A_val:.3f}', ha='center', va='bottom', fontsize=8)
    
    # Histogramm 3: Initialer Fluss
    bars3 = ax3.bar(x_pos, flux_values, yerr=flux_errors, capsize=5,
                   color='mediumpurple', edgecolor='black', linewidth=1.5, alpha=0.7)
    ax3.set_xlabel('Messung', fontsize=12)
    ax3.set_ylabel('Flux (k·A) [1/s]', fontsize=12)
    ax3.set_title('Initialer Fluss (k·A)', fontsize=14, fontweight='bold')
    ax3.set_xticks(x_pos)
    ax3.set_xticklabels(names, rotation=45, ha='right', fontsize=9)
    ax3.grid(axis='y', alpha=0.3)
    
    for i, (bar, flux_val, flux_err) in enumerate(zip(bars3, flux_values, flux_errors)):
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height + flux_err,
                f'{flux_val:.5f}', ha='center', va='bottom', fontsize=7)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f'Saved parameter histograms: {save_path}')
    
    # Drucke Zusammenfassung
    print('\n=== Parameter-Zusammenfassung ===')
    print(f'{"Messung":<30} {"k [1/s]":<18} {"A":<18} {"Flux [1/s]":<18}')
    print('-' * 90)
    for name, k, k_e, A, A_e, flux, flux_e in zip(names, k_values, k_errors, 
                                                    A_values, A_errors, 
                                                    flux_values, flux_errors):
        print(f'{name:<30} {k:.4f}±{k_e:.4f}    {A:.3f}±{A_e:.3f}    {flux:.5f}±{flux_e:.5f}')
    
    return fig, (ax1, ax2, ax3)


# Funktion 4: Linearer Fit (optional Zeitfenster)
def fit_linear(data_frames, time_limit=None):
    """
    Fittet Daten mit linearer Funktion: y = m*t + b
    
    Parameters:
    -----------
    data_frames : list of tuples
        Liste mit (name, marker, dataframe) für jede Messung
        
    Returns:
    --------
    results : dict
        Dictionary mit Namen als Keys und dict mit Fit-Parametern als Values
        Format: {'name': {'m': slope, 'm_err': err, 'b': intercept, 'b_err': err,
                         'times': array, 'means': array, 'stds': array}}
    """
    from scipy.optimize import curve_fit
    
    def linear(t, m, b):
        """Lineare Funktion: y = m*t + b"""
        return m * t + b
    
    results = {}
    
    for name, marker, df in data_frames:
        times = df["Time"].values
        means = df["Mittelwerte"].values
        stds = df["Standardabweichung"].values
        
        # Entferne NaN-Werte
        mask = ~(np.isnan(times) | np.isnan(means))
        times_clean = times[mask]
        means_clean = means[mask]
        stds_clean = stds[mask]

        # Optional auf die ersten time_limit Sekunden begrenzen
        if time_limit is not None:
            time_mask = times_clean <= time_limit
            times_clean = times_clean[time_mask]
            means_clean = means_clean[time_mask]
            stds_clean = stds_clean[time_mask]
        
        try:
            # Linearer Fit
            popt, pcov = curve_fit(linear, times_clean, means_clean, 
                                  p0=[0.001, means_clean[0]],
                                  maxfev=10000)
            
            m_fit, b_fit = popt
            
            # Berechne Unsicherheiten
            m_err = np.sqrt(pcov[0,0])
            b_err = np.sqrt(pcov[1,1])
            
            # Speichere Ergebnisse
            results[name] = {
                'm': m_fit,
                'm_err': m_err,
                'b': b_fit,
                'b_err': b_err,
                'times': times_clean,
                'means': means_clean,
                'stds': stds_clean,
                'marker': marker
            }
            
        except Exception as e:
            print(f'Linearer Fit für {name} fehlgeschlagen: {e}')
            results[name] = None
    
    return results


# Funktion 5: Plot für linearen Fit
def plot_linear_fits(results, title='Linear Fits', save_path=None, uniform_axes=True):
    """
    Erstellt Subplot-Grid mit Daten und linearen Fits
    
    Parameters:
    -----------
    results : dict
        Dictionary aus fit_linear()
    title : str
        Titel für die gesamte Figur
    save_path : str, optional
        Pfad zum Speichern der Figur
    uniform_axes : bool
        Ob alle Subplots gleiche Achsenskalierung haben sollen
        
    Returns:
    --------
    fig, axes : matplotlib Figure und Axes
    """
    def linear(t, m, b):
        return m * t + b
    
    n = len(results)
    cols = 3
    rows = (n + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(18, 5*rows))
    if n == 1:
        axes = [axes]
    else:
        axes = axes.flatten()
    
    # Sammle Daten für uniforme Skalierung
    all_times = []
    all_values = []
    
    for idx, (name, data) in enumerate(results.items()):
        if idx >= len(axes):
            break
            
        ax = axes[idx]
        
        if data is None:
            ax.text(0.5, 0.5, f'{name}\nFit fehlgeschlagen', 
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title(name, fontsize=10)
            continue
        
        times = data['times']
        means = data['means']
        stds = data['stds']
        marker = data['marker']
        
        all_times.extend(times)
        all_values.extend(means)
        
        # Plotte Originaldaten
        ax.errorbar(times, means, yerr=stds, 
                   marker=marker, linestyle='', label='Data', capsize=4, 
                   markerfacecolor='white', markeredgewidth=1.5, markersize=6, 
                   alpha=0.7, color='C0')
        
        # Plotte Fit
        t_fit = np.linspace(times.min(), times.max(), 200)
        y_fit = linear(t_fit, data['m'], data['b'])
        ax.plot(t_fit, y_fit, '-', linewidth=2, color='C1', label='Linear Fit')
        
        # Füge Parameter als Text hinzu
        textstr = f"m = {data['m']:.4f} ± {data['m_err']:.4f}\n"
        textstr += f"b = {data['b']:.3f} ± {data['b_err']:.3f}"
        ax.text(0.95, 0.95, textstr, transform=ax.transAxes, 
               fontsize=8, verticalalignment='top', horizontalalignment='right',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        ax.set_title(name, fontsize=10)
        ax.set_xlabel('Time [s]')
        ax.set_ylabel(r'$\Delta F/F_0$', rotation=0, labelpad=20)
        ax.yaxis.set_label_coords(-0.15, 0.5)
        ax.grid(alpha=0.3)
        ax.legend(loc='upper left', fontsize=8)
    
    # Verstecke überschüssige Subplots
    for idx in range(len(results), len(axes)):
        axes[idx].axis('off')
    
    # Setze uniforme Achsenskalierung
    if uniform_axes and all_times and all_values:
        x_min, x_max = min(all_times), max(all_times)
        y_min, y_max = min(all_values), max(all_values)
        x_margin = (x_max - x_min) * 0.05
        y_margin = (y_max - y_min) * 0.1
        
        for idx in range(len(results)):
            axes[idx].set_xlim(x_min - x_margin, x_max + x_margin)
            axes[idx].set_ylim(y_min - y_margin, y_max + y_margin)
    
    plt.suptitle(title, fontsize=14, y=0.995)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f'Saved plot: {save_path}')
    
    return fig, axes


# Funktion 6: Histogramm für Steigungen vergleichen
def plot_slope_histogram(results, save_path=None, figsize=(12, 6)):
    """
    Erstellt ein Histogramm zum Vergleich der Steigungen (m)
    
    Parameters:
    -----------
    results : dict
        Dictionary aus fit_linear()
    save_path : str, optional
        Pfad zum Speichern der Figur
    figsize : tuple
        Größe der Figur (width, height)
        
    Returns:
    --------
    fig, ax : matplotlib Figure und Axes
    """
    # Extrahiere Daten
    names = []
    m_values = []
    m_errors = []
    
    for name, data in results.items():
        if data is None:
            continue
        names.append(name)
        m_values.append(data['m'])
        m_errors.append(data['m_err'])
    
    # Erstelle Histogramm
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    
    x_pos = np.arange(len(names))
    
    # Histogramm: Steigungen
    bars = ax.bar(x_pos, m_values, yerr=m_errors, capsize=5, 
                 color='steelblue', edgecolor='black', linewidth=1.5, alpha=0.7)
    ax.set_xlabel('Messung', fontsize=12)
    ax.set_ylabel('Steigung m [1/s]', fontsize=12)
    ax.set_title('Vergleich der Steigungen (m)', fontsize=14, fontweight='bold')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(names, rotation=45, ha='right', fontsize=10)
    ax.grid(axis='y', alpha=0.3)
    
    # Füge Werte über den Balken ein
    for i, (bar, m_val, m_err) in enumerate(zip(bars, m_values, m_errors)):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + m_err,
                f'{m_val:.4f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f'Saved slope histogram: {save_path}')
    
    # Drucke Zusammenfassung
    print('\n=== Steigungen-Zusammenfassung (m) ===')
    print(f'{"Messung":<30} {"m [1/s]":<20}')
    print('-' * 50)
    for name, m, m_e in zip(names, m_values, m_errors):
        print(f'{name:<30} {m:.4f} ± {m_e:.4f}')
    
    return fig, ax


print("Funktionen erfolgreich definiert:")
print("1. fit_exponential_decay(data_frames)")
print("2. plot_data_with_fits(results, title, save_path, uniform_axes)")
print("3. plot_parameter_histograms(results, save_path, figsize)")
print("4. fit_linear(data_frames)")
print("5. plot_linear_fits(results, title, save_path, uniform_axes)")
print("6. plot_slope_histogram(results, save_path, figsize)")
