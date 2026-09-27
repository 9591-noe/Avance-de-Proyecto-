"""
Tarea 2 — Relación entre dos variables clave
Proyecto: Festa Banquetes (Calkiní, Campeche) — Periodo 2025
Variables analizadas: Mesas rentadas (X) vs Precio del servicio (Y)

Autor: Equipo While
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ----------------------------------------------------------------------
# 1) Carga y limpieza del dataset
# ----------------------------------------------------------------------
# El CSV corregido viene con:
#   - varias líneas de encabezado libre arriba (título/fuente) y una nota
#     de cierre al final (después de la última fila real de datos)
#   - separador de línea "\r" (estilo Mac clásico) y codificación cp1252
#   - cada fila ya trae Temporada, Mes y Tipo_Evento completos (no hace
#     falta rellenar celdas hacia abajo como en la versión anterior)
# Se busca el CSV en la MISMA carpeta que este script, sin importar desde
# dónde lo ejecutes (VS Code, terminal, doble clic, etc.)
NOMBRE_CSV = "GRAFICAS_DATASET_FESTA.csv"
script_dir = os.path.dirname(os.path.abspath(__file__))
RUTA_CSV = os.path.join(script_dir, NOMBRE_CSV)

with open(RUTA_CSV, "rb") as f:
    crudo = f.read()

contenido = None
for enc in ("utf-8", "cp1252", "latin-1"):
    try:
        contenido = crudo.decode(enc)
        break
    except UnicodeDecodeError:
        continue
if contenido is None:
    raise ValueError("No se pudo decodificar el CSV con utf-8, cp1252 ni latin-1")

# .splitlines() reconoce automaticamente \r, \n o \r\n, sin importar
# como haya quedado guardado el archivo en cada computadora.
filas = [r for r in contenido.splitlines() if r.strip(",").strip() != ""]
idx_encabezado = next(i for i, r in enumerate(filas) if r.startswith("Temporada"))
filas_datos = filas[idx_encabezado + 1:]

registros = []
for fila in filas_datos:
    partes = fila.split(",")
    if len(partes) < 6:
        break  # se llegó a la nota de cierre del archivo
    temporada, mes, tipo, mesas, sillas, precio = (p.strip() for p in partes[:6])
    if not (mesas.isdigit() and sillas.isdigit() and precio.isdigit()):
        break  # ya no son datos numéricos válidos (nota de cierre)
    registros.append([temporada, mes, tipo, int(mesas), int(sillas), int(precio)])

df = pd.DataFrame(registros, columns=["Temporada", "Mes", "Tipo_Evento",
                                       "Mesas", "Sillas", "Precio"])

# Se descartan los renglones "en cero": representan combinaciones
# mes/tipo de evento que simplemente no ocurrieron ese mes, no eventos reales.
df = df[df["Mesas"] > 0].reset_index(drop=True)

# ----------------------------------------------------------------------
# 2) Extracción de las variables continuas (mismo tamaño por construcción,
#    ya vienen alineadas fila a fila en el DataFrame limpio)
# ----------------------------------------------------------------------
x = df["Mesas"].to_numpy(dtype=float)     # variable independiente
y = df["Precio"].to_numpy(dtype=float)    # variable dependiente
assert x.shape == y.shape

# Tercera variable, usada SOLO para color (no para el ajuste de tendencia):
# número de mes (1=Enero ... 12=Diciembre), como proxy de la época del año.
MESES_ORDEN = ["Enero", "Febrero", "Marzo", "Abril", "Mayo", "Junio", "Julio",
               "Agosto", "Septiembre", "Octubre", "Noviembre", "Diciembre"]
num_mes = df["Mes"].map(lambda m: MESES_ORDEN.index(m) + 1).to_numpy()

# ----------------------------------------------------------------------
# 3) Tipo de gráfico: ambas variables son mediciones independientes por
#    evento (no hay una evolución temporal continua entre ellas) ->
#    se usa un gráfico de DISPERSIÓN (scatter).
# ----------------------------------------------------------------------
# 4) Ajuste de tendencia: regresión lineal simple con numpy.polyfit
pendiente, intercepto = np.polyfit(x, y, deg=1)
y_pred = pendiente * x + intercepto

# Coeficiente de determinación R^2
ss_res = np.sum((y - y_pred) ** 2)
ss_tot = np.sum((y - np.mean(y)) ** 2)
r2 = 1 - ss_res / ss_tot

# Coeficiente de correlación de Pearson (para reportarlo también)
r_pearson = np.corrcoef(x, y)[0, 1]

# ----------------------------------------------------------------------
# 5) Gráfico con todos los elementos obligatorios
# ----------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(8, 6))

# Paleta: colormap 'viridis' (perceptualmente uniforme y legible para
# personas con daltonismo, criterio visto en clase) para codificar el mes
# del evento; así la figura suma una tercera dimensión (estacionalidad)
# sin dejar de ser un scatter de Mesas vs Precio.
disp = ax.scatter(x, y, c=num_mes, cmap="viridis", vmin=1, vmax=12,
                   edgecolor="white", s=70, zorder=3)

barra = fig.colorbar(disp, ax=ax)
barra.set_label("Mes del evento (1 = Enero, 12 = Diciembre)")

x_linea = np.linspace(x.min(), x.max(), 100)
ax.plot(x_linea, pendiente * x_linea + intercepto, color="#C00000",
        linewidth=2, label="Línea de tendencia (regresión lineal)", zorder=2)

signo = "+" if intercepto >= 0 else "-"
ecuacion = f"y = {pendiente:.1f}x {signo} {abs(intercepto):.1f}\n$R^2$ = {r2:.3f}"
ax.text(0.05, 0.95, ecuacion, transform=ax.transAxes,
        fontsize=12, verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="white", edgecolor="gray"))

ax.set_xlabel("Mesas rentadas (unidades)", fontsize=12)
ax.set_ylabel("Precio del servicio (MXN)", fontsize=12)
ax.set_title("Distribución Lineal Directa de Costos\npor Mobiliario Rentado (2025)", fontsize=13)
ax.legend(loc="lower right")
ax.grid(alpha=0.3)

fig.tight_layout()
output_img = os.path.join(script_dir, 'mesas_vs_precio.png')
plt.savefig(output_img, dpi=300)
plt.show()

# ----------------------------------------------------------------------
# 6) Resumen en consola (útil para redactar el documento corto)
# ----------------------------------------------------------------------
print(f"n = {len(x)} eventos analizados")
print(f"Pendiente:      {pendiente:.2f} MXN por mesa adicional")
print(f"Intercepto:     {intercepto:.2f}")
print(f"R^2:            {r2:.4f}")
print(f"Correlación r:  {r_pearson:.4f}")