import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Cargar el dataset
df = pd.read_csv("GRAFICAS_DATASET_FESTA.csv", encoding="latin1", skiprows=5)

# Eliminar registros donde no hay mesas
df = df[df["Mesas"] > 0]

# Seleccionar variables
x = df["Mesas"]
y = df["Precio"]

# Convertir los meses a números
meses = {
    "Enero": 1,
    "Febrero": 2,
    "Marzo": 3,
    "Abril": 4,
    "Mayo": 5,
    "Junio": 6,
    "Julio": 7,
    "Agosto": 8,
    "Septiembre": 9,
    "Octubre": 10,
    "Noviembre": 11,
    "Diciembre": 12
}

num_mes = df["Mes"].map(meses)

# Calcular regresión lineal
pendiente, intercepto = np.polyfit(x, y, 1)
y_pred = pendiente * x + intercepto

# Calcular R²
ss_res = np.sum((y - y_pred) ** 2)
ss_tot = np.sum((y - np.mean(y)) ** 2)
r2 = 1 - ss_res / ss_tot

# Calcular correlación
r = np.corrcoef(x, y)[0, 1]

# Crear gráfica
plt.figure(figsize=(7, 5))

# Agregar línea de tendencia primero
x_linea = np.linspace(x.min(), x.max(), 100)
plt.plot(
    x_linea,
    pendiente * x_linea + intercepto,
    color="#C00000",
    linewidth=2,
    label="Línea de tendencia (regresión lineal)",
    zorder=1
)

# Agregar los puntos 
puntos = plt.scatter( x, y, c=num_mes, cmap="viridis",
                     vmin=1, vmax=12, edgecolor="white",s=70, zorder=2)

#  Agregar barra de color
barra = plt.colorbar(puntos)
barra.set_label("Mes del evento (1 = Enero, 12 = Diciembre)")

# Agregar ecuación y R²
signo = "+" if intercepto >= 0 else "-"
ecuacion = f"y = {pendiente:.1f}x {signo} {abs(intercepto):.1f}\nR² = {r2:.3f}"

plt.text(
    0.05,
    0.95,
    ecuacion,
    transform=plt.gca().transAxes,
    fontsize=12,
    verticalalignment="top",
    bbox=dict(boxstyle="round", facecolor="white", edgecolor="gray")
)

# Títulos y etiquetas
plt.title("Distribución Lineal Directa de Costos\npor Mobiliario Rentado (2025)")
plt.xlabel("Mesas rentadas (unidades)")
plt.ylabel("Precio del servicio (MXN)")

# Cuadrícula
plt.grid(alpha=0.3)

# Leyenda
plt.legend(loc="lower right")

# Guardar imagen
plt.tight_layout()
plt.savefig("mesas_vs_precio.png", dpi=300)
plt.show()

# Mostrar resultados
print("Pendiente:", pendiente)
print("R²:", r2)
print("Correlación:", r)
