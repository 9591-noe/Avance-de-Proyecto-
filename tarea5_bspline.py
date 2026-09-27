import pandas as pd
import numpy as np

# 1. Cargar el dataset
df = pd.read_csv("GRAFICAS_DATASET_FESTA.csv", encoding="latin1", skiprows=5)
df = df.loc[:, ~df.columns.str.contains("^Unnamed")]

# 2. Mostrar las columnas disponibles
print("Columnas del dataset:")
print(df.columns.tolist())

# 3. Extraer la variable Precio en forma cruda
precio = pd.to_numeric(df["Precio"], errors="coerce")

# 4. Crear el número de observación
x = np.arange(1, len(precio) + 1)

# 5. Mostrar los primeros valores
print("\nPrimeros valores del perfil crudo de Precio:")
print(precio.head(10))

print("\nNúmero de observaciones:", len(precio))

import matplotlib.pyplot as plt

# 6. Graficar el perfil crudo
plt.figure(figsize=(10, 5))

plt.plot(x, precio, "o-", alpha=0.5, label="Datos crudos")

plt.xlabel("Número de observación")
plt.ylabel("Precio")
plt.title("Perfil crudo de Precio - Dataset FESTA")
plt.legend()
plt.grid(True, alpha=0.3)

plt.show()

# 7. Ajuste B-spline
from scipy.interpolate import UnivariateSpline

# Usar solamente los valores válidos para el ajuste
mask = np.isfinite(precio)

x_valid = x[mask]
precio_valid = precio[mask]

# Configuración 1: suavizado moderado
s1 = 500_000_000
spline1 = UnivariateSpline(x_valid, precio_valid, s=s1)

# Configuración 2: mayor suavizado
s2 = 2_000_000_000
spline2 = UnivariateSpline(x_valid, precio_valid, s=s2)

# Valores de las dos curvas
precio_suave1 = spline1(x_valid)
precio_suave2 = spline2(x_valid)

# Comparación de las dos configuraciones
plt.figure(figsize=(10, 5))

plt.plot(
    x_valid, precio_valid,
    "o", alpha=0.35,
    label="Datos crudos"
)

plt.plot(
    x_valid, precio_suave1,
    linewidth=2,
    label="B-spline: s = 500,000,000"
)

plt.plot(
    x_valid, precio_suave2,
    linewidth=3,
    label="B-spline: s = 2,000,000,000"
)

plt.xlabel("Número de observación")
plt.ylabel("Precio")
plt.title("Comparación de configuraciones de suavizado B-spline")
plt.legend()
plt.grid(True, alpha=0.3)

plt.show()

# ==========================================
# FIGURA FINAL PARA LA ENTREGA
# ==========================================

import matplotlib.pyplot as plt
from scipy.interpolate import UnivariateSpline

# Ajuste B-spline final
s_final = 5_000_000_000

# Eliminar únicamente valores no válidos para el ajuste
datos_validos = np.isfinite(precio)

x_bspl = x[datos_validos]
precio_bspl = precio[datos_validos]

# Crear B-spline
spline_final = UnivariateSpline(
    x_bspl,
    precio_bspl,
    s=s_final
)

# Puntos para dibujar la curva suavizada
x_suave = np.linspace(
    x_bspl.min(),
    x_bspl.max(),
    500
)

precio_suave = spline_final(x_suave)
# Evitar valores negativos, ya que el precio no puede ser menor que cero
precio_suave = np.maximum(precio_suave, 0)

# Comprobar que la curva tiene valores
print("Valor mínimo de la B-spline:", np.min(precio_suave))
print("Valor máximo de la B-spline:", np.max(precio_suave))

# Comprobar que la curva tiene valores
print("Valor mínimo de la B-spline:", np.min(precio_suave))
print("Valor máximo de la B-spline:", np.max(precio_suave))

# Crear figura
plt.figure(figsize=(10, 6))

# Datos crudos
plt.scatter(
    x,
    precio,
    s=35,
    alpha=0.45,
    label="Datos crudos"
)

# Curva B-spline
plt.plot(
    x_suave,
    precio_suave,
    linewidth=3,
    label="Ajuste B-spline"
)

# Etiquetas
plt.xlabel("Número de observación")
plt.ylabel("Precio ($)")

# Título
plt.title("Perfil de Precio: datos crudos y ajuste B-spline")

# Leyenda
plt.legend()

# Cuadrícula
plt.grid(alpha=0.3)

plt.tight_layout()

# Guardar en PNG a 300 dpi
plt.savefig(
    "figura_final_bspline.png",
    dpi=300,
    bbox_inches="tight"
)
plt.savefig("grafica_bspline_final.png", dpi=300, bbox_inches="tight")
plt.show()