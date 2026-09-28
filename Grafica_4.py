import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Cargar el dataset
script_dir = os.path.dirname(os.path.abspath(__file__))
csv_path = os.path.join(script_dir, "GRAFICAS_DATASET_FESTA.csv")

df = pd.read_csv(csv_path, skiprows=5, encoding="cp1252")
df = df.drop(columns=[c for c in df.columns if "Unnamed" in c])
df = df.dropna(subset=["Mes", "Tipo_Evento"])

# Limpiar datos
df.columns = [c.strip() for c in df.columns]
df["Mes"] = df["Mes"].str.strip()
df["Tipo_Evento"] = df["Tipo_Evento"].str.strip()

# Ordenar los meses
orden_meses = [
    "Enero", "Febrero", "Marzo", "Abril", "Mayo", "Junio",
    "Julio", "Agosto", "Septiembre", "Octubre", "Noviembre", "Diciembre"
]

df["Mes"] = pd.Categorical(
    df["Mes"],
    categories=orden_meses,
    ordered=True
)

# Calcular ingresos por mes
ingresos_mes = (
    df.groupby("Mes", observed=False)["Precio"]
    .sum()
    .reindex(orden_meses)
)

# Crear coordenadas
n = len(orden_meses)
angulos = np.linspace(0, 2 * np.pi, n, endpoint=False)
valores = ingresos_mes.values

# Crear gráfica polar
fig, ax = plt.subplots(
    figsize=(6, 6),
    subplot_kw={"projection": "polar"}
)

ax.set_theta_zero_location("N")
ax.set_theta_direction(-1)

# Paleta de colores
cmap = plt.get_cmap("viridis")
norm = plt.Normalize(valores.min(), valores.max())
colores = cmap(norm(valores))

# Crear barras
ancho_barra = (2 * np.pi / n) * 0.9

ax.bar(angulos, valores, width=ancho_barra, color=colores, edgecolor="white",
         linewidth=0.8, alpha=0.9)

# Etiquetas
ax.set_xticks(angulos)
ax.set_xticklabels(orden_meses, fontsize=8)
ax.tick_params(axis="y", labelsize=7)

ax.set_title(
    "Distribución Cíclica Anual de Ingresos\npor Renta de Mobiliario (2025)",
    fontsize=11,
    pad=35
)

# Barra de color
sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])

cbar = fig.colorbar(
    sm,
    ax=ax,
    pad=0.1,
    shrink=0.6
)

cbar.set_label(
    "Ingresos totales por mes (MXN)",
    fontsize=8
)

cbar.ax.tick_params(labelsize=7)

# Guardar y mostrar
plt.tight_layout()

output_img = os.path.join(
    script_dir,
    "tarea4_polar_ingresos.png"
)

plt.savefig(output_img, dpi=300)
plt.show()
