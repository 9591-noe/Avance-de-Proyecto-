import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Directorio del script para no tener problemas de lectura
script_dir = os.path.dirname(os.path.abspath(__file__))
csv_path = os.path.join(script_dir, 'GRAFICAS_DATASET_FESTA.csv')

# Cargar dataset (tiene 5 filas de metadatos/fuente antes del encabezado real,
# y una columna extra vacia al final por la coma sobrante del CSV)
df = pd.read_csv(csv_path, skiprows=5, encoding='cp1252')
df = df.drop(columns=[c for c in df.columns if 'Unnamed' in c])

# Quitar filas de pie de pagina / vacias (el archivo trae una nota al final)
df = df.dropna(subset=['Mes', 'Tipo_Evento'])

# Limpieza de textos
df.columns = [c.strip() for c in df.columns]
df['Mes'] = df['Mes'].str.strip()
df['Tipo_Evento'] = df['Tipo_Evento'].str.strip()

# --- Componente ciclico ---
# El mes del ano es un ciclo estacional natural (se repite cada 12 meses),
# y el propio dataset lo confirma con la columna Temporada (Baja/Media/Alta).
# Por eso usamos un grafico POLAR: angulo = mes, magnitud = ingresos totales.
orden_meses = ['Enero', 'Febrero', 'Marzo', 'Abril', 'Mayo', 'Junio',
               'Julio', 'Agosto', 'Septiembre', 'Octubre', 'Noviembre', 'Diciembre']

df['Mes'] = pd.Categorical(df['Mes'], categories=orden_meses, ordered=True)

# Ingresos totales por mes (suma de Precio de todos los eventos de ese mes)
ingresos_mes = df.groupby('Mes', observed=False)['Precio'].sum().reindex(orden_meses)

# Coordenadas polares: 12 meses distribuidos en 360 grados
n = len(orden_meses)
angulos = np.linspace(0, 2 * np.pi, n, endpoint=False)
valores = ingresos_mes.values

# --- Grafico polar ---
fig, ax = plt.subplots(figsize=(6, 6), subplot_kw={'projection': 'polar'})

ax.set_theta_zero_location('N')  # Enero arriba
ax.set_theta_direction(-1)       # sentido horario (Ene -> Dic)

# Barras coloreadas por magnitud con paleta viridis (consistente con el resto del proyecto)
cmap = plt.get_cmap('viridis')
norm = plt.Normalize(valores.min(), valores.max())
colores = cmap(norm(valores))

ancho_barra = (2 * np.pi / n) * 0.9
barras = ax.bar(angulos, valores, width=ancho_barra, color=colores,
                 edgecolor='white', linewidth=0.8, alpha=0.9)

# Etiquetas angulares = meses
ax.set_xticks(angulos)
ax.set_xticklabels(orden_meses, fontsize=8)
ax.tick_params(axis='y', labelsize=7)

# Etiquetas radiales con unidades
ax.set_ylabel('')
ax.set_title(
    'Distribución Cíclica Anual de Ingresos\npor Renta de Mobiliario (2025)',
    fontsize=11, pad=35, ha='center'
)

# Barra de color (magnitud = ingresos en MXN)
sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])
cbar = fig.colorbar(sm, ax=ax, pad=0.1, shrink=0.6)
cbar.set_label('Ingresos totales por mes (MXN)', fontsize=8)
cbar.ax.tick_params(labelsize=7)

plt.tight_layout()

# Guardar la imagen (300 dpi, como pide el entregable)
output_img = os.path.join(script_dir, 'tarea4_polar_ingresos.png')
plt.savefig(output_img, dpi=300)
plt.show()