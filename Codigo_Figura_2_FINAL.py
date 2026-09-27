# ============================================================
# FIGURA 2 — VERSIÓN PULIDA PARA LA TAREA 6
# Antes: versión original de la Tarea 2
# Después: versión revisada con criterios de publicación
# ============================================================

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

script_dir = Path(__file__).resolve().parent
csv_path = script_dir / "GRAFICAS_DATASET_FESTA.csv"

# 1. Carga y limpieza
with open(csv_path, "rb") as f:
    crudo = f.read()

contenido = None
for enc in ("utf-8", "cp1252", "latin-1"):
    try:
        contenido = crudo.decode(enc)
        break
    except UnicodeDecodeError:
        continue

if contenido is None:
    raise ValueError("No se pudo decodificar el CSV.")

filas = [r for r in contenido.splitlines() if r.strip(",").strip() != ""]
idx_encabezado = next(i for i, r in enumerate(filas) if r.startswith("Temporada"))
filas_datos = filas[idx_encabezado + 1:]

registros = []
for fila in filas_datos:
    partes = fila.split(",")
    if len(partes) < 6:
        break

    temporada, mes, tipo, mesas, sillas, precio = (p.strip() for p in partes[:6])

    if not (mesas.isdigit() and sillas.isdigit() and precio.isdigit()):
        break

    registros.append([
        temporada, mes, tipo,
        int(mesas), int(sillas), int(precio)
    ])

df = pd.DataFrame(
    registros,
    columns=["Temporada", "Mes", "Tipo_Evento", "Mesas", "Sillas", "Precio"]
)

# Se conservan únicamente eventos con mesas rentadas.
df = df[df["Mesas"] > 0].reset_index(drop=True)

# 2. Variables
x = df["Mesas"].to_numpy(dtype=float)
y = df["Precio"].to_numpy(dtype=float)

MESES_ORDEN = [
    "Enero", "Febrero", "Marzo", "Abril", "Mayo", "Junio",
    "Julio", "Agosto", "Septiembre", "Octubre", "Noviembre", "Diciembre"
]

num_mes = df["Mes"].map(
    lambda m: MESES_ORDEN.index(m) + 1
).to_numpy()

# 3. Regresión lineal y estadísticos
pendiente, intercepto = np.polyfit(x, y, deg=1)
y_pred = pendiente * x + intercepto

ss_res = np.sum((y - y_pred) ** 2)
ss_tot = np.sum((y - np.mean(y)) ** 2)
r2 = 1 - ss_res / ss_tot
r_pearson = np.corrcoef(x, y)[0, 1]

# 4. Figura pulida
plt.rcParams.update({
    "font.size": 11,
    "axes.titlesize": 14,
    "axes.labelsize": 12,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10
})

fig, ax = plt.subplots(figsize=(9, 6.5))

# Puntos: el color representa el mes del evento.
disp = ax.scatter(
    x, y,
    c=num_mes,
    s=58,
    alpha=0.88,
    zorder=3
)

# Barra de color
barra = fig.colorbar(
    disp,
    ax=ax,
    pad=0.025,
    fraction=0.05
)
barra.set_label(
    "Mes del evento (1 = Enero, 12 = Diciembre)",
    fontsize=11
)
barra.ax.tick_params(labelsize=9)

# Línea de tendencia
x_linea = np.linspace(x.min(), x.max(), 200)
ax.plot(
    x_linea,
    pendiente * x_linea + intercepto,
    linewidth=2.2,
    label="Regresión lineal",
    zorder=2
)

# Ecuación y R²
signo = "+" if intercepto >= 0 else "-"
ecuacion = (
    f"y = {pendiente:.1f}x {signo} {abs(intercepto):.1f}\n"
    f"$R^2$ = {r2:.3f}"
)

ax.text(
    0.03, 0.97, ecuacion,
    transform=ax.transAxes,
    fontsize=11,
    va="top",
    ha="left"
)

# Ejes con unidades
ax.set_xlabel("Mesas rentadas (unidades)", labelpad=8)
ax.set_ylabel("Precio del servicio (MXN)", labelpad=8)

# Título
ax.set_title(
    "Relación entre mesas rentadas y precio del servicio",
    pad=14
)

# Leyenda y cuadrícula
ax.legend(loc="lower right", frameon=False)
ax.grid(alpha=0.22, linewidth=0.8)

# Limpieza visual
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

fig.tight_layout()

# Exportación requerida por la rúbrica
output_png = script_dir / "Figura_2_FINAL_300dpi.png"
fig.savefig(output_png, dpi=300, bbox_inches="tight")

plt.show()

print("FIGURA FINAL GENERADA")
print(f"Eventos analizados: {len(x)}")
print(f"Pendiente: {pendiente:.2f} MXN por mesa adicional")
print(f"Intercepto: {intercepto:.2f}")
print(f"R²: {r2:.4f}")
print(f"Correlación de Pearson: {r_pearson:.4f}")
print(f"Archivo: {output_png}")
