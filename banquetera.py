import pandas as pd
import matplotlib.pyplot as plt
import unicodedata
import re

# Leer el archivo CSV
datos = pd.read_csv("GRAFICAS_DATASET_FESTA.csv", encoding="latin1",
    skiprows=5
                     )

# Limpiar nombres de columnas
datos.columns = datos.columns.str.strip()

# Función para limpiar textos
def limpiar_texto(texto):
    texto = str(texto)
    texto = texto.replace("\xa0", " ")
    texto = texto.strip()
    texto = re.sub(r"\s+", " ", texto)

    # Quitar acentos
    texto = unicodedata.normalize("NFKD", texto)
    texto = texto.encode("ascii", "ignore").decode()

    return texto.lower()

# Limpiar Mes y Evento
datos["Mes"] = datos["Mes"].apply(limpiar_texto)
datos["Tipo_Evento"] = datos["Tipo_Evento"].apply(limpiar_texto)

# Convertir Precio a número
datos["Precio"] = pd.to_numeric(
    datos["Precio"],
    errors="coerce"
)

# Convertir nombres a nombres definitivos
datos["Mes"] = datos["Mes"].replace({
    "enero": "Enero",
    "febrero": "Febrero",
    "marzo": "Marzo",
    "abril": "Abril",
    "mayo": "Mayo",
    "junio": "Junio",
    "julio": "Julio",
    "agosto": "Agosto",
    "septiembre": "Septiembre",
    "octubre": "Octubre",
    "noviembre": "Noviembre",
    "diciembre": "Diciembre"
})

# Convertir nombres de eventos
datos["Tipo_Evento"] = datos["Tipo_Evento"].replace({

    "boda": "Boda",
    "xv anos": "XV Anos",
    "cumpleanos": "Cumpleanos",
    "cumpleaños": "Cumpleanos",
    "bautizo": "Bautizo",
    "graduacion": "Graduacion"
})

# Crear la tabla
#Se suma el precio para obtener el ingreso total por mes y tipo de evento
tabla = datos.pivot_table(
    index="Mes",
    columns="Tipo_Evento",
    values="Precio",
    aggfunc="sum",
    fill_value=0
)

# Orden de los meses
orden_meses = [ 
    "Enero",
    "Febrero", 
    "Marzo", 
    "Abril", 
    "Mayo",
    "Junio", 
    "Julio", 
    "Agosto", 
    "Septiembre", 
    "Octubre", 
    "Noviembre", 
    "Diciembre" 
]
# Orden de los eventos
orden_eventos = [ 
    "Boda", 
    "XV Anos", 
    "Cumpleanos", 
    "Bautizo", 
    "Graduacion" 
]

tabla = tabla.reindex(index=orden_meses)
tabla = tabla.reindex(columns=orden_eventos)

# Crear gráfica
plt.figure(figsize=(11, 7))

plt.imshow(
    tabla.values,
    cmap="viridis",
    aspect="auto"
)

# Barra de colores
plt.colorbar(
    label="Ingreso total(MXN)"
)

# Nombres de los ejes
plt.xlabel("Tipo de Evento")
plt.ylabel("Mes")
plt.title("Ingreso por Renta de Mobiliario por Mes y Tipo de Evento")

# Etiquetas
plt.xticks(
    range(len(tabla.columns)),
    tabla.columns,
    rotation=30,
    ha="right"
)

plt.yticks(
    range(len(tabla.index)),
    tabla.index
)

# Mostrar los valores dentro de cada cuadro
for i in range(len(tabla.index)):
    for j in range(len(tabla.columns)):
        valor = tabla.iloc[i, j]

        plt.text(
            j,
            i,
            f"${valor:,.0f}",
            ha="center",
            va="center",
            fontsize=8
        )


plt.tight_layout()

# Guardar imágenes
plt.savefig("mapa_ingresos_150dpi.png", dpi=150, bbox_inches="tight")
plt.savefig("mapa_ingresos_300dpi.png", dpi=300)

# Mostrar
plt.show()