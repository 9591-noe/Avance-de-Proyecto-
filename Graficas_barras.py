import pandas as pd
import matplotlib.pyplot as plt

# 1. Cargar el dataset
df = pd.read_csv("GRAFICAS_DATASET_FESTA.csv", 
                 encoding="latin1", 
                 skiprows=5
                 )

# 2. Mostrar los datos para comprobar que se cargaron correctamente
print(df)

# 3.Eliminar registros donde no hay mesas utilizadas
df = df[df["Mesas"] > 0]

# 4. Calcular el promedio y la desviación estándar de mesas utilizadas por tipo de evento 
tabla = df.groupby("Tipo_Evento")["Mesas"].agg( ["mean", "std"] )

print("\nTabla para la gráfica:")
print(tabla)

# 5. Crear la gráfica de barras
barras = plt.bar( 
    tabla.index, 
    tabla["mean"],
    yerr=tabla["std"], 
    capsize=5 
    )

# 6. Agregar líneas de cuadrícula
plt.grid(
    axis="y",
    linestyle="--",
    alpha=0.7
)
# 7. Agregar títulos y etiquetas
plt.title("Promedio de Mesas Utilizadas por Tipo de Evento")
plt.xlabel("Tipo de Evento")
plt.ylabel("Promedio de Mesas Utilizadas")

# 8. Mostrar los nombres de los eventos correctamente
plt.xticks(rotation=0)

# 9. Ajustar el espacio
plt.tight_layout()

# 10. Guardar la gráfica como imagen
plt.savefig( "grafica_tarea3_festa.png", dpi=300, bbox_inches="tight" )

# 11. Mostrar la gráfica
plt.show()