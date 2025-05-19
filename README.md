# Prueba Habi  
**Juan Esteban Segura**

Este es un repositorio temporal que será deshabilitado una vez finalice el proceso de evaluación de la prueba técnica.

## 📘 Cómo abrir este cuaderno en Google Colab

Haz clic en el siguiente botón para abrir directamente el cuaderno en Google Colab:

[![Abrir en Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/JuanESteban201817451/prueba_hb_juan/Prueba_Data_Science.ipynb)



---

### 🔗 Opción 1: Desde GitHub

1. Ve a la página principal de este repositorio en GitHub.
2. Haz clic sobre el archivo `principal.ipynb`.
3. En la parte superior del archivo (donde se visualiza el código), haz clic en el botón **"Open in Colab"** (ícono de Colab: un círculo con un triángulo).

### 🌐 Opción 2: Desde Google Colab

1. Ingresa a [https://colab.research.google.com/](https://colab.research.google.com/).
2. En el menú superior, haz clic en **Archivo → Abrir cuaderno**.
3. Selecciona la pestaña **GitHub**.
4. En el campo de búsqueda, ingresa la siguiente ruta del repositorio:

# PRUEBA_HB_JUAN

Repositorio de la prueba técnica para **Habi**.  
Contiene un flujo completo ―desde la descarga y el pre-procesamiento de datos geográficos hasta la generación de modelos y visualizaciones― empaquetado en un cuaderno de Jupyter y varios scripts auxiliares.

## 🌳 Estructura de carpetas

```text
PRUEBA_HB_JUAN/
│
├── data/                    # Ubica aquí los datasets de trabajo (.csv, .parquet, shapefiles, etc.)
│
├── utils/                   # Módulos de apoyo reutilizables
│   ├── __pycache__/         # Caché de Python (se crea automáticamente)
│   ├── cruces_geograficos.py    # Cruces y joins espaciales
│   ├── descarga.py              # Funciones para descargar y descomprimir datasets
│   ├── mapas_y_graficas.py      # Gráficas estáticas e interactivas (Folium, Plotly, etc.)
│   ├── pipe_modelo.py           # Pipeline completo de entrenamiento y predicción
│   └── procesador.py            # Limpieza y transformación de datos
│
├── .gitignore               # Rutas y extensiones que no se suben al repo
├── Prueba_Data_Science.ipynb    # Notebook principal con la solución end-to-end
├── prueba_habi.yml          # Conda environment: recrea el entorno tal cual se usó
├── requirements.txt         # Lista de dependencias para instalación vía pip
└── README.md                # (Este archivo) Documentación y guía de uso
