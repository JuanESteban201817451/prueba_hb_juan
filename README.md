# PRUEBA_HB_JUAN

Repositorio de la prueba técnica para **Habi**.  
Contiene un flujo completo ―desde la descarga y el pre-procesamiento de datos geográficos hasta la generación de modelos y visualizaciones― empaquetado en un cuaderno de Jupyter y varios scripts auxiliares.

## 📘 Cómo abrir este cuaderno en Google Colab

Haz clic en el siguiente botón para abrir directamente el cuaderno en Google Colab:

[![Abrir en Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1mnoi86zEgi9kP-saYOmfRmgqkt9jrS7x?usp=sharing)



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
```


## ⚙️ Instalación

Puedes elegir entre **Conda** (recomendado) o **pip**:

### Opción 1 · Conda (idéntico al entorno original)

```bash
# Clona el repo
git clone https://github.com/tu-usuario/PRUEBA_HB_JUAN.git
cd PRUEBA_HB_JUAN

# Crea y activa el entorno
conda env create -f prueba_habi.yml
conda activate prueba_habi
````

### Opción 2 · pip + venv

```bash
python -m venv venv
source venv/bin/activate          # En Windows: venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt
```

> **Nota:** Si vas a ejecutar el notebook en **Google Colab**, basta con subir el `.ipynb`; Colab detectará e instalará la mayoría de los paquetes.
> Para que los widgets interactivos se rendericen, añade al comienzo del notebook:
>
> ```python
> from google.colab import output
> output.enable_custom_widget_manager()
> ```
>demás, recuerda verificar tanto los datos como las rutas de entrada, ya que podrían cambiar.
(Al hacer clic en el enlace encontrarás un notebook de Google Colab que se ejecuta por completo).
