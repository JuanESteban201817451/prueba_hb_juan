import requests
import pandas as pd
import geopandas as gpd
import io
import gzip
import zipfile

def descargar_y_cargar_df(url: str) -> pd.DataFrame | gpd.GeoDataFrame:
    """
    Descarga y carga datos desde una URL en diferentes formatos a un DataFrame de pandas 
    o GeoDataFrame de geopandas, según corresponda.

    Parámetros:
    ----------
    url : str
        URL del archivo remoto que se desea descargar. Puede apuntar a archivos en formato 
        .json, .geojson, .gz (conteniendo .json o .csv), o .zip (conteniendo .csv, .json o .geojson).

    Formatos soportados:
    --------------------
    - .json         → Cargado como pd.DataFrame
    - .geojson      → Cargado como gpd.GeoDataFrame
    - .gz           → Detecta internamente si es JSON o CSV
    - .zip          → Detecta el tipo de archivo dentro del ZIP (.csv, .json, .geojson) y lo carga apropiadamente

    Retorna:
    -------
    pd.DataFrame o geopandas.GeoDataFrame
        Un DataFrame o GeoDataFrame con el contenido del archivo descargado.

    Excepciones:
    -----------
    ValueError:
        Se lanza si el archivo tiene un formato no soportado o si no puede interpretarse correctamente.
    """
    response = requests.get(url)
    response.raise_for_status()
    filename = url.split("/")[-1].lower()

    # GeoJSON directo
    if filename.endswith(".geojson"):
        return gpd.read_file(io.BytesIO(response.content))

    # JSON directo
    elif filename.endswith(".json"):
        return pd.read_json(io.BytesIO(response.content))

    # .gz comprimido (puede contener JSON o CSV)
    elif filename.endswith(".gz"):
        with gzip.GzipFile(fileobj=io.BytesIO(response.content)) as gz:
            try:
                return pd.read_json(gz)
            except ValueError:
                gz.seek(0)
                return pd.read_csv(gz)

    # ZIP (puede contener CSV, JSON o GEOJSON)
    elif filename.endswith(".zip"):
        with zipfile.ZipFile(io.BytesIO(response.content)) as z:
            for nombre in z.namelist():
                # GeoJSON
                if nombre.endswith(".geojson"):
                    with z.open(nombre) as f:
                        return gpd.read_file(f)
                # CSV
                elif nombre.endswith(".csv"):
                    with z.open(nombre) as f:
                        return pd.read_csv(f)
                # JSON
                elif nombre.endswith(".json"):
                    with z.open(nombre) as f:
                        return pd.read_json(f)

    else:
        raise ValueError(f"Formato de archivo no soportado: {filename}")

