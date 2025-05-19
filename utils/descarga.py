import geopandas as gpd
import pandas as pd
import requests
import gzip
import io
import zipfile
from typing import Optional, Union, Dict
import os
from urllib.request import urlretrieve
import tempfile
import hashlib
import xml.etree.ElementTree as ET

class ExtractorDatosRemotos:
    """
    Clase para descargar y guardar archivos remotos en disco sin transformarlos,
    y luego cargarlos localmente como DataFrame o GeoDataFrame según su extensión.
    """

    def __init__(self, url: str, carpeta_destino: str = "./data/interim") -> None:
        self.url = url
        self.carpeta_destino = carpeta_destino
        os.makedirs(carpeta_destino, exist_ok=True)

    def nombre_archivo_local(self) -> str:
        """Genera un nombre de archivo único basado en la URL."""
        nombre_archivo = self.url.split("/")[-1].split("?")[0]
        return os.path.join(self.carpeta_destino, nombre_archivo)

    def descargar_y_guardar(self) -> str:
        """Descarga el archivo desde la URL y lo guarda localmente."""
        ruta_local = self.nombre_archivo_local()
        if not os.path.exists(ruta_local):
            print(f"Descargando y guardando en: {ruta_local}")
            urlretrieve(self.url, ruta_local)
        else:
            print(f"Archivo ya existe localmente en: {ruta_local}")
        return ruta_local

    def cargar_desde_local(self, ruta_local: Optional[str] = None) -> Union[pd.DataFrame, gpd.GeoDataFrame, Dict[str, Union[pd.DataFrame, str]]]:
        """
        Carga el archivo local como DataFrame o GeoDataFrame según su extensión.
        
        Parámetros
        ----------
        ruta_local : str
            Ruta del archivo local. Si no se proporciona, se usa la generada por la URL.

        Retorna
        -------
        DataFrame, GeoDataFrame o dict según el contenido.
        """
        if ruta_local is None:
            ruta_local = self.nombre_archivo_local()

        if ruta_local.endswith(('.geojson', '.json')):
            return gpd.read_file(ruta_local)

        elif ruta_local.endswith('.csv.gz'):
            with gzip.open(ruta_local, 'rt') as f:
                return pd.read_csv(f)

        elif ruta_local.endswith('.csv'):
            return pd.read_csv(ruta_local, sep=';', low_memory=False)

        elif ruta_local.endswith('.zip'):
            resultados: Dict[str, Union[pd.DataFrame, str, gpd.GeoDataFrame]] = {}
            with zipfile.ZipFile(ruta_local, 'r') as z:
                for nombre in z.namelist():
                    nombre_lower = nombre.lower()
                    with z.open(nombre) as f:
                        if nombre_lower.endswith('.csv'):
                            try:
                                df = pd.read_csv(f, sep=';', low_memory=False)
                            except Exception:
                                f.seek(0)
                                df = pd.read_csv(f, low_memory=False)
                            resultados['csv'] = df

                        elif any(nombre_lower.endswith(ext) for ext in ['.geojson', '.json', '.geoson']):
                            resultados['geo'] = gpd.read_file(f)

                        elif nombre_lower.endswith('.xml') or nombre_lower.endswith('.txt'):
                            try:
                                df = pd.read_xml(f)
                                resultados['xml'] = df
                            except Exception:
                                f.seek(0)
                                contenido = f.read().decode('utf-8')
                                resultados['xml'] = contenido

            if not resultados:
                raise ValueError("No se encontró un archivo compatible dentro del .zip")
            return resultados if len(resultados) > 1 else list(resultados.values())[0]

        else:
            raise ValueError(f"Extensión de archivo no soportada: {ruta_local}")

def extraer_procesos_xml(xml_text: Union[str, bytes]) -> pd.DataFrame:
    """
    Extrae los procesos contenidos en el nodo <lineage> de un XML de metadatos de ArcGIS 
    y los convierte en un DataFrame.

    Parámetros
    ----------
    xml_text : str o bytes
        Contenido XML como cadena o bytes.

    Retorna
    -------
    pd.DataFrame
        Tabla con las columnas: ToolSource, Date, Time y Description.
    """
    # Parsear XML
    root = ET.fromstring(xml_text)

    # Buscar los procesos dentro de <lineage>
    processes = root.findall(".//lineage/Process")

    # Extraer atributos y texto
    rows = []
    for proc in processes:
        rows.append({
            "ToolSource": proc.attrib.get("ToolSource", ""),
            "Date": proc.attrib.get("Date", ""),
            "Time": proc.attrib.get("Time", ""),
            "Description": proc.text.strip() if proc.text else ""
        })

    return pd.DataFrame(rows)