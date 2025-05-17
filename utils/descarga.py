import requests
import pandas as pd
import geopandas as gpd
import io
import gzip
import zipfile

def descargar_y_cargar_df(url: str) -> pd.DataFrame | gpd.GeoDataFrame:
    """
    Descarga y carga datos desde una URL en diferentes formatos como DataFrame o GeoDataFrame.
    """

    response = requests.get(url)
    response.raise_for_status()
    filename = url.split("/")[-1].lower()
    content = io.BytesIO(response.content)

    # GeoJSON explícito
    if filename.endswith(".geojson"):
        return gpd.read_file(content)

    # JSON que puede ser GeoJSON
    elif filename.endswith(".json"):
        import json
        parsed = json.load(content)
        if isinstance(parsed, dict) and "features" in parsed and parsed.get("type") == "FeatureCollection":
            return gpd.GeoDataFrame.from_features(parsed["features"])
        else:
            content.seek(0)
            return pd.read_json(content)

    # .gz comprimido
    elif filename.endswith(".gz"):
        with gzip.GzipFile(fileobj=content) as gz:
            try:
                return pd.read_json(gz)
            except ValueError:
                gz.seek(0)
                return pd.read_csv(gz)

    # .zip con uno o más archivos
    elif filename.endswith(".zip"):
        with zipfile.ZipFile(content) as z:
            for nombre in z.namelist():
                with z.open(nombre) as f:
                    if nombre.endswith(".geojson"):
                        return gpd.read_file(f)
                    elif nombre.endswith(".csv"):
                        return pd.read_csv(f)
                    elif nombre.endswith(".json"):
                        import json
                        parsed = json.load(f)
                        if isinstance(parsed, dict) and "features" in parsed and parsed.get("type") == "FeatureCollection":
                            return gpd.GeoDataFrame.from_features(parsed["features"])
                        else:
                            f.seek(0)
                            return pd.read_json(f)

    else:
        raise ValueError(f"Formato de archivo no soportado: {filename}")

