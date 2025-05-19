import pandas as pd
import geopandas as gpd
import numpy as np
from shapely.geometry import Point
from sklearn.neighbors import NearestNeighbors
from tqdm import tqdm



def estandarizar_codigo(df:pd.DataFrame,
                        columna_objetivo:str,
                        key:str,
                        largo:int)->pd.DataFrame:
    """
    Estandariza una columna de códigos, convirtiéndola a string y 
    rellenando con ceros a la izquierda hasta alcanzar un largo deseado.

    Parámetros
    ----------
    df : pd.DataFrame
        DataFrame original al que se le agregará la columna estandarizada.
    columna_objetivo : str
        Nombre de la columna que contiene los códigos a estandarizar.
    key : str
        Nombre de la nueva columna que se creará con los códigos estandarizados.
    largo : int
        Largo fijo al que deben ajustarse los códigos.

    Retorna
    -------
    pd.DataFrame
        El DataFrame original con la nueva columna agregada.
    """
    df = df.copy()
    df[key] = df[columna_objetivo].astype(str).str.zfill(largo).str[-largo:]
    return df

def obtener_centroides(gdf: gpd.GeoDataFrame, 
                       crs_proyectado: str = "EPSG:3116") -> gpd.GeoDataFrame:
    """
    Calcula los centroides de geometrías de un GeoDataFrame, usando un CRS proyectado para mayor precisión.
    
    Parámetros
    ----------
    gdf : gpd.GeoDataFrame
        GeoDataFrame original con geometrías tipo polígono.
    crs_proyectado : str
        Código EPSG de un CRS proyectado adecuado (por defecto: "EPSG:3116" para Colombia).
    
    Retorna
    -------
    gpd.GeoDataFrame
        Nuevo GeoDataFrame con los mismos datos pero geometría de centroides en el CRS original.
    """
    gdf_proj = gdf.to_crs(crs_proyectado)
    centroides = gdf_proj.geometry.centroid.to_crs(gdf.crs)
    
    return gpd.GeoDataFrame(gdf.drop(columns="geometry").copy(), geometry=centroides, crs=gdf.crs)

def imputar_area_anomala(df: gpd.GeoDataFrame, 
                          area_col='area', 
                          precio_col='precio', 
                          id_col='id', 
                          tipo_col='tipo_inmueble',
                          negocio_col='tiponegocio',
                          n_vecinos=15) -> gpd.GeoDataFrame:
    """
    Imputa valores anómalos de área en un GeoDataFrame usando lógica de errores de lectura y vecinos cercanos.

    - Si area < 25, prueba *area * 10* y *area * 100*.
    - Si area > 2100, prueba *area * 0.1* y *area * 0.01*.
    - Si ninguna opción está cerca del área estimada por vecinos, usa directamente esa estimación.

    Parámetros:
    -----------
    df : gpd.GeoDataFrame
        GeoDataFrame con columnas de geometría, precio y área.
    area_col : str
        Nombre de la columna de área.
    precio_col : str
        Nombre de la columna de precio.
    id_col : str
        Columna que contiene los identificadores únicos.
    tipo_col : str
        Columna que indica el tipo de inmueble (casa, apartamento).
    negocio_col : str
        Columna que indica si es venta o arriendo.
    n_vecinos : int
        Número de vecinos cercanos a usar para estimación.

    Retorna:
    --------
    df : gpd.GeoDataFrame
        Con columna `area_imputada` dummy e imputaciones aplicadas.
    """
    df = df.copy()
    df["area_imputada"] = 0
    df_proj = df.to_crs(epsg=3116)
    cambios = []

    subset_anomalos = df_proj.query(f"{area_col} < 25 or {area_col} > 2100 and {precio_col}.notnull()", engine='python')

    if subset_anomalos.empty:
        return df, pd.DataFrame()

    for _, row in subset_anomalos.iterrows():
        idx = row[id_col]
        precio = row[precio_col]
        area_original = row[area_col]
        geom = row.geometry
        tipo = row[tipo_col]
        negocio = row[negocio_col]

        if pd.isnull(precio):
            continue

        # Generar candidatos válidos (mismo tipo y negocio)
        df_vecinos = df_proj[
            (df_proj[id_col] != idx) &
            (df_proj[precio_col].notnull()) &
            (df_proj[area_col].between(25, 2100)) &
            (df_proj[tipo_col] == tipo) &
            (df_proj[negocio_col] == negocio)
        ].copy()

        if df_vecinos.empty:
            continue

        df_vecinos["x"] = df_vecinos.geometry.x
        df_vecinos["y"] = df_vecinos.geometry.y

        nn = NearestNeighbors(n_neighbors=min(n_vecinos, len(df_vecinos)), algorithm="ball_tree")
        nn.fit(df_vecinos[["x", "y"]])

        punto_df = pd.DataFrame([[geom.x, geom.y]], columns=["x", "y"])
        _, indices = nn.kneighbors(punto_df)
        vecinos = df_vecinos.iloc[indices[0]]
        m2_promedio = np.mean(vecinos[precio_col] / vecinos[area_col])
        area_estimada = precio / m2_promedio

        # Pruebas con múltiplos y divisiones
        alternativas = []
        if area_original < 25:
            alternativas = [area_original,area_original * 10, area_original * 100]
        elif area_original > 2100:
            alternativas = [area_original, area_original * 0.1, area_original * 0.01]

        usada = "estimada"
        area_final = area_estimada
        for alt in alternativas:
            if abs(alt - area_estimada) <= 15:
                area_final = alt
                usada = f"alternativa_{alt:.2f}"
                break

        # Aplicar imputación
        df.loc[df[id_col] == idx, area_col] = area_final
        df.loc[df[id_col] == idx, "area_imputada"] = 1
        cambios.append((idx, area_original, area_final, usada))

    cambios_df = pd.DataFrame(cambios, columns=["id", "area_original", "area_imputada", "metodo"])
    return df, cambios_df

def corregir_valores_grandes(df, id_col='id', precio_col='precio', area_col='area'):
    df = df.copy()
    cambios = []

    for idx, row in df.iterrows():
        precio = row[precio_col]
        area = row[area_col]

        # Saltar si precio o área es NaN
        if pd.isna(precio) or pd.isna(area):
            continue

        precio_str = f"{int(precio)}"  # Quitar decimales

        if len(precio_str) >= 14:
            parte_inicial = precio_str[:7]
            parte_final = precio_str[-7:]

            nuevo_precio = (int(parte_inicial) + int(parte_final)) / 2

            cambios.append({
                "id": row[id_col],
                "precio_viejo": precio,
                "precio_nuevo": nuevo_precio
            })

            df.at[idx, precio_col] = nuevo_precio

    cambios_df = pd.DataFrame(cambios)
    return df, cambios_df

def imputar_precio(df: gpd.GeoDataFrame,
                   precio_col='precio',
                   area_col='area',
                   id_col='id',
                   tipo_col='tipo_inmueble',
                   negocio_col='tiponegocio',
                   direccion_col='direccion',
                   n_vecinos=15,
                   umbral_precio=5_000_000_000,
                   margen_error=50_000_000) -> tuple[gpd.GeoDataFrame, pd.DataFrame]:

    df = df.copy()
    df['precio_imputado'] = 0
    df_proj = df.to_crs(epsg=3116)
    cambios = []

    subset = df_proj[(df_proj[precio_col].isna()) | (df_proj[precio_col] > umbral_precio)]

    for _, row in tqdm(subset.iterrows(), total=len(subset), desc="Imputando precios"):
        idx = row[id_col]
        area = row[area_col]
        precio_actual = row[precio_col]
        direccion = row[direccion_col]
        tipo = row[tipo_col]
        negocio = row[negocio_col]
        geom = row.geometry

        if pd.isna(area) or area == 0:
            continue  # no podemos estimar sin área

        # 1. Buscar vecinos con misma dirección y diferente precio
        mismos_vecinos = df_proj[
            (df_proj[direccion_col] == direccion) &
            (df_proj[precio_col].notnull()) &
            (df_proj[id_col] != idx) &
            (df_proj[tipo_col] == tipo) &
            (df_proj[negocio_col] == negocio)
        ]

        if not mismos_vecinos.empty:
            precio_estimado = mismos_vecinos[precio_col].mean()
            metodo = 'direccion_misma'
        else:
            # 2. Vecinos más cercanos con mismas características
            candidatos = df_proj[
                (df_proj[precio_col].notnull()) &
                (df_proj[area_col] > 0) &
                (df_proj[id_col] != idx) &
                (df_proj[tipo_col] == tipo) &
                (df_proj[negocio_col] == negocio)
            ].copy()

            if candidatos.empty:
                continue

            candidatos["x"] = candidatos.geometry.x
            candidatos["y"] = candidatos.geometry.y

            nn = NearestNeighbors(n_neighbors=min(n_vecinos, len(candidatos)))
            nn.fit(candidatos[["x", "y"]])

            punto_df = pd.DataFrame([[geom.x, geom.y]], columns=["x", "y"])
            _, indices = nn.kneighbors(punto_df)
            vecinos = candidatos.iloc[indices[0]]
            m2_prom = np.mean(vecinos[precio_col] / vecinos[area_col])
            precio_estimado = m2_prom * area
            metodo = 'vecinos_proximos'

        # 3. Elegir entre candidatos: original, original/10, estimado
        candidatos_precio = [precio_estimado]
        if not pd.isna(precio_actual):
            candidatos_precio.extend([precio_actual, precio_actual / 10])
            diferencias = [abs(p - precio_estimado) for p in candidatos_precio]
            posibles = [p for p, diff in zip(candidatos_precio, diferencias)
                        if diff <= margen_error]
            if posibles:
                precio_final = min(posibles, key=lambda p: abs(p - precio_estimado))
                metodo += "_ajuste"
            else:
                precio_final = precio_estimado
        else:
            precio_final = precio_estimado

        df.loc[df[id_col] == idx, precio_col] = precio_final
        df.loc[df[id_col] == idx, 'precio_imputado'] = 1
        cambios.append((idx, precio_actual, precio_final, metodo))

    cambios_df = pd.DataFrame(cambios, columns=['id', 'precio_original', 'precio_imputado', 'metodo'])
    return df, cambios_df