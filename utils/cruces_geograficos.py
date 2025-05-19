from shapely.prepared import prep
from tqdm import tqdm

def preparar_upls(gdf_upl):
    """
    Prepara las geometrías de los polígonos para cruces rápidos.

    Parámetros:
    ----------
    gdf_upl : gpd.GeoDataFrame
        GeoDataFrame con geometría de polígonos y columna 'CODIGO_UPL'.

    Retorna:
    -------
    List[Tuple[shapely.PreparedGeometry, str]]
        Lista de tuplas (polígono preparado, código de UPL).
    """
    return [(prep(geom), codigo) for geom, codigo in zip(gdf_upl.geometry, gdf_upl['CODIGO_UPL'])]

def asignar_upl_contains(punto, upls_preparados):
    """
    Retorna el código de UPL si el punto está contenido dentro del polígono.
    """
    for pol, codigo in upls_preparados:
        if pol.contains(punto):
            return codigo
    return None

def cruzar_puntos_con_upl(gdf_puntos, 
                          upls_preparados, 
                          metodo_cruce, 
                          col_salida='upl_codigo'):
    """
    Aplica un método de cruce espacial a cada punto para asignar el código UPL.

    Parámetros:
    ----------
    gdf_puntos : gpd.GeoDataFrame
        GeoDataFrame con geometría tipo punto.
    upls_preparados : List[Tuple[PreparedGeometry, str]]
        Lista de geometrías preparadas.
    metodo_cruce : Callable
        Función que recibe un punto y la lista de UPLs preparados, y retorna el código.
    col_salida : str
        Nombre de la nueva columna a crear.

    Retorna:
    -------
    gpd.GeoDataFrame
        GeoDataFrame con la columna `col_salida` añadida.
    """
    tqdm.pandas(desc=f"Aplicando cruce: {col_salida}")
    gdf = gdf_puntos.copy()
    gdf[col_salida] = gdf.geometry.progress_apply(lambda p: metodo_cruce(p, upls_preparados))
    return gdf
