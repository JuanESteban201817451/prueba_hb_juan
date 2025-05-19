# Visualización
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns
import plotly.express as px
from matplotlib_venn import venn2

# Geoespacial
import geopandas as gpd
from shapely.geometry import Point, Polygon, MultiPolygon, MultiPoint
from libpysal.weights import KNN
from esda.moran import Moran, Moran_Local

# Interactividad
import folium
from folium.plugins import Search
from folium.features import GeoJsonTooltip
from branca.colormap import linear
import ipywidgets as widgets

# Utilidades
import pandas as pd
import numpy as np
import warnings
from IPython.display import display, clear_output, Markdown



def analizar_categoricas(df: pd.DataFrame, 
                         columnas_categoricas: list, top_n: int = 10):
    """
    Analiza columnas categóricas mostrando el número de categorías y las más frecuentes.

    Parámetros:
    -----------
    df : pd.DataFrame
        DataFrame de entrada.
    columnas_categoricas : list
        Lista de columnas categóricas definidas manualmente.
    top_n : int
        Número de categorías más frecuentes a mostrar.
    """
    for col in columnas_categoricas:
        print(f" **Columna:** {col}")
        print(f" Número de categorías únicas: {df[col].nunique()}")
        print(f" Top {top_n} categorías más frecuentes:")
        print(df[col].value_counts(dropna=False).head(top_n).to_frame(name='Frecuencia'))


def visualizar_tabla_geografica(gdf: gpd.GeoDataFrame, nombre_columna: str = "NOMBRE"):
    """
    Visualiza un GeoDataFrame como un mapa interactivo sobre Bogotá.

    Parámetros:
    -----------
    gdf : GeoDataFrame
        GeoDataFrame con geometría tipo Polygon o Point.
    nombre_columna : str
        Columna que se mostrará al hacer hover.

    Retorna:
    --------
    fig : plotly.graph_objects.Figure
        Mapa interactivo listo para fig.show()
    """
    # Copia segura
    gdf = gdf.copy()

    # Reproyección para centroides en metros
    if gdf.crs is None:
        raise ValueError("El GeoDataFrame no tiene CRS definido.")
    
    if gdf.crs.to_epsg() != 3116:
        gdf = gdf.to_crs(epsg=3116)

    # Calcular centroides y reproyectar a WGS84
    gdf_centroides = gdf.geometry.centroid.to_crs(epsg=4326)
    gdf.loc[:, "lon"] = gdf_centroides.x
    gdf.loc[:, "lat"] = gdf_centroides.y

    # Suprimir warnings si aparecen por centroides en CRS geográficos
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=UserWarning)

        fig = px.scatter_mapbox(
            gdf,
            lat="lat",
            lon="lon",
            hover_name=nombre_columna if nombre_columna in gdf.columns else None,
            zoom=9,
            height=600,
            mapbox_style="carto-positron"
        )

    return fig

def analizar_numericas(df: pd.DataFrame, columnas_numericas: list, bins: int = 50): 
    """
    Analiza distribución y outliers de columnas numéricas.
    
    Si la columna es 'precio' y existe 'tiponegocio', desglosa por 'venta' y 'arriendo'
    y se escala a millones de pesos (COP).

    Parámetros:
    -----------
    df : pd.DataFrame
        DataFrame con los datos.
    columnas_numericas : list
        Lista de nombres de columnas numéricas.
    bins : int
        Número de bins para los histogramas.
    """
    for col in columnas_numericas:
        print(f"\n**Variable numérica:** {col}")

        if col == 'precio' and 'tiponegocio' in df.columns:
            for tipo in ['venta', 'arriendo']:
                subset = df[df['tiponegocio'] == tipo]
                if subset.empty:
                    continue

                print(f"\n- Estadísticas para `{col}` en {tipo.upper()} (en millones):")
                print((subset[col] / 1_000_000).describe(percentiles=np.arange(0, 1.05, 0.05)))

                fig, axs = plt.subplots(1, 2, figsize=(12, 4))
                fig.suptitle(f"Distribución y Outliers - {col} ({tipo})", fontsize=14, weight="bold")

                datos = subset[col] / 1_000_000

                # Histograma
                sns.histplot(datos, bins=bins, ax=axs[0], kde=True, color="#5C2D91")
                axs[0].set_title("Distribución")
                axs[0].set_xlabel("Precio (millones COP)")

                # Boxplot
                sns.boxplot(x=datos, ax=axs[1], color="#5C2D91")
                axs[1].set_title("Outliers")
                axs[1].set_xlabel("Precio (millones COP)")

                plt.tight_layout()
                plt.show()

        else:
            print(df[col].describe(percentiles=np.arange(0, 1.05, 0.05)))

            fig, axs = plt.subplots(1, 2, figsize=(12, 4))
            fig.suptitle(f"Distribución y Outliers - {col}", fontsize=14, weight="bold")

            # Histograma
            sns.histplot(df[col], bins=bins, ax=axs[0], kde=True, color="#5C2D91")
            axs[0].set_title("Distribución")
            axs[0].set_xlabel(col)

            # Boxplot
            sns.boxplot(x=df[col], ax=axs[1], color="#5C2D91")
            axs[1].set_title("Outliers")
            axs[1].set_xlabel(col)

            plt.tight_layout()
            plt.show()



def graficar_upl_vs_inmuebles(
    gdf_upl: gpd.GeoDataFrame,
    gdf_inmuebles: gpd.GeoDataFrame,
    departamento: str = None,
    titulo: str = "Cobertura de UPL vs Inmuebles",
    color_geometry=None
) -> None:
    """
    Grafica los polígonos de UPL y los inmuebles (puntos, polígonos o multipolígonos).

    Parámetros
    ----------
    gdf_upl : gpd.GeoDataFrame
        GeoDataFrame con geometría de tipo polígono (UPL).

    gdf_inmuebles : gpd.GeoDataFrame
        GeoDataFrame con geometría de inmuebles (pueden ser puntos o polígonos).
        Puede incluir la columna 'departamento'.

    departamento : str, opcional
        Si se especifica, se filtran los inmuebles por esa columna.

    titulo : str
        Título del gráfico.
    """
    # Filtrar inmuebles por departamento si se especifica y existe
    if departamento and 'departamento' in gdf_inmuebles.columns:
        df_filtrado = gdf_inmuebles.query("departamento == @departamento")
    else:
        df_filtrado = gdf_inmuebles

    if df_filtrado.empty:
        print("No hay inmuebles para graficar con el criterio dado.")
        return
    

    if color_geometry:
        color_geometry=color_geometry
    else:
        color_geometry="#333333"

    # Crear figura
    fig, ax = plt.subplots(figsize=(10, 8))

    # Dibujar polígonos UPL
    gdf_upl.plot(ax=ax, color='#5C2D91', edgecolor='white', linewidth=0.4, alpha=0.9)

    # Detectar el tipo de geometría
    tipo_geom = df_filtrado.geometry.iloc[0].geom_type

    # Dibujar inmuebles
    if tipo_geom in ['Point', 'MultiPoint']:
        df_filtrado.plot(ax=ax, color=color_geometry, markersize=0.1)
    elif tipo_geom in ['Polygon', 'MultiPolygon']:
        df_filtrado.plot(ax=ax, facecolor="#333333", edgecolor=color_geometry, linewidth=0.9, alpha=0.9)
    else:
        print(f"Tipo de geometría no soportado: {tipo_geom}")
        return

    # Título profesional
    titulo_final = titulo.upper()
    if departamento:
        titulo_final += f" - {departamento.upper()}"

    ax.set_title(titulo_final, fontsize=18, color="#2D2D2D", weight="bold", loc="center", pad=20)
    ax.set_axis_off()
    plt.tight_layout()
    plt.show()

def widget_mapa_1(gdf_up: gpd.GeoDataFrame):
    """
    Visualización interactiva con mapa de UPLs y filtros desplegables.
    
    Parámetros:
    -----------
    gdf_up : GeoDataFrame
        GeoDataFrame con columnas: CODIGO_UPL, NOMBRE, VOCACION, SECTOR, ofertas, precio_total, AREA_HA, geometry
    """

    def crear_mapa_upls(df_filtrado):
        centro = [4.65, -74.1]
        mapa = folium.Map(location=centro, zoom_start=10.5, tiles="CartoDB positron")

        capa_upl = folium.FeatureGroup(name="UPLs", show=True).add_to(mapa)

        for _, row in df_filtrado.iterrows():
            popup_text = (
                f"<b>{row['NOMBRE']}</b><br>"
                f"<b>UPL:</b> {row['CODIGO_UPL']}<br>"
                f"<b>Vocación:</b> {row['VOCACION']}<br>"
                f"<b>Sector:</b> {row['SECTOR']}<br>"
                f"<b>Ofertas:</b> {row['ofertas']}<br>"
                f"<b>Precio total Viviendas:</b> ${row['precio_total'] / 1_000_000:,.1f} M<br>"
            )

            folium.GeoJson(
                row['geometry'],
                name=row['NOMBRE'],
                style_function=lambda feature: {
                    'fillColor': '#5C2D91',
                    'color': 'white',
                    'weight': 0.7,
                    'fillOpacity': 0.4
                },
                highlight_function=lambda feature: {
                    'weight': 3,
                    'color': '#333',
                    'fillOpacity': 0.5
                },
                tooltip=folium.Tooltip(popup_text, sticky=True)
            ).add_to(capa_upl)

        folium.LayerControl(collapsed=True).add_to(mapa)
        return mapa

    # Crear widgets
    nombre_dropdown = widgets.Dropdown(
        options=["Todos"] + sorted(gdf_up["NOMBRE"].dropna().unique().tolist()),
        description="Nombre:",
        layout=widgets.Layout(width="300px")
    )
    vocacion_dropdown = widgets.Dropdown(
        options=["Todos"] + sorted(gdf_up["VOCACION"].dropna().unique().tolist()),
        description="Vocación:",
        layout=widgets.Layout(width="300px")
    )
    sector_dropdown = widgets.Dropdown(
        options=["Todos"] + sorted(gdf_up["SECTOR"].dropna().unique().tolist()),
        description="Sector:",
        layout=widgets.Layout(width="300px")
    )
    upl_dropdown = widgets.Dropdown(
        options=["Todos"] + sorted(gdf_up["CODIGO_UPL"].dropna().unique().tolist()),
        description="Código UPL:",
        layout=widgets.Layout(width="300px")
    )

    def actualizar_mapa(change=None):
        clear_output(wait=True)
        display(Markdown("##  **Visualización Interactiva de UPLs y Ofertas Inmobiliarias en Bogotá**"))

        df = gdf_up.copy()
        if nombre_dropdown.value != "Todos":
            df = df[df["NOMBRE"] == nombre_dropdown.value]
        if vocacion_dropdown.value != "Todos":
            df = df[df["VOCACION"] == vocacion_dropdown.value]
        if sector_dropdown.value != "Todos":
            df = df[df["SECTOR"] == sector_dropdown.value]
        if upl_dropdown.value != "Todos":
            df = df[df["CODIGO_UPL"] == upl_dropdown.value]

        display(widgets.HBox([nombre_dropdown, vocacion_dropdown, sector_dropdown, upl_dropdown]))
        mapa = crear_mapa_upls(df)
        display(mapa)

    nombre_dropdown.observe(actualizar_mapa, names='value')
    vocacion_dropdown.observe(actualizar_mapa, names='value')
    sector_dropdown.observe(actualizar_mapa, names='value')
    upl_dropdown.observe(actualizar_mapa, names='value')

    actualizar_mapa()

def analizar_interseccion_codigos(df1: pd.DataFrame,
                                  df2: pd.DataFrame,
                                  columna: str = 'Barmanpre_estandar') -> pd.DataFrame:
    """
    Genera un diagrama de Venn y una tabla resumen mostrando la intersección y diferencias
    entre los códigos únicos de dos columnas estandarizadas de DataFrames. Además, muestra
    el porcentaje que representa cada categoría sobre el total combinado.

    Parámetros:
    -----------
    df1 : pd.DataFrame
        Primer DataFrame (por ejemplo, construcciones).
    df2 : pd.DataFrame
        Segundo DataFrame (por ejemplo, predios).
    columna : str
        Nombre de la columna estandarizada para comparar.

    Retorna:
    --------
    pd.DataFrame
        Tabla resumen con las cantidades y porcentajes por categoría.
    """
    # Crear conjuntos únicos a partir de las columnas proporcionadas
    codigos_1 = set(df1[columna].dropna())
    codigos_2 = set(df2[columna].dropna())

    # Calcular cantidades
    solo_1 = codigos_1 - codigos_2
    solo_2 = codigos_2 - codigos_1
    interseccion = codigos_1 & codigos_2

    total_unico = len(solo_1) + len(solo_2) + len(interseccion)

    # Crear tabla resumen
    tabla_resumen = pd.DataFrame({
        'Categoría': ['Solo DF1', 'Solo DF2', 'Intersección'],
        'Cantidad': [len(solo_1), len(solo_2), len(interseccion)],
    })
    tabla_resumen['Porcentaje (%)'] = (tabla_resumen['Cantidad'] / total_unico * 100).round(2)

    # Graficar diagrama de Venn
    plt.figure(figsize=(8, 6))
    venn = venn2([codigos_1, codigos_2], set_labels=('DF1', 'DF2'))
    if venn.get_patch_by_id('10'): venn.get_patch_by_id('10').set_color('#B497BD')
    if venn.get_patch_by_id('01'): venn.get_patch_by_id('01').set_color('#D1B2D8')
    if venn.get_patch_by_id('11'): venn.get_patch_by_id('11').set_color('#7A4988')
    plt.title("Intersección entre códigos estandarizados")
    plt.show()
    return tabla_resumen

def widget_tabla_propiedades(df: pd.DataFrame):
    """
    Crea una tabla interactiva con filtros para analizar propiedades por UPL.

    Parámetros
    ----------
    df : pd.DataFrame
        DataFrame con las columnas ['CODIGO_UPL','NOMBRE','VOCACION','SECTOR','n_predios_residenciales']
    """

    # Asegurar que no haya NaNs visibles
    df = df.copy()
    for col in ['CODIGO_UPL', 'NOMBRE', 'VOCACION', 'SECTOR']:
        df[col] = df[col].fillna('No_Determinado')

    # Crear filtros
    upl_dropdown = widgets.Dropdown(
        options=['Todos'] + sorted(df['CODIGO_UPL'].unique().tolist()),
        description='UPL:',
        layout=widgets.Layout(width='250px')
    )

    nombre_dropdown = widgets.Dropdown(
        options=['Todos'] + sorted(df['NOMBRE'].unique().tolist()),
        description='Nombre:',
        layout=widgets.Layout(width='300px')
    )

    vocacion_dropdown = widgets.Dropdown(
        options=['Todos'] + sorted(df['VOCACION'].unique().tolist()),
        description='Vocación:',
        layout=widgets.Layout(width='250px')
    )

    sector_dropdown = widgets.Dropdown(
        options=['Todos'] + sorted(df['SECTOR'].unique().tolist()),
        description='Sector:',
        layout=widgets.Layout(width='250px')
    )

    output = widgets.Output()

    def filtrar(_=None):
        with output:
            clear_output()

            df_filtrado = df.copy()
            if upl_dropdown.value != 'Todos':
                df_filtrado = df_filtrado[df_filtrado['CODIGO_UPL'] == upl_dropdown.value]
            if nombre_dropdown.value != 'Todos':
                df_filtrado = df_filtrado[df_filtrado['NOMBRE'] == nombre_dropdown.value]
            if vocacion_dropdown.value != 'Todos':
                df_filtrado = df_filtrado[df_filtrado['VOCACION'] == vocacion_dropdown.value]
            if sector_dropdown.value != 'Todos':
                df_filtrado = df_filtrado[df_filtrado['SECTOR'] == sector_dropdown.value]

            display(widgets.HTML(f"<h3 style='color:#5C2D91;'>Resumen de Predios Residenciales por UPL</h3>"))

            # Totales
            total = df_filtrado['n_predios_residenciales'].sum()
            display(widgets.HTML(
                f"<b>Total predios residenciales:</b> {total:,}"
            ))

            # Mostrar tabla ordenada
            display(df_filtrado.groupby(['CODIGO_UPL','NOMBRE']).agg(
                    predios_residenciales=('n_predios_residenciales', 'sum'),
                    ).reset_index().sort_values(by='predios_residenciales',ascending=False).reset_index(drop=True)
                    )

    # Asociar evento
    for w in [upl_dropdown, nombre_dropdown, vocacion_dropdown, sector_dropdown]:
        w.observe(filtrar, names='value')

    # Mostrar todo
    filtro_box = widgets.HBox([upl_dropdown, nombre_dropdown, vocacion_dropdown, sector_dropdown])
    display(filtro_box, output)

    filtrar()  # Mostrar primera vez

def graficar_percentil_histograma(
    df: pd.DataFrame,
    columna: str,
    percentil_min: float = 0,
    percentil_max: float = 100,
    bins: int = 200,
    linea_corte: float = None
):
    """
    Grafica un histograma filtrando por un rango de percentiles de una columna numérica.

    Parámetros:
    -----------
    df : pd.DataFrame
        DataFrame de entrada.
    columna : str
        Nombre de la columna numérica a analizar.
    percentil_min : float
        Percentil mínimo para filtrar (entre 0 y 100).
    percentil_max : float
        Percentil máximo para filtrar (entre 0 y 100).
    bins : int
        Número de bins para el histograma.
    linea_corte : float, opcional
        Valor en el eje x donde trazar una línea vertical de referencia (en negro).
    """
    assert 0 <= percentil_min < percentil_max <= 100, "Los percentiles deben estar entre 0 y 100 y min < max"

    # Eliminar NaNs y filtrar por percentiles
    serie = df[columna].dropna()
    p_min = serie.quantile(percentil_min / 100)
    p_max = serie.quantile(percentil_max / 100)
    subset = serie[(serie >= p_min) & (serie <= p_max)]

    print(f" **Variable:** {columna} | Percentiles {percentil_min}-{percentil_max}")
    print(subset.describe())

    # Graficar histograma
    plt.figure(figsize=(10, 4))
    ax = sns.histplot(subset, bins=bins, kde=True, color="#5C2D91")
    plt.title(f"Distribución de '{columna}' entre P{percentil_min} y P{percentil_max}", fontsize=13)
    plt.xlabel(columna)
    plt.ylabel("Frecuencia")

    # Desactivar notación científica en eje X
    ax.get_xaxis().set_major_formatter(ticker.FuncFormatter(lambda x, _: f'{int(x):,}'))

    # Línea de corte opcional
    if linea_corte is not None:
        plt.axvline(x=linea_corte, color='black', linestyle='--', linewidth=2)
        plt.text(linea_corte, ax.get_ylim()[1]*0.95, f'Corte: {linea_corte:,.0f}', 
                 rotation=90, verticalalignment='top', horizontalalignment='right', color='black')

    plt.tight_layout()
    plt.show()

def analizar_lisa_interactiva(
    gdf: gpd.GeoDataFrame,
    variables: list[str],
    k_vecinos: int = 5,
    alpha: float = 0.05,
    nombre_columna: str = 'NOMBRE'
) -> gpd.GeoDataFrame:
    """
    Realiza análisis de autocorrelación espacial global y local (LISA) 
    para un conjunto de variables en un GeoDataFrame y genera mapas interactivos usando folium.

    Parámetros:
    -----------
    gdf : gpd.GeoDataFrame
        GeoDataFrame con geometría y las variables numéricas a analizar.

    variables : list[str]
        Lista de nombres de las columnas numéricas para análisis LISA.

    k_vecinos : int
        Número de vecinos a considerar en el análisis KNN (default: 5).

    alpha : float
        Nivel de significancia para identificar clusters locales (default: 0.05).

    nombre_columna : str
        Columna con los nombres para mostrar en el tooltip (ej. 'NOMBRE').

    Retorna:
    --------
    gdf : gpd.GeoDataFrame
        El mismo GeoDataFrame con columnas extra para los resultados de LISA.
    """

    gdf = gdf.copy()

    # Reproyectar a CRS proyectado para evitar warning en centroides
    gdf_proj = gdf.to_crs(epsg=3116)
    centroids = gdf_proj.geometry.centroid
    coords = np.array([[pt.x, pt.y] for pt in centroids])

    # Crear pesos espaciales
    w = KNN.from_array(coords, k=k_vecinos)
    w.transform = 'r'

    color_dict = {
        'Alto-Alto': '#d7191c',
        'Bajo-Bajo': '#2c7bb6',
        'Alto-Bajo': '#fdae61',
        'Bajo-Alto': '#abd9e9',
        'No significativo': 'lightgrey'
    }

    for var in variables:
        print(f'\nMoran Global para {var}')
        mi = Moran(gdf[var], w)
        print(f"I de Moran: {mi.I:.4f}, p-valor: {mi.p_sim:.4f}")

        moran_loc = Moran_Local(gdf[var], w)
        sig = moran_loc.p_sim < alpha
        quadrant = moran_loc.q

        labels = ['No significativo'] * len(gdf)
        for i in range(len(gdf)):
            if sig[i]:
                if quadrant[i] == 1:
                    labels[i] = 'Alto-Alto'
                elif quadrant[i] == 2:
                    labels[i] = 'Bajo-Alto'
                elif quadrant[i] == 3:
                    labels[i] = 'Bajo-Bajo'
                elif quadrant[i] == 4:
                    labels[i] = 'Alto-Bajo'

        gdf[f'lisa_{var}'] = labels
        gdf[f'color_{var}'] = gdf[f'lisa_{var}'].map(color_dict)

        # Centro del mapa: usar centroides en WGS84 para folium
        gdf_wgs = gdf.to_crs(epsg=4326)
        center = [gdf_wgs.geometry.centroid.y.mean(), gdf_wgs.geometry.centroid.x.mean()]
        m = folium.Map(location=center, zoom_start=11, tiles="CartoDB Positron")

        # Mapa LISA interactivo
        folium.GeoJson(
            gdf_wgs,
            style_function=lambda feature: {
                'fillColor': feature['properties'][f'color_{var}'],
                'color': 'black',
                'weight': 0.5,
                'fillOpacity': 0.75,
            },
            tooltip=GeoJsonTooltip(
                fields=[nombre_columna, var, f'lisa_{var}'],
                aliases=['UPL:', f'{var}:', 'Clúster LISA:'],
                localize=True
            )
        ).add_to(m)

        # Leyenda HTML
        legend_html = '''
        <div style="position: fixed; bottom: 20px; left: 20px; width: 200px; 
                    border:2px solid grey; z-index:9999; font-size:14px;
                    background-color: white; padding: 10px;">
        <b>Clúster LISA</b><br>
        ''' + ''.join([
            f'<i style="background:{color_dict[k]};width:12px;height:12px;display:inline-block;margin-right:5px;"></i>{k}<br>'
            for k in color_dict]) + '</div>'
        m.get_root().html.add_child(folium.Element(legend_html))

        display(m)

    return gdf