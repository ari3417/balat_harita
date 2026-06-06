import streamlit as st
import os
import pandas as pd
import geopandas as gpd
import osmnx as ox
import json
import cv2
import numpy as np
import base64
import math
from shapely.geometry import Point, mapping
from sklearn.cluster import KMeans
from bs4 import BeautifulSoup
import re
import streamlit.components.v1 as components

# Streamlit sayfası tam ekran olsun
st.set_page_config(layout="wide", page_title="Balat Semantic Morfoloji")

# ==========================================
# GITHUB KLASÖR YOLLARI (Colab yolları değil, bulunduğumuz klasör)
# ==========================================
EXCEL_V4 = 'Balat_Soyut_Tipoloji_Analizi(5).xlsx'
EXCEL_GIS = 'Balat_Kesinlesmis_Bina_Koordinatlari.xlsx'
KLASOR_SOYUT_KIRPILMIS = 'balat-soyut-kirpilmis/'
KLASOR_SEMBOLIK_GERCEK = 'balat_sembolik_binalar_gerçek_fotoğraflar/'
KLASOR_SEMBOLIK_ABSTRACT = 'balat_sembolik_binalar_abstract/'

# st.cache_data ile bu ağır işlemi sunucuda sadece 1 KERE yapıyoruz! (Kasmayı engeller)
@st.cache_data
def haritayi_olustur():
    # 1. Veri Okuma
    df_v4 = pd.read_excel(EXCEL_V4)
    df_gis = pd.read_excel(EXCEL_GIS)

    def id_cikar(isim):
        try: return int(str(isim).split('_')[1]) 
        except: return 0

    df_v4['Cikartilan_ID'] = df_v4['Fotoğraf İsmi'].apply(id_cikar)
    df_gis['Bina_ID'] = pd.to_numeric(df_gis['Bina_ID'], errors='coerce')
    df = pd.merge(df_gis, df_v4, left_on='Bina_ID', right_on='Cikartilan_ID', how='inner')
    df = df.dropna(subset=['Bina_Lat', 'Bina_Lon']) 

    # 2. OSM Verisi
    place_name = "Balat, Istanbul, Turkey"
    binalar_gdf = ox.features_from_place(place_name, tags={"building": True})
    binalar_gdf = binalar_gdf.to_crs(epsg=32635) 

    yollar_gdf = ox.features_from_place(place_name, tags={"highway": True})
    yollar_gdf = yollar_gdf[yollar_gdf.geometry.type.isin(['LineString', 'MultiLineString'])]
    yollar_gdf = yollar_gdf.to_crs(epsg=4326)
    yollar_geojson = yollar_gdf.to_json()

    geometry = [Point(xy) for xy in zip(df['Bina_Lon'], df['Bina_Lat'])]
    noktalar_gdf = gpd.GeoDataFrame(df, geometry=geometry, crs="EPSG:4326").to_crs(epsg=32635)
    binalar_gdf = binalar_gdf.reset_index()

    for coord, group in noktalar_gdf.groupby(['Bina_Lat', 'Bina_Lon']):
        pt = group.geometry.iloc[0]
        circle = pt.buffer(25) 
        intersecting = binalar_gdf[binalar_gdf.intersects(circle)]
        if len(intersecting) == 0:
            nearest_idx = binalar_gdf.geometry.distance(pt).idxmin()
            bldg_indices = [nearest_idx]
        else:
            intersecting = intersecting.copy()
            intersecting['dist'] = intersecting.geometry.distance(pt)
            intersecting = intersecting.sort_values('dist')
            bldg_indices = intersecting.index.tolist()
            
        for i, (idx, row) in enumerate(group.iterrows()):
            assigned_bldg = bldg_indices[i % len(bldg_indices)]
            noktalar_gdf.loc[idx, 'index_right'] = assigned_bldg
            
    eslesen_veriler = noktalar_gdf.copy()
    eslesen_veriler = eslesen_veriler.to_crs(epsg=4326)
    binalar_gdf = binalar_gdf.to_crs(epsg=4326)
    eslesen_veriler['poly_geom'] = binalar_gdf.loc[eslesen_veriler['index_right'], 'geometry'].values

    # 3. Görsel İşleme ve Base64
    def get_hex_and_base64(image_path):
        if not os.path.exists(image_path): return "#cccccc", ""
        img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
        if img is None: return "#cccccc", ""
        
        if len(img.shape) == 3 and img.shape[2] == 4:
            vis = img[img[:, :, 3] > 0]
            avg = np.average(vis, axis=0) if len(vis) > 0 else [200, 200, 200]
        else:
            h, w = img.shape[:2]
            avg = np.average(np.average(img[int(h*0.4):int(h*0.6), int(w*0.4):int(w*0.6)], axis=0), axis=0)
        hex_color = "#{:02x}{:02x}{:02x}".format(int(avg[2]), int(avg[1]), int(avg[0]))
        
        h, w = img.shape[:2]
        oran = 45.0 / h 
        yeni_w = max(int(w * oran), 8)
        img_kucuk = cv2.resize(img, (yeni_w, 45), interpolation=cv2.INTER_AREA)
        _, buffer = cv2.imencode('.png', img_kucuk)
        b64_string = "data:image/png;base64," + base64.b64encode(buffer).decode('utf-8')
        return hex_color, b64_string

    def hex_to_rgb(hx): return [int(hx.lstrip('#')[0:2],16), int(hx.lstrip('#')[2:4],16), int(hx.lstrip('#')[4:6],16)]
    def rgb_to_hex(rgb): return "#{:02x}{:02x}{:02x}".format(int(rgb[0]), int(rgb[1]), int(rgb[2]))

    rgb_list, ozellikler, hacimler = [], [], []

    for idx, row in eslesen_veriler.iterrows():
        dosya_yolu = os.path.join(KLASOR_SOYUT_KIRPILMIS, row['Dosya_Adi'])
        hex_renk, b64_img = get_hex_and_base64(dosya_yolu)
        
        try: aks = sum([int(s) for s in re.findall(r'\d+', str(row['Duvar / Cumba Aksı']))])
        except: aks = 2
        hacim = max(1, int(row['Kat Sayısı']) * aks)
        hacimler.append(hacim)

        malzemeler = str(row['Kat Malzemeleri']).split(' | ')
        foto_ismi = str(row.get('Fotoğraf İsmi', ''))
        sokak_ismi = "Bilinmiyor"
        for part in foto_ismi.split('_'):
            if "Cd" in part or "Sk" in part or "Sok" in part or "Cad" in part:
                sokak_ismi = part
                break
        if sokak_ismi == "Bilinmiyor":
            parts = foto_ismi.split('_')
            if len(parts) > 2: sokak_ismi = parts[2]
                
        kat_val = str(row.get('Kat Sayısı', 'Belirtilmemiş')).replace('.0', '') if str(row.get('Kat Sayısı', 'nan')).strip().lower() != 'nan' else 'Belirtilmemiş'
        zemin_val = 'Belirtilmemiş'
        for olasi_kolon in ['Zemin Kat Fonksiyonu', 'Zemin Kat (Fonksiyon)', 'Zemin Kat', 'Fonksiyon']:
            if olasi_kolon in row.index and pd.notna(row[olasi_kolon]) and str(row[olasi_kolon]).strip().lower() != 'nan':
                zemin_val = str(row[olasi_kolon]).strip()
                break
                
        ozellikler.append({
            "id": str(row['Bina_ID']), "img": b64_img, "malzeme": malzemeler[0] if malzemeler else "Sıvalı",
            "renk_orj": hex_renk, "sokak": sokak_ismi, "kat": kat_val, "zemin": zemin_val
        })
        rgb_list.append(hex_to_rgb(hex_renk))

    X = np.array(rgb_list)
    W = np.array(hacimler)
    palet_10 = KMeans(n_clusters=min(10, len(X)), random_state=42).fit(X, sample_weight=W)
    palet_5 = KMeans(n_clusters=min(5, len(X)), random_state=42).fit(X, sample_weight=W)
    palet_3 = KMeans(n_clusters=min(3, len(X)), random_state=42).fit(X, sample_weight=W)
    palet_1 = KMeans(n_clusters=1, random_state=42).fit(X, sample_weight=W)

    unique_rgb = np.unique(X, axis=0)
    palet_25_model = KMeans(n_clusters=min(25, len(unique_rgb)), random_state=42).fit(unique_rgb)

    def get_vibrancy(c):
        r, g, b = float(c[0]), float(c[1]), float(c[2])
        chroma = max(r, g, b) - min(r, g, b)
        return chroma * 3.0 + r * 1.5 + g * 1.0 - b * 0.5 

    palet_25_colors = []
    for i in range(palet_25_model.n_clusters):
        cluster_colors = unique_rgb[palet_25_model.labels_ == i]
        best_c = max(cluster_colors, key=get_vibrancy) if len(cluster_colors) > 0 else palet_25_model.cluster_centers_[i]
        palet_25_colors.append(rgb_to_hex(best_c))

    labels_25_full = palet_25_model.predict(X)
    most_dominant_color = palet_25_colors[np.argmax(np.bincount(labels_25_full))]

    features_geojson = []
    poly_groups = {}
    for i, row in enumerate(eslesen_veriler.itertuples()):
        idx_right = row.index_right
        if idx_right not in poly_groups: poly_groups[idx_right] = []
        poly_groups[idx_right].append(i)

    for poly_idx, indices in poly_groups.items():
        geom = binalar_gdf.loc[poly_idx, 'geometry']
        coords = mapping(geom)
        n = len(indices)
        step_size = 0.00004 
        for j, i in enumerate(indices):
            factor = j - (n - 1) / 2.0
            lat = geom.centroid.y + math.sin(0) * step_size * factor
            lon = geom.centroid.x + math.cos(0) * step_size * factor

            props = ozellikler[i].copy()
            props["c10"] = rgb_to_hex(palet_10.cluster_centers_[palet_10.labels_[i]])
            props["c5"] = rgb_to_hex(palet_5.cluster_centers_[palet_5.labels_[i]])
            props["c3"] = rgb_to_hex(palet_3.cluster_centers_[palet_3.labels_[i]])
            props["c1"] = rgb_to_hex(palet_1.cluster_centers_[palet_1.labels_[i]])
            props["c25_closest"] = palet_25_colors[labels_25_full[i]]
            props["center_lat"] = lat
            props["center_lon"] = lon

            features_geojson.append({
                "type": "Feature", "geometry": coords, "properties": props
            })

    features_geojson.sort(key=lambda f: get_vibrancy(hex_to_rgb(f['properties']['renk_orj'])))
    geojson_data = json.dumps({"type": "FeatureCollection", "features": features_geojson})
    palet_json = json.dumps(palet_25_colors)

    # 4. KML ve Manuel Pinler
    manuel_pin_data = []
    if os.path.exists("binalar.kml"):
        with open("binalar.kml", "r", encoding="utf-8") as file:
            soup = BeautifulSoup(file, "xml")
            for pm in soup.find_all("Placemark"):
                name_tag = pm.find("name")
                coord_tag = pm.find("coordinates")
                if name_tag and coord_tag:
                    isim = name_tag.text.strip()
                    coords = coord_tag.text.strip().split(',')
                    if len(coords) >= 2:
                        bulunan_b64 = ""
                        for ext in ['.png', '.jpg', '.jpeg']:
                            yol = os.path.join(KLASOR_SEMBOLIK_ABSTRACT, isim + ext)
                            if os.path.exists(yol):
                                img = cv2.imread(yol, cv2.IMREAD_UNCHANGED)
                                if img is not None:
                                    h, w = img.shape[:2]
                                    oran = 45.0 / h
                                    yeni_w = max(int(w * oran), 8)
                                    img_kucuk = cv2.resize(img, (yeni_w, 45), interpolation=cv2.INTER_AREA)
                                    _, buffer = cv2.imencode('.png', img_kucuk)
                                    bulunan_b64 = "data:image/png;base64," + base64.b64encode(buffer).decode('utf-8')
                                break
                        manuel_pin_data.append({"isim": isim, "lat": float(coords[1]), "lon": float(coords[0]), "b64": bulunan_b64})
    manuel_pin_json = json.dumps(manuel_pin_data)

    # 5. Intro Klasörleri
    def load_folder_to_b64(folder_path):
        arr = []
        if os.path.exists(folder_path):
            for img_name in os.listdir(folder_path):
                if img_name.endswith(('.png', '.jpg', '.jpeg')):
                    img_path = os.path.join(folder_path, img_name)
                    img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
                    if img is not None:
                        h, w = img.shape[:2]
                        oran = 300.0 / h 
                        yeni_w = max(int(w * oran), 8)
                        img_kucuk = cv2.resize(img, (yeni_w, 300), interpolation=cv2.INTER_AREA)
                        _, buffer = cv2.imencode('.png', img_kucuk)
                        arr.append("data:image/png;base64," + base64.b64encode(buffer).decode('utf-8'))
        return arr

    landmarks_gercek_json = json.dumps(load_folder_to_b64(KLASOR_SEMBOLIK_GERCEK))
    landmarks_abstract_json = json.dumps(load_folder_to_b64(KLASOR_SEMBOLIK_ABSTRACT))

    sinir_gdf = ox.geocode_to_gdf("Balat, Istanbul, Turkey")
    balat_geojson = sinir_gdf.to_json()

    min_lat, max_lat = binalar_gdf.geometry.centroid.y.min(), binalar_gdf.geometry.centroid.y.max()
    min_lon, max_lon = binalar_gdf.geometry.centroid.x.min(), binalar_gdf.geometry.centroid.x.max()

    # HTML OLUŞTURMA
    html_template = f"""
    <!DOCTYPE html>
    <html lang="tr">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Balat Semantic Morfoloji</title>
        <link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css" />
        <script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
        <link href="https://fonts.googleapis.com/css2?family=Fredoka+One&family=Varela+Round&display=swap" rel="stylesheet">
        <link href="https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@700&display=swap" rel="stylesheet">
        
        <style>
            body, html {{ margin: 0; padding: 0; width: 100%; height: 100vh; font-family: 'Varela Round', sans-serif; background: #fff; overflow: hidden; }}
            #map {{ width: 100%; height: 100%; background: #ffffff; z-index: 1; }}
            
            #sidebar-wrapper {{ position: absolute; top: 0; left: 0; width: 260px; height: 100%; z-index: 2000; transition: transform 0.4s ease; pointer-events: none; }}
            #sidebar-wrapper.closed {{ transform: translateX(-260px); }}
            
            #sidebar {{ position: absolute; top: 0; left: 0; width: 100%; height: 100%; background: #eacbd0; box-shadow: 2px 0 10px rgba(0,0,0,0.1); padding: 0; overflow-y: auto; pointer-events: auto; }}
            #sidebar::-webkit-scrollbar {{ width: 0px; background: transparent; }}
            
            #toggle-btn {{ position: absolute; top: 20px; right: -25px; width: 25px; height: 40px; background: #ea1f48; color: white; border: none; font-size: 18px; cursor: pointer; border-radius: 0 15px 15px 0; font-weight: bold; display: flex; align-items: center; justify-content: center; box-shadow: 3px 0 5px rgba(0,0,0,0.15); pointer-events: auto; font-family: monospace; }}
            
            .sidebar-header {{ background: #ea1f48; color: white; padding: 15px 20px; font-family: 'Fredoka One', cursive; font-size: 22px; line-height: 1.1; border-bottom-right-radius: 15px; margin-bottom: 15px; letter-spacing: 0.5px; }}
            .sidebar-content {{ padding: 0 20px 20px 20px; display: flex; flex-direction: column; gap: 10px; }}
            
            .pill-btn {{ width: 100%; padding: 10px; background: white; border: 2px dashed #ea1f48; border-radius: 20px; color: #ea1f48; font-weight: bold; cursor: pointer; transition: 0.2s; font-family: 'Fredoka One', cursive; font-size: 14px; text-transform: uppercase; }}
            .pill-btn.active {{ background: #ea1f48; color: white; border-style: solid; }}
            
            .white-box {{ background: white; border-radius: 25px; padding: 15px; box-shadow: 0 4px 10px rgba(0,0,0,0.05); position: relative; margin-top: 5px; text-align: center; }}
            
            .box-title {{ font-family: 'Fredoka One', cursive; font-size: 24px; margin: 10px 0 5px 0; text-transform: uppercase; }}
            .title-blue {{ color: #0f8cc6; }}
            .title-orange {{ color: #f2623a; }}
            .desc {{ font-size: 11px; color: #000; line-height: 1.3; margin-top: 0; margin-bottom: 10px; font-weight: bold; }}

            .mat-container {{ display: flex; flex-direction: column; align-items: flex-start; gap: 5px; margin-bottom: 10px; padding-left: 10px; }}
            .mat-row {{ display: flex; align-items: center; cursor: pointer; transition: 0.2s; position: relative; width: 100%; }}
            .mat-row:hover {{ transform: translateX(5px); }}
            .mat-row.inactive {{ opacity: 0.3; filter: grayscale(100%); }}
            
            .mat-icon-wrapper {{ width: 45px; height: 25px; position: relative; margin-right: 15px; border-radius: 6px; overflow: hidden; border: 1px solid #333; background: #fff; flex-shrink:0; }}
            
            .icon-tugla-bg {{ width: 100%; height: 100%; background: linear-gradient(90deg, #333 1px, transparent 1px) 0 0, linear-gradient(#333 1px, transparent 1px) 0 0; background-size: 10px 8px; position: absolute; z-index: 2; }}
            .icon-ahsap-bg {{ width: 100%; height: 100%; background: repeating-linear-gradient(180deg, transparent, transparent 6px, #333 6px, #333 7px); position: absolute; z-index: 2; }}
            .icon-sivali-bg {{ width: 100%; height: 100%; background: radial-gradient(#333 1px, transparent 1px); background-size: 6px 6px; position: absolute; z-index: 2; }}
            
            .mat-text {{ font-size: 14px; color: #333; font-family: 'Varela Round', sans-serif; z-index: 10; position: relative; font-weight: normal; }}

            .color-graphic {{ width: 100%; height: 120px; background: #fff; border-radius: 15px; position: relative; overflow: hidden; margin-bottom: 5px; display: grid; grid-template-columns: repeat(5, 1fr); grid-template-rows: repeat(5, 1fr); gap: 0; }}
            .color-graphic::before {{ content: ''; position: absolute; top:0; left:0; width:100%; height:100%; border-radius: 15px; pointer-events: none; z-index: 20; box-shadow: inset 0 0 0 2px #fff; }}
            
            .color-circle-btn {{ width: 150%; height: 150%; border-radius: 50%; transform: translate(-15%, -15%); transition: 0.2s ease-in-out; cursor: pointer; border: none; padding: 0; position: relative; z-index: 10; }}
            .color-circle-btn:hover {{ transform: translate(-15%, -15%) scale(1.2); z-index: 15; box-shadow: 0 0 5px rgba(0,0,0,0.5); }}

            .dominant-color-container {{ display: flex; align-items: center; justify-content: center; gap: 10px; margin-top: 15px; cursor: pointer; transition: 0.2s; padding-bottom: 10px; }}
            .dominant-color-container:hover {{ transform: scale(1.05); }}
            .dominant-circle {{ width: 32px; height: 32px; border-radius: 50%; background: {most_dominant_color}; box-shadow: 0 2px 5px rgba(0,0,0,0.2); flex-shrink: 0; }}
            .dominant-text {{ font-family: 'Varela Round', sans-serif; font-size: 13px; color: #333; font-weight: bold; text-align: left; line-height: 1.1; }}

            .bottom-tab {{ position: absolute; bottom: 85px; left: 260px; background: #e31836; color: white; font-family: 'Fredoka One', cursive; font-size: 12px; padding: 8px 15px; border-radius: 0 15px 0 0; box-shadow: 2px -2px 5px rgba(0,0,0,0.1); z-index: 1500; cursor: pointer; transition: transform 0.4s ease, bottom 0.4s ease, left 0.4s ease; letter-spacing: 0.5px; }}
            #sidebar-wrapper.closed ~ .bottom-tab {{ left: 0px; }} 
            
            #bottombar {{ position: absolute; bottom: 0; left: 260px; width: calc(100% - 260px); height: 85px; background: #f5f5f5; z-index: 1000; display: flex; align-items: center; padding: 5px 15px; overflow-x: auto; gap: 12px; white-space: nowrap; transition: transform 0.4s ease, left 0.4s ease, width 0.4s ease; box-shadow: 0 -2px 10px rgba(0,0,0,0.05); border-radius: 0 20px 0 0; }}
            #bottombar::-webkit-scrollbar {{ height: 8px; }}
            #bottombar::-webkit-scrollbar-track {{ background: #e0e0e0; border-radius: 4px; margin: 0 15px; }}
            #bottombar::-webkit-scrollbar-thumb {{ background: #c0c0c0; border-radius: 4px; }}
            #bottombar::-webkit-scrollbar-thumb:hover {{ background: #a0a0a0; }}
            
            #sidebar-wrapper.closed ~ #bottombar {{ left: 0; width: 100%; border-radius: 0; }}
            #bottombar.closed {{ transform: translateY(85px); }}
            .bottom-tab.closed {{ transform: translateY(85px); }}

            .bottom-img {{ height: 60px; width: auto; object-fit: contain; filter: drop-shadow(2px 2px 3px rgba(0,0,0,0.2)); transition: 0.2s; cursor: pointer; flex-shrink: 0; border-radius: 4px; }}
            .bottom-img:hover {{ transform: translateY(-3px) scale(1.05); filter: drop-shadow(2px 4px 6px rgba(0,0,0,0.3)); }}
            
            .selected-card {{ display: flex; align-items: center; background: white; padding: 5px 12px; border-radius: 12px; border: 1px solid #ddd; min-width: 200px; box-shadow: 0 2px 5px rgba(0,0,0,0.05); flex-shrink: 0; transition: 0.2s; cursor: pointer; margin-right: 5px; }}
            .selected-card:hover {{ transform: translateY(-3px); box-shadow: 0 4px 8px rgba(0,0,0,0.1); background: #fdfdfd; }}
            .selected-card img {{ height: 60px; width: auto; object-fit: contain; margin-right: 12px; border-radius: 4px; }}
            .selected-card-info {{ font-family: 'Varela Round', sans-serif; font-size: 11px; color: #555; line-height: 1.4; display: flex; flex-direction: column; }}
            .selected-card-info b {{ color: #e31836; font-weight: bold; font-family: 'Fredoka One', cursive; letter-spacing: 0.5px; margin-right: 3px; }}
            
            .custom-facade-icon {{ background: transparent; border: none; cursor: pointer; }}
            .custom-facade-icon:hover img {{ transform: scale(1.1); filter: drop-shadow(2px 2px 4px rgba(0,0,0,0.8)); }}

            .special-landmark-icon {{ background: transparent; border: none; cursor: pointer; z-index: 1000 !important; }}
            .special-landmark-icon img {{ transform: scale(1.0); filter: drop-shadow(1px 1px 2px rgba(0,0,0,0.8)); transition: 0.2s; }}
            .special-landmark-icon:hover img {{ transform: scale(1.15); filter: drop-shadow(2px 2px 4px rgba(0,0,0,0.9)); }}
            
            #info-box {{ position: absolute; top: 20px; right: 20px; background: rgba(0,0,0,0.7); color: white; padding: 10px 15px; border-radius: 8px; z-index: 1000; font-size: 12px; font-weight: bold; font-family: monospace; pointer-events: none; }}

            #facade-btn-container {{ position: absolute; top: 60px; right: 20px; z-index: 3500; display: flex; align-items: center; background: white; border-radius: 30px; box-shadow: 0 4px 10px rgba(0,0,0,0.15); cursor: pointer; transition: 0.2s; overflow: hidden; border: 1px solid rgba(0,0,0,0.05); }}
            #facade-btn-container:hover {{ transform: scale(1.05); box-shadow: 0 6px 15px rgba(0,0,0,0.25); }}
            .facade-btn-text {{ color: #e31836; font-family: 'Fredoka One', cursive; font-size: 14px; padding: 10px 15px 10px 20px; letter-spacing: 0.5px; }}
            .facade-btn-icon {{ background: #e31836; color: white; width: 40px; height: 40px; border-radius: 50%; display: flex; justify-content: center; align-items: center; font-size: 18px; font-weight: bold; margin-left: -5px; }}

            #collage-overlay {{ position: absolute; top: 0; left: 0; width: 100%; height: 100%; background: rgba(245, 245, 245, 0.98); z-index: 3000; visibility: hidden; opacity: 0; transition: opacity 0.4s ease; overflow: hidden; cursor: grab; }}
            #collage-overlay.active {{ visibility: visible; opacity: 1; }}
            #collage-content {{ position: absolute; top: 0; left: 0; width: 100%; height: 100%; transform-origin: 0 0; will-change: transform; transition: transform 0.1s ease-out; }}
            .flying-facade {{ position: absolute; top: 0; left: 0; transition: transform 1.2s cubic-bezier(0.25, 1, 0.5, 1), opacity 0.3s; object-fit: contain; filter: drop-shadow(2px 2px 4px rgba(0,0,0,0.3)); z-index: 3001; will-change: transform; backface-visibility: hidden; }}

            #intro-screen {{ position: fixed; top: 0; left: 0; width: 100vw; height: 100vh; background: #fcfcfc; z-index: 9999; display: flex; align-items: center; justify-content: center; overflow: hidden; transition: background 1.5s ease; cursor: pointer; }}
            #intro-collage {{ position: absolute; top: -10%; left: -5%; width: 110%; height: 120%; display: flex; align-items: stretch; justify-content: center; gap: 4px; pointer-events: none; }}
            .intro-col {{ flex: 1; display: flex; flex-direction: column; gap: 4px; transition: transform 1.5s cubic-bezier(0.25, 1, 0.5, 1), opacity 1s; }}
            .intro-col.intro-col-large {{ flex: 4; z-index: 5; filter: drop-shadow(5px 5px 15px rgba(0,0,0,0.3)); }}
            .intro-col img {{ width: 100%; height: auto; object-fit: cover; flex-grow: 1; }}
            
            .intro-center-content {{ position: relative; z-index: 10000; text-align: center; pointer-events: none; transition: opacity 0.5s, transform 0.5s; }}
            .intro-title {{ font-family: 'Space Grotesk', sans-serif; font-size: 10vw; color: #ffffff; -webkit-text-stroke: 4px #76a8cd; margin: 0; line-height: 1; letter-spacing: 5px; text-shadow: 10px 10px 20px rgba(0,0,0,0.4); }}
        </style>
    </head>
    <body>

        <div id="intro-screen" onclick="closeIntro()">
            <div id="intro-collage"></div>
            <div class="intro-center-content"><h1 class="intro-title">BALAT</h1></div>
        </div>

        <div id="sidebar-wrapper">
            <button id="toggle-btn" onclick="toggleSidebar()">‹</button>
            <div id="sidebar">
                <div class="sidebar-header">Characteristic<br>of BALAT</div>
                <div class="sidebar-content">
                    <button class="pill-btn active" onclick="resetFilters(this)">ALL BUILDINGS</button>
                    <button class="pill-btn" style="background: transparent; color: #ea1f48;">SELECTIONS</button>
                    
                    <div class="white-box">
                        <div class="mat-container">
                            <div class="mat-row" onclick="filterMaterial('Tuğla', this)">
                                <div class="mat-icon-wrapper"><div class="icon-tugla-bg"></div></div><span class="mat-text">brick (tuğla)</span>
                            </div>
                            <div class="mat-row" onclick="filterMaterial('Ahşap', this)">
                                <div class="mat-icon-wrapper"><div class="icon-ahsap-bg"></div></div><span class="mat-text">wood (ahşap)</span>
                            </div>
                            <div class="mat-row" onclick="filterMaterial('Sıvalı', this)">
                                <div class="mat-icon-wrapper"><div class="icon-sivali-bg"></div></div><span class="mat-text">plaster (sıvalı)</span>
                            </div>
                        </div>
                        <h3 class="box-title title-blue">MATERIAL</h3><p class="desc">Choose the buildings<br>that have the <b>material</b><br>you want to see.</p>
                    </div>
                    
                    <div class="white-box">
                        <div class="color-graphic" id="color-graphic"></div>
                        <h3 class="box-title title-orange">COLOR</h3><p class="desc">Choose the buildings in<br>the <b>color</b> you want to<br>see.</p>
                    </div>
                    
                    <div class="dominant-color-container" onclick="filterColor('{most_dominant_color}')">
                        <div class="dominant-circle"></div><div class="dominant-text">the most<br>dominant color</div>
                    </div>
                </div>
            </div>
        </div>

        <div class="bottom-tab" id="bottom-tab" onclick="toggleBottomBar()">Selected facades</div>
        <div id="bottombar"></div>
        <div id="info-box">Architectural Abstracted Facades</div>
        
        <div id="facade-btn-container" onclick="toggleCollage()">
            <div class="facade-btn-text">click for all facade</div>
            <div class="facade-btn-icon">▶</div>
        </div>
        
        <div id="collage-overlay"><div id="collage-content"></div></div>
        <div id="map"></div>

        <script>
            const minLat = {min_lat} - 0.002;
            const maxLat = {max_lat} + 0.002;
            const minLon = {min_lon} - 0.002;
            const maxLon = {max_lon} + 0.002;
            const bounds = [[minLat, minLon], [maxLat, maxLon]];

            const map = L.map('map', {{ 
                zoomControl: false, minZoom: 16, maxZoom: 19,
                maxBounds: bounds, maxBoundsViscosity: 1.0, wheelPxPerZoomLevel: 120
            }}).setView([41.0315, 28.9480], 16);
            
            L.tileLayer('https://{{s}}.basemaps.cartocdn.com/light_nolabels/{{z}}/{{x}}/{{y}}{{r}}.png', {{ attribution: '', maxZoom: 22, maxNativeZoom: 18 }}).addTo(map);

            map.createPane('yollarPane'); map.getPane('yollarPane').style.zIndex = 700; map.getPane('yollarPane').style.pointerEvents = 'none';
            map.createPane('sinirPane'); map.getPane('sinirPane').style.zIndex = 800; map.getPane('sinirPane').style.pointerEvents = 'none';
            map.createPane('heatmapPane'); map.getPane('heatmapPane').style.zIndex = 450; map.getPane('heatmapPane').style.pointerEvents = 'none'; 

            const geojsonData = {geojson_data};
            const balatSiniri = {balat_geojson};
            const topColors = {palet_json};
            const yollarData = {yollar_geojson};
            const specialLandmarksGercek = {landmarks_gercek_json};
            const specialLandmarksAbstract = {landmarks_abstract_json};
            const manuelPinsData = {manuel_pin_json};

            const worldCoords = [[-90, -180], [90, -180], [90, 180], [-90, 180]];
            const balatCoords = balatSiniri.features[0].geometry.coordinates[0];
            L.polygon([worldCoords, balatCoords], {{ color: 'transparent', fillColor: '#e5e5e5', fillOpacity: 0.8 }}).addTo(map);
            L.geoJSON(balatSiniri, {{ pane: 'sinirPane', style: {{ color: '#ea1f48', weight: 2, fillOpacity: 0, dashArray: '5, 5' }} }}).addTo(map);
            L.geoJSON(yollarData, {{ pane: 'yollarPane', style: {{ color: '#ffffff', weight: 3, opacity: 1.0 }}, pointToLayer: function (feature, latlng) {{ return L.circleMarker(latlng, {{radius: 0, opacity: 0, fillOpacity: 0}}); }} }}).addTo(map);

            const colorGraphic = document.getElementById('color-graphic');
            topColors.forEach((color) => {{
                const btn = document.createElement('button');
                btn.className = 'color-circle-btn'; btn.style.backgroundColor = color;
                btn.setAttribute('data-color', color); btn.onclick = () => filterColor(color);
                colorGraphic.appendChild(btn);
            }});

            let activeMaterial = 'Tümü'; let activeColor = 'Tümü';
            let semanticStep = 0; let isBottomBarOpen = true; let selectedFeatures = [];

            window.removeBuilding = function(id) {{ selectedFeatures = selectedFeatures.filter(f => f.id !== id); renderBottomBar(); }};
            window.selectBuilding = function(props) {{
                if (!selectedFeatures.some(f => f.id === props.id)) {{ selectedFeatures.unshift(props); }}
                renderBottomBar(); if(!isBottomBarOpen) toggleBottomBar();
                setTimeout(() => {{ document.getElementById('bottombar').scrollLeft = 0; }}, 50);
            }};
            window.selectBuildingFromIcon = function(el, id, img, sokak, kat, zemin, malzeme) {{ selectBuilding({{ id: id, img: img, sokak: sokak, kat: kat, zemin: zemin, malzeme: malzeme }}); }};

            function renderBottomBar() {{
                const bottomBar = document.getElementById('bottombar'); bottomBar.innerHTML = ''; 
                selectedFeatures.forEach(props => {{
                    const safeSokak = (props.sokak || "Bilinmiyor").replace(/'/g, ""); const safeKat = (props.kat || "").replace(/'/g, "");
                    const safeZemin = (props.zemin || "").replace(/'/g, ""); const safeMalzeme = (props.malzeme || "").replace(/'/g, "");
                    const cardHtml = `
                        <div class="selected-card" onclick="removeBuilding('${{props.id}}')">
                            <img src="${{props.img}}">
                            <div class="selected-card-info">
                                <span><b>Street:</b> ${{safeSokak}}</span><span><b>Floors:</b> ${{safeKat}}</span>
                                <span><b>Ground:</b> ${{safeZemin}}</span><span><b>Material:</b> ${{safeMalzeme}}</span>
                            </div>
                        </div>`;
                    bottomBar.innerHTML += cardHtml;
                }});
                
                let visibleCount = 0; const viewBounds = map.getBounds();
                geojsonData.features.forEach(feature => {{
                    let props = feature.properties; let isHidden = false;
                    if (activeMaterial !== 'Tümü' && props.malzeme !== activeMaterial) isHidden = true;
                    if (activeColor !== 'Tümü' && props.c25_closest !== activeColor) isHidden = true;
                    const latLng = L.latLng(props.center_lat, props.center_lon);
                    
                    if (!isHidden && viewBounds.contains(latLng) && visibleCount < 100) {{
                        if (!selectedFeatures.some(f => f.id === props.id)) {{
                            const safeSokak = (props.sokak || "Bilinmiyor").replace(/'/g, ""); const safeKat = (props.kat || "").replace(/'/g, "");
                            const safeZemin = (props.zemin || "").replace(/'/g, ""); const safeMalzeme = (props.malzeme || "").replace(/'/g, "");
                            bottomBar.innerHTML += `<img class="bottom-img" src="${{props.img}}" onclick="selectBuildingFromIcon(this, '${{props.id}}', '${{props.img}}', '${{safeSokak}}', '${{safeKat}}', '${{safeZemin}}', '${{safeMalzeme}}')">`;
                            visibleCount++;
                        }}
                    }}
                }});
                if ((activeMaterial !== 'Tümü' || activeColor !== 'Tümü') && !isBottomBarOpen && (visibleCount > 0 || selectedFeatures.length > 0)) {{ toggleBottomBar(); }}
            }}

            let poligonLayer = L.geoJSON(geojsonData, {{ pane: 'heatmapPane', style: function(feature) {{ return {{ fillColor: 'transparent', color: 'transparent' }}; }} }}).addTo(map);
            let iconLayer = L.layerGroup().addTo(map);

            map.getContainer().addEventListener('wheel', function(e) {{
                const atMinZoom = (map.getZoom() <= map.getMinZoom() + 0.05);
                if (semanticStep > 0) {{
                    e.preventDefault(); e.stopPropagation();
                    if (map.scrollWheelZoom.enabled()) map.scrollWheelZoom.disable();
                    if (e.deltaY > 0) {{ if (semanticStep < 5) {{ semanticStep++; updateMap(); }} }} 
                    else if (e.deltaY < 0) {{ semanticStep--; updateMap(); if (semanticStep === 0) map.scrollWheelZoom.enable(); }}
                    return; 
                }}
                if (semanticStep === 0 && atMinZoom) {{
                    if (e.deltaY > 0) {{ e.preventDefault(); e.stopPropagation(); map.scrollWheelZoom.disable(); semanticStep++; updateMap(); return; }}
                }}
                if (semanticStep === 0 && !map.scrollWheelZoom.enabled()) {{ map.scrollWheelZoom.enable(); }}
            }}, {{passive: false}});

            function updateMap() {{
                const z = map.getZoom(); const viewBounds = map.getBounds();
                iconLayer.clearLayers();
                let status = "";
                if (z > 16) {{ status = "Architectural Abstracted Facades"; }} 
                else {{
                    if (semanticStep === 0) status = "Architectural Abstracted Facades";
                    else if (semanticStep === 1) status = "Facade Color";
                    else if (semanticStep === 2) status = "Ten Dominant Color";
                    else if (semanticStep === 3) status = "Five Dominant Color";
                    else if (semanticStep === 4) status = "Three Dominant Color";
                    else if (semanticStep === 5) status = "The Dominant Color of Balat";
                }}
                document.getElementById('info-box').innerText = status;

                poligonLayer.setStyle(function(feature) {{
                    let props = feature.properties; let isHidden = false;
                    if (activeMaterial !== 'Tümü' && props.malzeme !== activeMaterial) isHidden = true;
                    if (activeColor !== 'Tümü' && props.c25_closest !== activeColor) isHidden = true;
                    let fillColor = props.renk_orj; let strokeColor = '#ffffff'; let fillOp = isHidden ? 0.05 : 0.95; let strokeOp = 0; let weight = 0;
                    
                    if (z > 16 || semanticStep === 0) {{ return {{ fillColor: fillColor, fillOpacity: 0.0, color: 'transparent', weight: 0 }}; }}
                    
                    if (semanticStep >= 1) {{
                        if (semanticStep === 2) fillColor = props.c10;
                        if (semanticStep === 3) fillColor = props.c5;
                        if (semanticStep === 4) fillColor = props.c3;
                        if (semanticStep === 5) fillColor = props.c1;
                        strokeColor = fillColor; weight = semanticStep * 15; fillOp = isHidden ? 0.0 : 0.85; strokeOp = isHidden ? 0.0 : 0.4; 
                    }}
                    return {{ fillColor: fillColor, fillOpacity: fillOp, color: strokeColor, weight: weight, opacity: strokeOp }};
                }});

                geojsonData.features.forEach(feature => {{
                    let props = feature.properties; let isHidden = false;
                    if (activeMaterial !== 'Tümü' && props.malzeme !== activeMaterial) isHidden = true;
                    if (activeColor !== 'Tümü' && props.c25_closest !== activeColor) isHidden = true;
                    if ((z > 16 || semanticStep === 0) && !isHidden) {{
                        const iconHtml = `<div style="width:100%; height:100%; display:flex; justify-content:center; align-items:center; transition:0.2s;"><img src="${{props.img}}" style="max-height:100%; max-width:100%; object-fit:contain; filter: drop-shadow(1px 1px 1px rgba(0,0,0,0.5)); transition:0.2s;"></div>`;
                        const icon = L.divIcon({{ html: iconHtml, className: 'custom-facade-icon', iconSize: [12, 24], iconAnchor: [6, 12] }});
                        L.marker([props.center_lat, props.center_lon], {{ icon: icon }}).addTo(iconLayer);
                    }}
                }});
                
                if (z > 16 || semanticStep === 0) {{
                    manuelPinsData.forEach(pin => {{
                        if (pin.b64 && pin.b64 !== "") {{
                            const iconHtml = `<div style="width:100%; height:100%; display:flex; justify-content:center; align-items:center; transition:0.2s;"><img src="${{pin.b64}}" style="max-height:100%; max-width:100%; object-fit:contain; filter: drop-shadow(1px 1px 2px rgba(0,0,0,0.8)); transition:0.2s;"></div>`;
                            const icon = L.divIcon({{ html: iconHtml, className: 'special-landmark-icon', iconSize: [12, 24], iconAnchor: [6, 12] }});
                            let marker = L.marker([pin.lat, pin.lon], {{ icon: icon }}).addTo(iconLayer);
                            marker.on('click', function() {{ selectBuilding({{ id: "manuel_" + pin.isim, img: pin.b64, sokak: pin.isim, kat: "Sembolik", zemin: "Sembolik", malzeme: "Özel" }}); }});
                        }}
                    }});
                }}
                renderBottomBar();
            }}

            map.on('zoomend', updateMap); map.on('moveend', updateMap);

            function toggleSidebar() {{
                const wrapper = document.getElementById('sidebar-wrapper'); const btn = document.getElementById('toggle-btn');
                if(wrapper.classList.contains('closed')) {{ wrapper.classList.remove('closed'); btn.innerHTML = '‹'; }} 
                else {{ wrapper.classList.add('closed'); btn.innerHTML = '›'; }}
            }}
            function toggleBottomBar() {{
                const bar = document.getElementById('bottombar'); const tab = document.getElementById('bottom-tab');
                if (isBottomBarOpen) {{ bar.classList.add('closed'); tab.classList.add('closed'); isBottomBarOpen = false; }} 
                else {{ bar.classList.remove('closed'); tab.classList.remove('closed'); isBottomBarOpen = true; }}
            }}
            function resetFilters(btnElement) {{
                activeMaterial = 'Tümü'; activeColor = 'Tümü';
                document.querySelectorAll('.pill-btn').forEach(el => el.classList.remove('active')); btnElement.classList.add('active');
                document.querySelectorAll('.mat-row').forEach(el => el.classList.remove('inactive'));
                document.querySelectorAll('.color-circle-btn').forEach(el => {{ el.style.opacity = '1'; el.style.transform = 'translate(-15%, -15%) scale(1)'; el.style.zIndex = '10'; }});
                updateMap();
            }}
            function filterMaterial(malzeme, element) {{
                activeMaterial = malzeme; activeColor = 'Tümü'; 
                document.querySelectorAll('.pill-btn').forEach(el => el.classList.remove('active'));
                document.querySelectorAll('.mat-row').forEach(el => {{ if(el === element) el.classList.remove('inactive'); else el.classList.add('inactive'); }});
                document.querySelectorAll('.color-circle-btn').forEach(el => {{ el.style.opacity = '1'; el.style.transform = 'translate(-15%, -15%) scale(1)'; el.style.zIndex = '10'; }});
                updateMap();
            }}
            function filterColor(renk) {{
                activeColor = renk; activeMaterial = 'Tümü'; 
                document.querySelectorAll('.pill-btn').forEach(el => el.classList.remove('active'));
                document.querySelectorAll('.mat-row').forEach(el => el.classList.remove('inactive')); 
                document.querySelectorAll('.color-circle-btn').forEach(el => {{
                    if (renk === 'Tümü' || el.getAttribute('data-color') === renk) {{ el.style.opacity = '1'; el.style.transform = 'translate(-15%, -15%) scale(1.1)'; el.style.zIndex = '15'; }} 
                    else {{ el.style.opacity = '0.15'; el.style.transform = 'translate(-15%, -15%) scale(1)'; el.style.zIndex = '10'; }}
                }});
                updateMap();
            }}

            let isCollageMode = false; let flyingElements = []; let collageZoom = 1; let collagePanX = 0; let collagePanY = 0; let isDraggingCollage = false; let startDragX, startDragY;
            const overlay = document.getElementById('collage-overlay'); const cContent = document.getElementById('collage-content');

            overlay.addEventListener('wheel', (e) => {{
                if (!isCollageMode) return; e.preventDefault();
                const zoomPointX = e.clientX; const zoomPointY = e.clientY;
                let zoomDelta = e.deltaY < 0 ? 1.15 : 0.85; let newZoom = collageZoom * zoomDelta;
                if (newZoom < 1.0) newZoom = 1.0; if (newZoom > 15) newZoom = 15;   
                collagePanX = zoomPointX - (zoomPointX - collagePanX) * (newZoom / collageZoom); collagePanY = zoomPointY - (zoomPointY - collagePanY) * (newZoom / collageZoom);
                collageZoom = newZoom; cContent.style.transition = 'none'; cContent.style.transform = `translate3d(${{collagePanX}}px, ${{collagePanY}}px, 0) scale(${{collageZoom}})`;
            }});
            overlay.addEventListener('mousedown', (e) => {{ if (!isCollageMode) return; isDraggingCollage = true; startDragX = e.clientX - collagePanX; startDragY = e.clientY - collagePanY; overlay.style.cursor = 'grabbing'; }});
            window.addEventListener('mousemove', (e) => {{ if (!isDraggingCollage) return; collagePanX = e.clientX - startDragX; collagePanY = e.clientY - startDragY; cContent.style.transition = 'none'; cContent.style.transform = `translate3d(${{collagePanX}}px, ${{collagePanY}}px, 0) scale(${{collageZoom}})`; }});
            window.addEventListener('mouseup', () => {{ isDraggingCollage = false; overlay.style.cursor = 'grab'; }});

            function hexToHSL(H) {{
                let r = 0, g = 0, b = 0;
                if (H.length == 4) {{ r = "0x" + H[1] + H[1]; g = "0x" + H[2] + H[2]; b = "0x" + H[3] + H[3]; }}
                else if (H.length == 7) {{ r = "0x" + H[1] + H[2]; g = "0x" + H[3] + H[4]; b = "0x" + H[5] + H[6]; }}
                r /= 255; g /= 255; b /= 255;
                let cmin = Math.min(r,g,b), cmax = Math.max(r,g,b), delta = cmax - cmin, h = 0, s = 0, l = 0;
                if (delta == 0) h = 0;
                else if (cmax == r) h = ((g - b) / delta) % 6;
                else if (cmax == g) h = (b - r) / delta + 2;
                else h = (r - g) / delta + 4;
                h = Math.round(h * 60); if (h < 0) h += 360; l = (cmax + cmin) / 2; return {{h, l}};
            }}

            function toggleCollage() {{
                const btnText = document.querySelector('.facade-btn-text'); const btnIcon = document.querySelector('.facade-btn-icon');
                isCollageMode = !isCollageMode;
                if (isCollageMode) {{
                    overlay.classList.add('active'); btnText.innerText = "back to map"; btnIcon.innerText = "×";
                    document.querySelectorAll('.custom-facade-icon').forEach(el => el.style.opacity = '0');
                    let activeFeatures = [...geojsonData.features];
                    activeFeatures.sort((a, b) => {{ let hslA = hexToHSL(a.properties.renk_orj); let hslB = hexToHSL(b.properties.renk_orj); if (Math.abs(hslA.h - hslB.h) > 15) return hslA.h - hslB.h; return hslB.l - hslA.l; }});
                    const ww = window.innerWidth; const wh = window.innerHeight; let imgW = 24; let imgH = 48; let cols = Math.floor(ww / imgW); let startX = (ww - (cols * imgW)) / 2;    
                    let totalBuildings = activeFeatures.length; let baseHeight = Math.floor(totalBuildings / cols); let remainder = totalBuildings % cols;
                    let colCapacities = new Array(cols).fill(baseHeight);
                    for(let k=0; k<remainder; k++) {{ let mid = Math.floor(cols/2); let offset = k % 2 === 0 ? k/2 : -Math.ceil(k/2); colCapacities[(mid + offset + cols) % cols]++; }}
                    for (let k = 0; k < Math.floor(totalBuildings * 0.1); k++) {{ let from = Math.floor(Math.random() * cols); let to = Math.floor(Math.random() * cols); if (colCapacities[from] > 1) {{ colCapacities[from]--; colCapacities[to]++; }} }}
                    collageZoom = 1; collagePanX = 0; collagePanY = 0; cContent.style.transition = 'transform 1.2s cubic-bezier(0.25, 1, 0.5, 1)'; cContent.style.transform = `translate3d(${{collagePanX}}px, ${{collagePanY}}px, 0) scale(${{collageZoom}})`;
                    let currentFeatureIdx = 0;
                    for (let c = 0; c < cols; c++) {{
                        let capacity = colCapacities[c];
                        for (let r = 0; r < capacity; r++) {{
                            if (currentFeatureIdx >= totalBuildings) break;
                            let f = activeFeatures[currentFeatureIdx]; let pt = map.latLngToContainerPoint([f.properties.center_lat, f.properties.center_lon]);
                            let img = document.createElement('img'); img.src = f.properties.img; img.className = 'flying-facade';
                            let startTransformX = pt.x - collagePanX; let startTransformY = pt.y - collagePanY; let startScale = 12 / imgW; 
                            img.style.transform = `translate3d(${{startTransformX}}px, ${{startTransformY}}px, 0) scale(${{startScale}})`; img.style.width = imgW + 'px'; img.style.height = imgH + 'px';
                            cContent.appendChild(img);
                            let targetX = startX + (c * imgW); let targetY = wh - 50 - (r * imgH) - imgH;
                            flyingElements.push({{ el: img, lat: f.properties.center_lat, lon: f.properties.center_lon, tS: startScale, trgX: targetX, trgY: targetY }});
                            currentFeatureIdx++;
                        }}
                    }}
                    flyingElements.forEach((item, i) => {{ setTimeout(() => {{ item.el.style.transform = `translate3d(${{item.trgX}}px, ${{item.trgY}}px, 0) scale(1)`; }}, 50 + (i * 2)); }});
                }} else {{
                    btnText.innerText = "click for all facade"; btnIcon.innerText = "▶";
                    collageZoom = 1; collagePanX = 0; collagePanY = 0; cContent.style.transition = 'transform 1.2s cubic-bezier(0.25, 1, 0.5, 1)'; cContent.style.transform = `translate3d(0px, 0px, 0) scale(1)`;
                    overlay.style.transition = 'background-color 0.8s ease'; overlay.style.backgroundColor = 'transparent';
                    flyingElements.forEach((item, i) => {{ let pt = map.latLngToContainerPoint([item.lat, item.lon]); setTimeout(() => {{ item.el.style.transform = `translate3d(${{pt.x}}px, ${{pt.y}}px, 0) scale(${{item.tS}})`; item.el.style.opacity = '0'; }}, i * 2); }});
                    setTimeout(() => {{ document.querySelectorAll('.custom-facade-icon').forEach(el => el.style.opacity = '1'); flyingElements.forEach(item => item.el.remove()); flyingElements = []; overlay.classList.remove('active'); setTimeout(() => {{ overlay.style.backgroundColor = ''; overlay.style.transition = 'opacity 0.4s ease'; }}, 100); }}, 1300 + (flyingElements.length * 2));
                }}
            }}

            function initIntro() {{
                let features = [...geojsonData.features]; let allImages = [];
                features.forEach(f => {{ allImages.push(f.properties.img); let s = (f.properties.sokak || "").toLowerCase(); if (s.includes('kiremit') || s.includes('merdiven')) {{ allImages.push(f.properties.img); allImages.push(f.properties.img); }} }});
                for(let i=0; i<40; i++) {{ if(specialLandmarksGercek && specialLandmarksGercek.length > 0) {{ allImages = allImages.concat(specialLandmarksGercek); }} }}
                allImages.sort(() => Math.random() - 0.5);
                const collage = document.getElementById('intro-collage'); let numCols = 30; 
                let abstractLandmarks = [];
                if (specialLandmarksAbstract && specialLandmarksAbstract.length > 0) {{ for(let i=0; i<10; i++) abstractLandmarks = abstractLandmarks.concat(specialLandmarksAbstract); abstractLandmarks.sort(() => Math.random() - 0.5); }}
                for(let i=0; i<numCols; i++) {{
                    let col = document.createElement('div'); col.className = 'intro-col';
                    if ((i === 4 || i === 10 || i === 16 || i === 22 || i === 28) && abstractLandmarks.length > 0) {{
                        col.classList.add('intro-col-large'); let imgCount = Math.floor(Math.random() * 2) + 3; 
                        for(let j=0; j<imgCount; j++) {{ let img = document.createElement('img'); let absImg = abstractLandmarks.shift() || abstractLandmarks[0]; img.src = absImg; col.appendChild(img); }}
                    }} else {{
                        let imgCount = Math.floor(Math.random() * 5) + 6;
                        for(let j=0; j<imgCount; j++) {{ if(allImages.length > 0) {{ let img = document.createElement('img'); img.src = allImages[Math.floor(Math.random() * allImages.length)]; col.appendChild(img); }} }}
                    }}
                    let offset = (Math.random() * 40 - 20); col.style.transform = `translateY(${{offset}}%)`; collage.appendChild(col);
                }}
            }}

            function closeIntro() {{
                const intro = document.getElementById('intro-screen'); const cols = document.querySelectorAll('.intro-col'); const centerContent = document.querySelector('.intro-center-content');
                centerContent.style.opacity = '0'; centerContent.style.transform = 'scale(0.8)';
                cols.forEach((col, i) => {{ let dir = i % 2 === 0 ? -100 : 100; col.style.transform = `translateY(${{dir}}vh)`; col.style.opacity = '0'; }});
                intro.style.background = 'transparent';
                setTimeout(() => {{ intro.style.visibility = 'hidden'; updateMap(); }}, 1500);
            }}

            setTimeout(() => {{ initIntro(); updateMap(); }}, 100);
        </script>
    </body>
    </html>
    """
    return html_template

# Uygulamayı ekranda göster
st.markdown("<h2 style='text-align: center; color: #ea1f48; font-family: sans-serif;'>Balat Morfoloji Dashboard Yükleniyor...</h2>", unsafe_allow_html=True)

with st.spinner('Kentsel veriler hesaplanıyor, lütfen bekleyin...'):
    harita_html = haritayi_olustur()

# HTML'i Streamlit içine göm ve çalıştır
components.html(harita_html, height=1000, scrolling=True)
"""

with open("app.py", "w", encoding="utf-8") as f:
    f.write(code_content)

print("Streamlit app.py generated successfully.")}}
