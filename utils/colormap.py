from matplotlib.colors import ListedColormap

# Definir 12 colores distintos para cada etiqueta
colors = [
    "#000000",  # 0 - Out of border (Negro)
    "#7FFF00",  # 1 - Herbaceous crops (Verde claro)
 
    "#006400",  # 2 - Woody crops (verde oscuro)
   
    "#ADFF2F",  # 3 - Forest (Verde lima)
    "#A52A2A",  # 4 - Scrub,  meadows and grasslands (Marrón tierra)
   
    "#8B4513",  # 5 - Bare  soil (Marrón cuero)
  
    "#D3D3D3",  # 6 - urban area (Gris claro)

    "#C71585",  # 7 - Isolated urban areas  (Violeta oscuro)
  
    "#00FA9A",  # 8 - Green areas (verde  menta)

    "#FF4500",  # 9 - Industrial or commercial and leisure areas (Naranja fuerte)
   "#FFD700",  # 10 - mining or landfills (Amarillo)

     "#808080",  # 11 - Transport network (Gris medio)

    "#000080"  # 12 - watercourses and water bodies (Azul marino) <- Esta puedes excluirla si no quieres que el modelo la use

]

# Crear el colormap
land_cover_cmap = ListedColormap(colors)