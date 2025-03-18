from matplotlib.colors import ListedColormap

'''
This file defines a custom colormap using matplotlib, with colors representing different classes for semantic segmentation tasks
Each color corresponds to a specific class index for visualizing segmented images.
'''

colors = [
    "#000000",  # (Black)
    "#FF0000",  # (Red)
    "#00FF00",  # (Bright Green)
    "#0000FF",  # (Strong Blue)
    "#FFFF00",  # (Yellow)
    "#FF00FF",  # (Magenta)
    "#00FFFF",  # (Cyan)
    "#FFA500",  # (Orange)
    "#800080",  # (Purple)
    "#008000",  # (Dark Green)
    "#A52A2A",  # (Brown)
    "#808080",  # (Gray)
]

land_cover_cmap = ListedColormap(colors)