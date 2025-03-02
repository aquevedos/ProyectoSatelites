import os
from rasterio.windows import Window
import shutil

# Separar las masks i les imatges que tenen tot zeros
# No puc entrenar amb màscares en què la banda són tot zeros
# Els fitxers de masks amb tot zeros els tinc identificats amb la paraula "blanks" al nom
# Per tant, només haig de moure les imatges corresponents a un directori que anomero tot_ceros

def move_blank_masks_and_corresponding_images(mask_dir, tile_dir_august, tile_dir_april):
    tot_ceros_dir = r'\tot_ceros'
    mask_dir_blanks = mask_dir + tot_ceros_dir
    tile_dir_august_cero_folder = tile_dir_august + tot_ceros_dir
    tile_dir_april_cero_folder = tile_dir_april + tot_ceros_dir


    if not os.path.exists(mask_dir_blanks):
        os.makedirs(mask_dir_blanks)

    if not os.path.exists(tile_dir_august_cero_folder):
        os.makedirs(tile_dir_august_cero_folder)

    if not os.path.exists(tile_dir_april_cero_folder):
        os.makedirs(tile_dir_april_cero_folder)

    for mask_file in os.listdir(mask_dir):
        mask_path = os.path.join(mask_dir, mask_file)
        if mask_file.endswith('blanks.tif'):

            # Moure la imatge corresponent i la mask
            tile_file = mask_file.replace("mask", "tile_norm")
            tile_file = tile_file.replace("_blanks.tif", ".png")

            # Si les imatges que hem de moure, existeixen
            if os.path.exists(os.path.join(tile_dir_august, tile_file)) and \
               os.path.exists(os.path.join(tile_dir_april, tile_file)):
                
                shutil.move(os.path.join(tile_dir_august, tile_file), os.path.join(tile_dir_august_cero_folder, tile_file))
                shutil.move(os.path.join(tile_dir_april, tile_file), os.path.join(tile_dir_april_cero_folder, tile_file))
    
                shutil.move(mask_path, os.path.join(mask_dir_blanks, mask_file))

                print(f"He mogut {mask_file} {tile_file} per august i april als directoris ceros")
            else:
                print(f"No trobo el tile {tile_file} - això és un error greu")

def copy_image_files_for_training(src_dir, dst_dir, prefix, eliminar_dir_existent=True):
    dst_dir = r'D:\aidl_projecte\sentinel2aprilandaugustfortraining'

    #Eliminar el contingut del directori d'entrenament
    if eliminar_dir_existent:
        if os.path.exists(dst_dir):
            shutil.rmtree(dst_dir)
            os.makedirs(dst_dir)
    
    for file_name in os.listdir(src_dir):
        full_file_name = os.path.join(src_dir, file_name)
        if os.path.isfile(full_file_name) and file_name.endswith('.png'):
            new_file_name = prefix + "_" + file_name
            new_full_file_name = os.path.join(dst_dir, new_file_name)
            shutil.copy(full_file_name, new_full_file_name)
    
    file_count = len(os.listdir(dst_dir))
    print(f"Nombre de fitxers: {file_count} a: {dst_dir} (haurien de ser 386)")


def copy_masks_for_training():
    src_dir = r'D:\aidl_projecte\tiles_masks\tiffsv2'
    dst_dir = r'D:\aidl_projecte\masksfortraining'

    if not os.path.exists(dst_dir):
        os.makedirs(dst_dir)

    for file_name in os.listdir(src_dir):
        full_file_name = os.path.join(src_dir, file_name)
        if os.path.isfile(full_file_name) and file_name.endswith('.tif'):
            april_file_name = "april_" + file_name
            august_file_name = "august_" + file_name
            shutil.copy(full_file_name, os.path.join(dst_dir, april_file_name))
            shutil.copy(full_file_name, os.path.join(dst_dir, august_file_name))

    file_count = len(os.listdir(dst_dir))
    print(f"Nombre de fitxers duplicats: {file_count} a: {dst_dir}")

if __name__ == '__main__':

    # ATENCIO: Si ho volem fer per a August i April cal fer-ho 2 vegades i retornar les masks al directori original
    mask_dir = r'D:\aidl_projecte\tiles_masks\tiffsv2'
    tile_dir_august_norm= r'D:\aidl_projecte\sentinel2august\tile_norm'
    tile_dir_april_norm = r'D:\aidl_projecte\sentinel2april\tile_norm'

    # move_blank_masks_and_corresponding_images(mask_dir=mask_dir, tile_dir_august = tile_dir_august_norm, tile_dir_april =tile_dir_april_norm)

    # Copiar els fitxers que no són tot zeros a un directori per a l'entrenament de la Unet
    #copy_image_files_for_training(tile_dir_august_norm, 'august',  eliminar_dir_existent=True)
    #copy_image_files_for_training(tile_dir_april_norm, 'april', eliminar_dir_existent=False)
    copy_masks_for_training()


