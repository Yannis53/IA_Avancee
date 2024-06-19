import os
import random
import shutil

def collect_images(src_dir):
    """
    Parcourt récursivement le dossier source et collecte tous les chemins d'images .jpg.
    """
    img_paths = []
    for root, _, files in os.walk(src_dir):
        for file in files:
            if file.lower().endswith('.jpg'):
                img_paths.append(os.path.join(root, file))
    return img_paths

def copy_random_images(src_dir, dest_dir, num_images=500):
    """
    Copie aléatoirement un nombre donné d'images .jpg d'un dossier source vers un dossier de destination,
    en les renommant avec un chiffre incrémental.
    """
    # Vérifier que le répertoire source existe
    if not os.path.exists(src_dir):
        raise ValueError(f"Le répertoire source '{src_dir}' n'existe pas.")
    
    # Récupérer toutes les images .jpg dans le dossier source
    img_paths = collect_images(src_dir)
    
    # S'assurer que nous avons assez d'images
    if len(img_paths) < num_images:
        raise ValueError(f"Pas assez d'images dans le dossier source pour en copier {num_images}. Seulement {len(img_paths)} images trouvées.")

    # Sélectionner aléatoirement les images
    random_images = random.sample(img_paths, num_images)
    
    # Créer le répertoire de destination s'il n'existe pas
    os.makedirs(dest_dir, exist_ok=True)
    
    for idx, img_path in enumerate(random_images, start=1):
        # Définir le nouveau nom de fichier avec un chiffre incrémental
        new_filename = f"{idx}.jpg"
        new_filepath = os.path.join(dest_dir, new_filename)
        
        try:
            # Copier le fichier avec le nouveau nom
            shutil.copy(img_path, new_filepath)
        except IOError as e:
            print(f"Erreur lors de la copie de {img_path} vers {new_filepath}: {e}")
    
    print(f"{num_images} images ont été copiées vers {dest_dir} avec des noms incrémentaux.")

# Exemple d'utilisation
if __name__ == "__main__":
    src_directory = './Image'  # Remplacer par le chemin du répertoire source
    dest_directory = './500'  # Remplacer par le chemin du répertoire destination
    
    try:
        copy_random_images(src_directory, dest_directory, 500)
    except ValueError as e:
        print(e)