# Import required packages:
import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
import argparse

repertoire_source = "/home/baptiste/vision_arti/TP1/Images/"
repertoire_cible = "/home/baptiste/vision_arti/TP1/Images_gris/"
dossiers_source = ["Chambre", "Cuisine", "Salon"]

for dossier in dossiers_source:

    chemin_dossier_source = os.path.join(repertoire_source, dossier)
    chemin_dossier_cible = os.path.join(repertoire_cible, dossier)
    os.makedirs(chemin_dossier_cible, exist_ok=True)

    for fichier in os.listdir(chemin_dossier_source):
        if fichier.endswith(".JPG") or fichier.endswith(".jpg"):

            image = cv2.imread(os.path.join(chemin_dossier_source, fichier), 0)
            image = cv2.resize(image, (0, 0), fx=0.1, fy=0.1)

            gamma = 0.8
            image_corrigee = image ** gamma
            image_filtree = cv2.GaussianBlur(image_corrigee, (5, 5), 0)
            image_filtree = image_filtree.astype(np.uint8)
            
            seuil = 100  
            _, image_seuillee = cv2.threshold(image_filtree, seuil, 255, cv2.THRESH_BINARY)
            _, image_otsu = cv2.threshold(image_filtree, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

            # Définir le nom du fichier de sortie
            nom_fichier_sortie = os.path.splitext(fichier)[0]

            # Enregistrer les images seuillées dans le dossier cible
            chemin_fichier_sortie_seuil = os.path.join(chemin_dossier_cible, nom_fichier_sortie + "_seuil.jpg")
            chemin_fichier_sortie_otsu = os.path.join(chemin_dossier_cible, nom_fichier_sortie + "_otsu.jpg")

            cv2.imwrite(chemin_fichier_sortie_seuil, image_seuillee)
            cv2.imwrite(chemin_fichier_sortie_otsu, image_otsu)

            # Charger l'image de référence en niveaux de gris (CV_8UC1)
            image_reference = cv2.imread(os.path.join(repertoire_source, dossier, "Reference.JPG"), 0)

            # Convertir l'image en niveaux de gris en une image en noir et blanc (RVB)
            image_reference_rgb = cv2.cvtColor(image_reference, cv2.COLOR_GRAY2BGR)

            # Afficher l'image de référence
            cv2.imshow("Image de Référence", image_reference_rgb)

            # Charger l'image avec les changements détectés
            image_changements = cv2.imread(chemin_fichier_sortie_otsu,0)

            # Afficher l'image de référence et l'image avec les changements détectés
            image_reference = cv2.resize(image_reference, (0, 0), fx=0.1, fy=0.1)
            cv2.imshow("Image de Référence", image_reference)
            cv2.imshow("Image de base", image)
            cv2.imshow("Image avec Changements Détectés", image_changements)

            # Définir les coordonnées de positionnement des fenêtres
            x_offset = 10  # Décalage horizontal
            y_offset = 10  # Décalage vertical

            # Déplacer les fenêtres à leurs positions respectives
            cv2.moveWindow("Image de Référence", x_offset, y_offset)
            cv2.moveWindow("Image de base", x_offset * 2 + image_reference.shape[1], y_offset)
            cv2.moveWindow("Image avec Changements Détectés", x_offset * 3 + 2 * image_reference.shape[1], y_offset)

            cv2.waitKey(0)  # Attendre une touche pour fermer la fenêtre


            cv2.waitKey(0)  # Attendre une touche pour fermer la fenêtre

# Fermer toutes les fenêtres OpenCV (si vous en avez ouvert)
cv2.destroyAllWindows()
