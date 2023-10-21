# import cv2
# import os
# import numpy as np
# import matplotlib.pyplot as plt
# import argparse

# repertoire_source = "/home/baptiste/vision_arti/TP1/Images/"
# repertoire_cible = "/home/baptiste/vision_arti/TP1/Images_gris/"
# dossiers_source = ["Chambre", "Cuisine", "Salon"]

# for dossier in dossiers_source:
#     chemin_dossier_source = os.path.join(repertoire_source, dossier)
#     chemin_dossier_cible = os.path.join(repertoire_cible, dossier)
#     os.makedirs(chemin_dossier_cible, exist_ok=True)

#     for fichier in os.listdir(chemin_dossier_source):
#         if fichier.endswith(".JPG") or fichier.endswith(".jpg"):
#             image = cv2.imread(os.path.join(chemin_dossier_source, fichier), 0)
#             image = cv2.resize(image, (0, 0), fx=0.1, fy=0.1)

#             gamma = 0.8
#             image_corrigee = image ** gamma
#             image_filtree = cv2.GaussianBlur(image_corrigee, (5, 5), 0)
#             image_filtree = image_filtree.astype(np.uint8)

#             seuil = 50
#             _, image_seuillee = cv2.threshold(image_filtree, seuil, 255, cv2.THRESH_BINARY)
#             _, image_otsu = cv2.threshold(image_filtree, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

#             nom_fichier_sortie = os.path.splitext(fichier)[0]
#             chemin_fichier_sortie_seuil = os.path.join(chemin_dossier_cible, nom_fichier_sortie + "_seuil.jpg")
#             chemin_fichier_sortie_otsu = os.path.join(chemin_dossier_cible, nom_fichier_sortie + "_otsu.jpg")
#             cv2.imwrite(chemin_fichier_sortie_otsu, image_otsu)

#             image_reference = cv2.imread(os.path.join(repertoire_source, dossier, "Reference.JPG"), 0)
#             image_reference_rgb = cv2.cvtColor(image_reference, cv2.COLOR_GRAY2BGR)
#             image_reference = cv2.resize(image_reference, (0, 0), fx=0.1, fy=0.1)

#             image_changements = cv2.imread(chemin_fichier_sortie_otsu, 0)
#             image_changements = cv2.resize(image_changements, (image_reference.shape[1], image_reference.shape[0]))

#             difference = cv2.absdiff(image_reference, image_changements)
#             contours, _ = cv2.findContours(image_otsu, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

#             image_resultat = image_reference_rgb.copy()
#             image_resultat = cv2.resize(image_resultat, (0, 0), fx=0.1, fy=0.1)

#             for contour in contours:
#                 (x, y, w, h) = cv2.boundingRect(contour)
#                 cv2.rectangle(image_resultat, (x, y), (x + w, y + h), (0, 0, 255), 2)

#             x_offset = 10
#             y_offset = 10
#             cv2.imshow("Image de Référence", image_reference_rgb)
#             cv2.imshow("Image de base", image)
#             cv2.imshow("Image avec Changements Détectés", image_changements)
#             cv2.imshow("Résultat", image_resultat)

#             cv2.moveWindow("Image de Référence", x_offset, y_offset)
#             cv2.moveWindow("Image de base", x_offset * 2 + image_reference.shape[1], y_offset)
#             cv2.moveWindow("Image avec Changements Détectés", x_offset * 3 + 2 * image_reference.shape[1], y_offset)
#             cv2.moveWindow("Résultat", x_offset * 3 + 2 * image_reference.shape[1], y_offset * 2 + image_reference.shape[0])

#             cv2.waitKey(0)

# cv2.destroyAllWindows()

import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
import argparse

repertoire_source = "/home/baptiste/vision_arti/TP1/Images/"
repertoire_cible = "/home/baptiste/vision_arti/TP1/Images_resultantes/"
dossiers_source = ["Chambre", "Cuisine", "Salon"]

for dossier in dossiers_source:
    chemin_dossier_source = os.path.join(repertoire_source, dossier)
    chemin_dossier_cible = os.path.join(repertoire_cible, dossier)
    os.makedirs(chemin_dossier_cible, exist_ok=True)

    image_reference = cv2.imread(os.path.join(repertoire_source, dossier, "Reference.JPG"), 0)
    image_reference = cv2.resize(image_reference, (0, 0), fx=0.1, fy=0.1)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    image_reference = clahe.apply(image_reference)

    for fichier in os.listdir(chemin_dossier_source):
        if fichier.endswith(".JPG") or fichier.endswith(".jpg"):
            image = cv2.imread(os.path.join(chemin_dossier_source, fichier), 0)
            image = cv2.resize(image, (0, 0), fx=0.1, fy=0.1)
            image = cv2.GaussianBlur(image, (5, 5), 0)
            image = clahe.apply(image)

            hist_reference = cv2.calcHist([image_reference], [0], None, [256], [0, 256])
            hist_image = cv2.calcHist([image], [0], None, [256], [0, 256])
            correlation = cv2.compareHist(hist_reference, hist_image, cv2.HISTCMP_CORREL)

            seuil_correlation = 0.9
            if correlation < seuil_correlation:
                print(f"Changement détecté dans {fichier} (Corrélation : {correlation})")
                
                # Pour encadrer les objets détectés avec des rectangles rouges
                _, image_seuillee = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                contours, _ = cv2.findContours(image_seuillee, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                image_resultat = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)  # Convertir en image couleur

                for contour in contours:
                    (x, y, w, h) = cv2.boundingRect(contour)
                    cv2.rectangle(image_resultat, (x, y), (x + w, y + h), (0, 0, 255), 2)

                cv2.imshow("Image avec Changements Détectés", image_resultat)
            else:
                cv2.imshow("Image de base", image)
                cv2.imshow("Image de Référence", image_reference)

            x_offset = 10
            y_offset = 10
            cv2.moveWindow("Image avec Changements Détectés", x_offset, y_offset)
            cv2.waitKey(0)

cv2.destroyAllWindows()
    