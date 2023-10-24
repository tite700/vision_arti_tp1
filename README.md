# Rapport sur le Programme de Détection d'Objets dans des Images

Ce rapport explique le fonctionnement du programme de détection d'objets dans des images, y compris les différentes étapes du programme et les points difficiles du projet. Le programme vise à détecter les objets qui ont changé entre des images de référence et des images de test dans trois pièces distinctes : Chambre, Cuisine et Salon. 

## Fonctionnement du Programme

1. **Chargement des Données** :
   - Le programme commence par charger les images à partir des dossiers correspondants (Chambre, Cuisine, Salon).
   - Chaque dossier contient une image référence stockée pour la soustraction.

2. **Prétraitement des Images** :
   - Le programme effectue un prétraitement sur les images pour améliorer la qualité de l'image :
       - Il retire les ombres des images en utilisant la fonction `remove_shadow_from_img`.
       - Les images sont converties en niveaux de gris.
       - Un flou gaussien est appliqué pour réduire le bruit.
       - Un seuillage adaptatif est utilisé pour détecter les différences entre l'image de référence et l'image de test.

3. **Détection des Différences** :
   - Le programme détecte les différences entre l'image de référence et l'image de test en utilisant un seuillage d'otsu.
   - Des rectangles rouges sont dessinés autour des objets dont les différences sont supérieures à un seuil.

4. **Affichage des Images** :
   - Le programme affiche les images d'origine, l'image de référence, l'image de différence et l'image résultante avec les rectangles rouges.
  
5. **Utilisation de masques** :
   - Nous avons mis en place des masques pour chaque pièce pour ne s'intéresser qu'aux zones utiles de celles-ci.

## Points Difficiles

- **Détection des Objets** : L'une des parties les plus complexes du projet était de déterminer quels objets étaient significatifs à détecter. Les objets très petits ou les artefacts dus au bruit pouvaient donner lieu à des détections inutiles. C'est pourquoi un seuil de taille minimale et un seuil de différence ont été ajoutés pour éviter ces cas.

- **Réglage des Paramètres** : Le réglage des paramètres, tels que les seuils de seuillage adaptatif et de différence, ainsi que les paramètres de flou gaussien, était un défi. Il a fallu des essais et des erreurs pour trouver les valeurs appropriées pour chaque pièce et type d'objet.

- **Gestion de Dossiers Dynamique** : Le programme est capable de traiter des images provenant de différents dossiers et avec des noms de fichiers variables, ce qui rend l'automatisation de la détection plus robuste.

## Groupe

- Membre 1 : Philippe Lavoie LAVP01069901
- Membre 2 : Baptiste David DAVB30100100
- Membre 3 : Théodore Perrin PERT03090200
- Membre 4 : Gaëlle Thibaudat THIG24539900

N'hésitez pas à ajouter les noms complets des membres de votre groupe dans les sections appropriées.
