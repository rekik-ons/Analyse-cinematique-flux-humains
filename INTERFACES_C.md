# Partie C — interfaces à convenir

- C fournit `PoseDetector.extract_keypoints(frame, timestamp_ms)` : ndarray float32 (33,2), coordonnées image normalisées, ou None. `visibility` contient 33 scores. `get_hip_center` retourne (2,) ou None. `in_zone(point, polygon)` retourne bool.
- A fournit la fonction de normalisation, la gestion des 30 images et l'inférence MLP (classe, confiance), les poids, les dépendances, la cadence attendue et la stratégie des points manquants. Ne pas confondre coordonnées image et normalisation pour apprentissage.
- B fournit un module cinématique (timestamp en secondes, point ou None, calibration, vitesse m/s, accélération m/s²) et un module de règles (état zone, cinématique, pose, événements horodatés). Définir signatures, valeurs manquantes et réinitialisation par vidéo.
- C appelle ensuite ces modules dans main.py, affiche leurs résultats et transmet leurs événements à EventLogger.log(timestamp_s,event,source,details).
- Aucun résultat MLP ou cinématique fictif. Le CSV actuel ne contient que des changements d'état géométrique, pas les événements métier de B.

## Utilisation
`python main.py` ou `python main.py --video chemin.avi`. S sauvegarde, R réinitialise, P pause, Q quitte. Clic gauche ajoute un sommet, droit annule. Configuration séparée par source. Reconfigurer si caméra/scène change. Utiliser des vidéos autorisées, une personne par scène. Les zones sont dans l'image, pas calibrées au sol. Heatmap de présence 2D pondérée par temps observé, pas carte métrique. La vidéo webcam est encodée à cadence fixe déclarée par caméra : durée possiblement différente de la durée réelle. FPS affiché instantané, moyenne traitement exclut l'attente d'affichage. Aucun enregistrement des images originales.

Ne pas publier data/, outputs/, .venv/, __pycache__/ ni les vidéos originales. Ne pas remplacer le README commun sans accord. Tester les polygones simples non croisés.
