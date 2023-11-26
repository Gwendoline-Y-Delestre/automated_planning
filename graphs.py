import pandas as pd
import matplotlib.pyplot as plt
import os

# Lire le fichier CSV
nom_fichier = "resultats.csv"
data = pd.read_csv(nom_fichier)

# Afficher les premières lignes du DataFrame pour vérification
print(data.head())

# Diviser les données par domaine
domaines = data['domain'].unique()

# Créer un répertoire pour sauvegarder les images s'il n'existe pas
if not os.path.exists('images'):
    os.makedirs('images')

# Parcourir chaque domaine
for domaine in domaines:
    # Sélectionner les données pour le domaine spécifique
    domaine_data = data[data['domain'] == domaine]

   # Créer un graphique pour le temps
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.plot(domaine_data['n_problem'], domaine_data['MCTS_temps'], label='MCTS', marker='o')
    plt.plot(domaine_data['n_problem'], domaine_data['HSP_temps'], label='HSP', marker='o')
    plt.title(f'Temps pour {domaine}')
    plt.xlabel('Problème')
    plt.ylabel('Temps')
    plt.legend()

    # Créer un graphique pour la taille
    plt.subplot(1, 2, 2)
    plt.plot(domaine_data['n_problem'], domaine_data['MCTS_taille'], label='MCTS', marker='o')
    plt.plot(domaine_data['n_problem'], domaine_data['HSP_taille'], label='HSP', marker='o')
    plt.title(f'Taille pour {domaine}')
    plt.xlabel('Problème')
    plt.ylabel('Taille')
    plt.legend()


    # Sauvegarder les graphiques au format image
    plt.tight_layout()
    image_path = os.path.join('images', f'{domaine}_graph.png')
    plt.savefig(image_path)
    plt.close()  # Fermer la figure pour libérer la mémoire

print("Les graphiques ont été sauvegardés dans le dossier 'images'.")
