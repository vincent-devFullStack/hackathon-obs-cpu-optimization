import json
import numpy as np
from sklearn.decomposition import PCA
import csv

# Chemins des fichiers
input_file = r"c:\Hackathon\ProjetPCA\OrangeIA\vectors-OrangeIA.json"
output_file = r"c:\Hackathon\ProjetPCA\OrangeIA\vectors-PCA-OrangeIA.json"
projection_matrix_csv = r"c:\Hackathon\ProjetPCA\OrangeIA\PCAProjectionMatrix-testes.csv"

def apply_pca_and_generate_projection(input_file, output_file, projection_matrix_csv, variance_threshold=0.95):
    """
    Applique PCA pour réduire la dimensionnalité des vecteurs et génère une matrice de projection.
    """
    # Charger le fichier JSON
    with open(input_file, "r", encoding="utf-8") as file:
        data = json.load(file)

    # Extraire les vecteurs et les labels correspondants
    vectors = []
    labels = []

    for item in data.get("labels", []):
        vector = item.get("vector")
        label_name = item.get("name")
        if vector and label_name:
            vectors.append(vector)
            labels.append(label_name)

    # Convertir les vecteurs en un tableau numpy
    vectors = np.array(vectors)

    # Appliquer PCA
    pca = PCA()
    vectors_pca = pca.fit_transform(vectors)

    # Calculer le nombre de dimensions à conserver pour atteindre le seuil de variance
    cumulative_variance = np.cumsum(pca.explained_variance_ratio_)
    num_dimensions = np.argmax(cumulative_variance >= variance_threshold) + 1

    # Réduire les vecteurs à la nouvelle dimensionnalité
    vectors_reduced = vectors_pca[:, :num_dimensions]

    # Créer un nouveau JSON avec les vecteurs réduits
    reduced_data = {"labels": []}
    for label, vector in zip(labels, vectors_reduced):
        reduced_data["labels"].append({
            "label_name": label,
            "vector": vector.tolist()
        })

    # Sauvegarder le nouveau fichier JSON
    with open(output_file, "w", encoding="utf-8") as outfile:
        json.dump(reduced_data, outfile, indent=4)

    # Extraire la matrice de projection (composantes principales)
    projection_matrix = pca.components_[:num_dimensions]

    # Sauvegarder la matrice de projection dans un fichier CSV
    with open(projection_matrix_csv, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow([f"Dimension {i+1}" for i in range(projection_matrix.shape[1])])  # En-têtes des colonnes
        writer.writerows(projection_matrix)

    print(f"Réduction de dimension terminée. Résultat enregistré dans : {output_file}")
    print(f"Matrice de projection enregistrée dans : {projection_matrix_csv}")

if __name__ == "__main__":
    apply_pca_and_generate_projection(input_file, output_file, projection_matrix_csv, variance_threshold=0.95)