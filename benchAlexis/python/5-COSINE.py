import json
import csv
from pathlib import Path
import numpy as np

# Chemins des fichiers (relatifs au dossier du script)
BASE_DIR = Path(__file__).resolve().parent
embeddings_file = BASE_DIR / "vectors-OrangeIA.json"
axis_file = BASE_DIR / "axis-vectors-OrangeIA.json"
output_csv = BASE_DIR / "CosineCompute-OrangeIA.csv"

def cosine_similarity(vec1, vec2):
    v1 = np.array(vec1)
    v2 = np.array(vec2)
    dot_product = np.dot(v1, v2)
    norm1 = np.linalg.norm(v1)
    norm2 = np.linalg.norm(v2)
    if norm1 == 0 or norm2 == 0:
        return 0.0
    return dot_product / (norm1 * norm2)

def main(embeddings_file, axis_file, output_csv):
    # Charger les fichiers JSON
    with open(embeddings_file, 'r', encoding='utf-8') as f:
        embeddings_data = json.load(f)
    with open(axis_file, 'r', encoding='utf-8') as f:
        axis_data = json.load(f)

    # Extraire les labels et vecteurs
    embeddings = embeddings_data['labels']
    axes = axis_data['labels']

    results = []

    for emb in embeddings:
        emb_label = emb['name']
        emb_vector = emb['vector']
        for axis in axes:
            axis_label = axis['name']
            axis_vector = axis['vector']
            cos = cosine_similarity(emb_vector, axis_vector)
            sqrt_val = np.sqrt(1 - cos**2) if abs(cos) <= 1 else 0  # sécurité numérique
            results.append([
                emb_label,
                axis_label,
                cos,
                sqrt_val
            ])

    # Écriture dans un CSV
    with open(output_csv, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow([
            'label_name_embeddings',
            'label_name_axis',
            'cosine_value',
            'sqrt(1-cosine^2)'
        ])
        for row in results:
            writer.writerow(row)

    print(f"Cosines générés et enregistrées dans : {output_csv}")

if __name__ == "__main__":
    main(embeddings_file, axis_file, output_csv)
