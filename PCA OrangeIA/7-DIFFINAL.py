import json
import csv
import itertools
import math

# Chemins des fichiers
input_file = r"c:\Hackathon\ProjetPCA\OrangeIA\TCD-OrangeIA.json"
output_file = r"c:\Hackathon\ProjetPCA\OrangeIA\DIF-OrangeIA.json"
dif_data_csv = r"c:\Hackathon\ProjetPCA\OrangeIA\DIF-OrangeIA.csv"

# Paramètre pour activer/désactiver l'écriture du CSV
write_csv = True  # Passe à True pour générer le CSV

# Lecture du fichier JSON d'entrée
with open(input_file, 'r', encoding='utf-8') as f:
    data = json.load(f)

labels = data["Labels"]

# Préparation des combinaisons (ordre : vecteur1 - vecteur2)
combinations = list(itertools.combinations(labels, 2))

# Pour le JSON de sortie
diff_labels = []

# Pour le CSV de sortie
csv_rows = []

for vec1, vec2 in combinations:
    name1 = vec1["name"]
    name2 = vec2["name"]
    v1 = vec1["vector"]
    v2 = vec2["vector"]
    # Calcul de la différence (v1 - v2)
    diff_vector = [a - b for a, b in zip(v1, v2)]
    # Calcul de la norme
    norm = math.sqrt(sum(x**2 for x in diff_vector))
    # Ajout au JSON
    diff_labels.append({
        "name": f"{name1} <-> {name2}",
        "norm": norm,
        "vector": diff_vector
    })
    # Ajout au CSV si demandé
    if write_csv:
        for idx, value in enumerate(diff_vector):
            csv_rows.append({
                "name_vector_1": name1,
                "name_vector_2": name2,
                "dif_value": value
            })

# Tri du JSON par ordre croissant de la norme
diff_labels_sorted = sorted(diff_labels, key=lambda x: x["norm"])

# Format final pour le JSON
result_json = {"Labels": diff_labels_sorted}

# Écriture du fichier JSON de sortie
with open(output_file, 'w', encoding='utf-8') as f:
    json.dump(result_json, f, indent=2, ensure_ascii=False)

# Écriture du fichier CSV de sortie (si activé)
if write_csv:
    with open(dif_data_csv, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=["name_vector_1", "name_vector_2", "dif_value"])
        writer.writeheader()
        writer.writerows(csv_rows)

print("Traitement terminé. Fichier JSON généré :", output_file)
if write_csv:
    print("Fichier CSV généré :", dif_data_csv)
else:
    print("Écriture du CSV désactivée (write_csv=False)")