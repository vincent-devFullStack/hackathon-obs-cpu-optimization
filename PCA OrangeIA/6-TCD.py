import csv
import json
from collections import defaultdict

# Chemins des fichiers
input_file = r"c:\Hackathon\ProjetPCA\OrangeIA\CosineCompute-OrangeIA.csv"
output_file = r"c:\Hackathon\ProjetPCA\OrangeIA\TCD-OrangeIA.json"

# Dictionnaire temporaire pour stocker les valeurs
temp = defaultdict(dict)

# Première lecture pour récupérer tous les axes
axes_set = set()
with open(input_file, newline='', encoding='utf-8') as f:
    reader = csv.DictReader(f)
    for row in reader:
        axes_set.add(row['label_name_axis'])
axes_order = sorted(list(axes_set))

# Deuxième lecture pour organiser les valeurs
with open(input_file, newline='', encoding='utf-8') as f:
    reader = csv.DictReader(f)
    for row in reader:
        label = row['label_name_embeddings']
        axis = row['label_name_axis']
        value = float(row['cosine_value'])
        temp[label][axis] = value

# Construction de la liste finale
labels_list = []
for label, axis_values in temp.items():
    vector = [axis_values[axis] for axis in axes_order]
    labels_list.append({"name": label, "vector": vector})

# Format final
result = {"Labels": labels_list}

# Écriture dans le fichier de sortie
with open(output_file, 'w', encoding='utf-8') as f:
    json.dump(result, f, indent=2, ensure_ascii=False)

print("Traitement terminé. Fichier JSON généré :", output_file)