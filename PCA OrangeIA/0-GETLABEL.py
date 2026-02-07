import json

# Chemins des fichiers
input_file = r"c:\Hackathon\ProjetPCA\OrangeIA\ontology-OrangeIA.json"
output_file = r"c:\Hackathon\ProjetPCA\OrangeIA\labels-OrangeIA.json"

def extract_names_recursive(data, labels):
    """
    Fonction récursive pour extraire tous les champs 'name' dans une structure JSON imbriquée.
    """
    if "name" in data:
        labels.append(data["name"])
    if "children" in data:
        for child in data["children"]:
            extract_names_recursive(child, labels)

def find_node_by_filter(data, filter_key, filter_value):
    """
    Fonction récursive pour trouver un nœud correspondant au filtre (id ou name).
    """
    if data.get(filter_key) == filter_value:
        return data
    if "children" in data:
        for child in data["children"]:
            result = find_node_by_filter(child, filter_key, filter_value)
            if result:
                return result
    return None

def extract_labels(input_file, output_file, filter_key=None, filter_value=None):
    """
    Extrait les 'name' depuis un fichier JSON imbriqué en fonction d'un filtre, ou tout si aucun filtre n'est fourni.
    """
    # Charger le fichier JSON
    with open(input_file, "r", encoding="utf-8") as file:
        data = json.load(file)

    # Initialiser la liste des labels
    labels = []

    # Appliquer le filtre si un filtre est fourni
    if filter_key and filter_value:
        for root_label in data.get("root_labels", []):
            filtered_node = find_node_by_filter(root_label, filter_key, filter_value)
            if filtered_node:
                extract_names_recursive(filtered_node, labels)
                break
    else:
        # Si aucun filtre n'est fourni, extraire tous les noms
        for root_label in data.get("root_labels", []):
            extract_names_recursive(root_label, labels)

    # Créer la structure de sortie
    output_data = {"Labels": labels}

    # Sauvegarder les résultats dans un fichier JSON
    with open(output_file, "w", encoding="utf-8") as outfile:
        json.dump(output_data, outfile, indent=4, ensure_ascii=False)

    print(f"Labels extraits et enregistrés dans : {output_file}")

if __name__ == "__main__":
    # Exemple d'utilisation : définir un filtre
    filter_key = ""  # Peut être "id" ou "name"
    filter_value = ""  # Valeur à rechercher

    # Appeler la fonction principale
    extract_labels(input_file, output_file, filter_key, filter_value)