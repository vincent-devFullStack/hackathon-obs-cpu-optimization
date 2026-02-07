import json

# Chemins des fichiers
input_file = r"c:\Hackathon\ProjetPCA\OrangeIA\vectors-PCA-OrangeIA.json"
output_file = r"c:\Hackathon\ProjetPCA\OrangeIA\minmax-vectors-OrangeIA.json"

def process_vectors(input_file, output_file):
    """
    Analyse les vecteurs et génère une liste de labels basée sur les valeurs des vecteurs.
    """
    # Charger le fichier JSON
    with open(input_file, "r", encoding="utf-8") as file:
        data = json.load(file)

    labels_data = data.get("labels", [])
    
    # Calculer dim (nombre de positions dans un vecteur) et nb_vectors (nombre total de vecteurs)
    if not labels_data:
        print("Aucun label trouvé dans le fichier.")
        return

    dim = len(labels_data[0]["vector"])  # Nombre de positions dans un vecteur
    nb_vectors = len(labels_data)  # Nombre total de vecteurs

    # Vérification pour éviter les erreurs de division
    if nb_vectors < 2:
        print("Le nombre de vecteurs est insuffisant pour effectuer le traitement.")
        return
    if nb_vectors > 20:
        print("Trop de vecteurs donc reduction à 20 Top-Min")
        nb_vectors = 20

    # Générer les listes de labels
    output_data = []
    for i in range(dim):
        # Trier les labels par la valeur de la position i dans le vecteur
        sorted_labels = sorted(labels_data, key=lambda x: x["vector"][i])

        # Récupérer les (nb_vectors/2) plus petites valeurs
        smallest_labels = [label["label_name"] for label in sorted_labels[:nb_vectors // 2]]
        
        # Récupérer les (nb_vectors/2) plus grandes valeurs
        largest_labels = [label["label_name"] for label in sorted_labels[-(nb_vectors // 2):]]

        # Combiner les deux listes
        combined_labels = smallest_labels + largest_labels

        # Ajouter les données au format demandé
        output_data.append({
            "label_list_id": i + 1,
            "label_list": combined_labels
        })

    # Sauvegarder les résultats dans un fichier JSON
    with open(output_file, "w", encoding="utf-8") as outfile:
        json.dump(output_data, outfile, indent=4)

    print(f"Fichier JSON généré avec succès : {output_file}")

if __name__ == "__main__":
    process_vectors(input_file, output_file)