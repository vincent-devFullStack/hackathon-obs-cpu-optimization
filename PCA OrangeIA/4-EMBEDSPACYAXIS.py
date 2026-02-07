import json
import spacy

# Chemins des fichiers
input_file = r"c:\Hackathon\ProjetPCA\OrangeIA\axis-OrangeIA.json"
output_file = r"c:\Hackathon\ProjetPCA\OrangeIA\axis-vectors-OrangeIA.json"

def generate_embeddings(input_file, output_file):
    """
    Génère les embeddings pour les mots clés de la liste Labels en utilisant spaCy.
    """
    # Charger le modèle spaCy
    nlp = spacy.load("pt_core_news_sm")

    # Charger le fichier JSON
    with open(input_file, "r", encoding="utf-8") as file:
        data = json.load(file)

    labels = data.get("Labels", [])

    # Générer les embeddings
    embeddings = []
    for label in labels:
        label_vector = nlp(label).vector.tolist()
        embeddings.append({
            "name": label,
            "vector": label_vector
        })

    # Ajouter la structure en en-tête {"labels": [...]}
    output_data = {"labels": embeddings}

    # Sauvegarder les résultats dans un fichier JSON
    with open(output_file, "w", encoding="utf-8") as outfile:
        json.dump(output_data, outfile, indent=4)

    print(f"Embeddings générés et enregistrés dans : {output_file}")

if __name__ == "__main__":
    generate_embeddings(input_file, output_file)