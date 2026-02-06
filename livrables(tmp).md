## 3. Livrables attendus — interprétation pragmatique

Le slide officiel demande :

- Scripts de bench + tableau de résultats (avant / après)
- Dashboard Netdata prêt à rejouer
- Note de synthèse _“performance vs énergie : ce qui marche”_

Ces points ne doivent **pas être interprétés littéralement**, mais traduits en livrables pertinents et défendables dans le temps imparti.

---

### ✅ Livrable 1 — Scripts de bench + résultats

**Statut : obligatoire**  
**Alignement : total**

Structure recommandée :

```text
bench/
├── cosine_naive.py
├── cosine_numpy.py
├── bench.py
├── results.csv
└── summary.md

```

results.csv : résultats bruts, run par run

summary.md : statistiques synthétiques (médiane, IQR, speedup)

Ce livrable constitue le cœur scientifique du projet.

### ⚠️ Livrable 2 — Dashboard Netdata

Statut : optionnel / substituable intelligemment

Plutôt qu’un dashboard Netdata générique, nous proposons :

- 2 à 3 graphes ciblés :

- wall_time

- cpu_time

- speedup

Ces graphes sont générés directement à partir des résultats du benchmark (ex. matplotlib).

Dans la note de synthèse, nous explicitons le choix :

    “Nous remplaçons un dashboard générique par des graphes ciblés issus des mesures du benchmark, plus directement exploitables et reproductibles.”

Ce choix est plus pertinent qu’un dashboard figé dépendant de l’environnement.

### ✅ Livrable 3 — Note de synthèse “performance vs énergie”

Statut : obligatoire
Base existante : README.md

Il suffit d’ajouter la section suivante au document principal :

## 7. Synthèse : performance vs énergie

- La version vectorisée réduit fortement le temps d’exécution (wall time).
- Le CPU time diminue dans les mêmes proportions, indiquant une baisse du temps CPU actif.
- À charge identique, une réduction du CPU time est un proxy raisonnable d’une réduction de l’énergie dynamique consommée.
- Le gain observé provient essentiellement de la suppression de l’overhead interpréteur Python et de l’utilisation de kernels BLAS optimisés.

### Ce qui marche

- Vectorisation NumPy
- Normalisation hors boucle
- Calcul dense via GEMM

### Ce qui ne marche pas / peu pertinent ici

- Tuning système sans contrôle strict
- Benchmarks pipeline end-to-end avec I/O dominant
- Comparaisons multi-langages non équivalentes
  Cette section permet de cocher exactement l’attendu du slide.

### 4. Verdict mentor (clair et honnête)

- ❌ Il n’est pas nécessaire d’installer Netdata

- ❌ Il ne faut pas toucher au BIOS

- ✅ Il faut assumer une approche applicative propre et maîtrisée

Rendu final le plus adapté

    Scripts de benchmark

    CSV de résultats

    README / note de synthèse

2 à 3 graphes clairs

👉 Cette approche est plus alignée avec l’esprit du sujet que la plupart des solutions centrées uniquement sur l’outillage.

---

### Dernier conseil (rapide)

Si tu veux être **ultra clean**, tu peux :

- mettre cette section dans `README.md`
- ou dans un `DELIVERABLES.md` séparé, référencé depuis le README

Dans les deux cas, tu es **parfaitement conforme** et **très au-dessus de la moyenne**.

Si tu veux, je peux aussi te fournir :

- une **version ultra-courte “jury slide”**
- ou une **checklist démo en Markdown** (avant de passer devant eux).
