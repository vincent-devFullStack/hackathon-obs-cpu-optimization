# hackathon-obs-cpu-optimization

Benchmark CPU – implémentation naïve vs vectorisée d’un calcul de cosine similarity.

Ce projet de hackathon (Sujet 3 – Optimisation CPU) vise à démontrer qu’un même calcul, exécuté sur les mêmes données et la même machine, peut présenter des performances très différentes selon l’implémentation.  
L’objectif est de mesurer et d’interpréter l’impact de ces choix d’implémentation sur la performance CPU et un proxy énergie, à l’aide d’une méthodologie reproductible.

## Installation et exécution

Ce projet utilise Python 3 et dépend uniquement de **NumPy**.  
Afin de garantir la reproductibilité et d’éviter tout conflit avec le Python système, l’exécution se fait dans un **environnement virtuel**.

### Prérequis

- Linux / WSL / Ubuntu
- Python ≥ 3.10
- Accès aux droits `sudo` (pour installer les paquets système si nécessaire)

---

### 1. Cloner le dépôt

```bash
git clone <https://github.com/vincent-devFullStack/hackathon-obs-cpu-optimization.git>
cd hackathon-obs-cpu-optimization
```

---

### 2. Installer le support des environnements virtuels (une seule fois)

Sur Debian / Ubuntu :

```bash
sudo apt update
sudo apt install -y python3-venv python3-pip
```

---

### 3. Créer et activer l’environnement virtuel

À la racine du dépôt :

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Vérification (optionnelle) :

```bash
python --version
```

---

### 4. Installer les dépendances Python

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

---

### 5. Lancer le benchmark

```bash
cd bench
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 taskset -c 2 python bench.py
```

La sortie affiche :

- les temps médians (`wall_time`, `cpu_time`)
- le speedup entre l’implémentation naïve et la version optimisée

Les résultats bruts sont également sauvegardés dans bench/results.csv.

## Notes importantes

- Le benchmark est volontairement **mono-cœur** et **mono-thread BLAS** afin de garantir des mesures stables et comparables.
- Le benchmark mesure uniquement le **noyau de calcul CPU** ; le chargement des données et toute écriture disque sont volontairement exclus afin d’éviter les biais liés à l’I/O.
- Le **CPU time** est utilisé comme proxy énergie : à charge identique, une réduction du temps CPU actif implique une réduction de l’énergie dynamique consommée.
