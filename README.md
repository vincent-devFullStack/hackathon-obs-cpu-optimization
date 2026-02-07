# hackathon-obs-cpu-optimization

Suite de benchmark multi-langages pour un calcul de similarite cosinus (all-pairs) avec controle de reproductibilite et metriques systeme.

## Prerequis (Linux)

Installer les outils systeme :

```bash
sudo apt update
sudo apt install -y \
  python3 python3-venv python3-pip \
  make gcc g++ \
  openjdk-21-jdk \
  golang-go \
  rustc cargo \
  util-linux
```

Optionnel (metriques `perf stat`) :

```bash
sudo apt install -y linux-tools-common linux-tools-generic
```

## Installation Python (venv obligatoire)

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Generer le dataset

```bash
python3 data/generate_data.py \
  --output-dir data \
  --metadata metadata.json \
  --N 2000 --M 15 --D 96 \
  --seed 42 --force
```

Le script ecrit `data/metadata.json`, `data/E.f64`, `data/A.f64`.

## Lancer les benchmarks (2 modes separes)

Le projet fournit 3 scripts :
- `run_single_core.sh` : run de reference comparable (single-core)
- `run_multi_core.sh` : run de debit/scalabilite (multi-core)
- `bench_three_mode` : lance les 2 runs a la suite

### 1) Single-core (baseline comparative)

```bash
./run_single_core.sh 2 data/metadata.json
```

- `2` = coeur CPU fixe (affinite mono-core)
- baseline par defaut : `python-naive`
- profil C/C++ : `portable`

Sorties :
- `results/results_single_core.csv`
- `results/summary_single_core.json`

### 2) Multi-core (throughput / scalabilite)

```bash
./run_multi_core.sh 8 data/metadata.json
```

- `8` = nombre de threads runtime (`OMP`, `BLAS`, `Go`, `Rayon`, etc.)
- argument optionnel 3 = cpu-set multi-core (defaut: `0-(T-1)`, ex `0-7`)
- baseline par defaut : `python-naive`
- profil C/C++ : `native`

Sorties :
- `results/results_multi_core.csv`
- `results/summary_multi_core.json`
- `results/results_multi_core_scalable.csv`
- `results/summary_multi_core_scalable.json`

### 3) Lancer les 2 d'un coup

```bash
./bench_three_mode 2 8 data/metadata.json
```

Parametres :
- argument 1 : coeur single-core
- argument 2 : nombre de threads multi-core
- argument 3 : chemin metadata
- argument 4 (optionnel) : cpu-set multi-core (ex: `0-7`)

Chaque script preserve l'historique : si un fichier de sortie existe deja, il est renomme avec un suffixe timestamp.

## Java: options JVM robustes

Le runner valide les options JVM avant execution (`java -version` en dry-check).

- Si `--java-opts` est invalide, fallback automatique :
1. `strict_single_core`
2. `relaxed_single_core`
- Option explicitement bloquee : `-XX:CICompilerCount=1`

Exemple Java uniquement :

```bash
python runner/run_all.py \
  --metadata data/metadata.json \
  --warmup 5 --runs 30 --repeat 50 \
  --stability-enable --stability-mode wait --stability-timeout-sec 60 \
  --cpu-util-max 20 --disk-io-mbps-max 5 --mem-available-min-mb 2048 \
  --enforce-single-thread --cpu-affinity 2 \
  --impls java-naive \
  --java-opts=-XX:CICompilerCount=1
```

Le fallback effectif est trace dans `results/summary.json` (`java_control`).

## Nettoyage rapide

Pour supprimer les artefacts de build :

```bash
rm -f c/benchmark_c cpp/benchmark_cpp go/benchmark_go java/*.class
```
