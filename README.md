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

## Lancer tous les tests

Commande de reference :

```bash
python runner/run_all.py \
  --metadata data/metadata.json \
  --warmup 5 --runs 30 --repeat 50 \
  --stability-enable --stability-mode wait --stability-timeout-sec 60 \
  --cpu-util-max 20 --disk-io-mbps-max 5 --mem-available-min-mb 2048 \
  --enforce-single-thread --cpu-affinity 2 \
  --impls c-naive,cpp-naive,rust-naive,go-naive,java-naive,python-naive,python-numpy
```

Sorties :
- `results/results.csv`
- `results/summary.json`

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
