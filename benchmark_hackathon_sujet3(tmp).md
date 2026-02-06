# Benchmark Hackathon – Sujet 3 (Optimisation machine)

## 1. Objectif du projet
- **Démonstration visée** : montrer qu’un même calcul, sur le même dataset et la même machine, peut avoir des performances très différentes selon l’implémentation.
- **Lien avec le sujet 3** : le cas couvre performance CPU, proxy énergie (CPU time), et rigueur de mesure (protocole reproductible).
- **Message clé pour le jury** : l’optimisation utile n’est pas “magique”, elle vient d’un choix d’implémentation mesuré proprement et interprété correctement.

## 2. Choix techniques et justifications
| Choix | Justification |
|---|---|
| Calcul benchmarké = cosine similarity all-pairs | Calcul dense, fréquent en ML, déterministe, et déjà présent dans le pipeline réel. |
| Ne pas benchmarker tout le pipeline | Un benchmark global mélange I/O, NLP, parsing, écriture CSV, donc impossible d’attribuer les gains au calcul CPU ciblé. |
| Rester dans un seul langage (Python) | On isole l’effet “implémentation” sans confondre avec l’effet “langage/runtime”. |
| Comparer naïf vs vectorisé | Contraste clair, pédagogique, rapide à implémenter en 1 journée, et effet mesurable généralement net. |
| Exclure NLP, I/O, BIOS tuning, multi-threading | Trop de facteurs confondants, faible reproductibilité hackathon, et coût de mise en place disproportionné. |
| Mono-cœur + BLAS mono-thread | Réduit la variabilité, facilite la comparaison juste entre versions. |

## 3. Description du calcul benchmarké
- **Définition** : pour deux vecteurs \(u, v \in \mathbb{R}^D\),  
  \[
  \cos(u,v)=\frac{u \cdot v}{\|u\|\|v\|}
  \]
- **All-pairs** : on calcule une matrice \(C \in \mathbb{R}^{N \times M}\), avec \(C_{ij}=\cos(E_i, A_j)\).
- **Dimensions typiques** : \(N\) embeddings, \(M\) axes, \(D\) dimensions (exemple courant du projet : \(N \approx 560\), \(M \approx 15\), \(D \approx 96\)).
- **Pourquoi CPU-bound** : essentiellement multiplications/additions en mémoire, sans attente réseau/disque quand I/O est exclu.
- **Pourquoi déterministe** : mêmes entrées, même ordre d’opérations par implémentation, sortie numérique stable à tolérance flottante près.

## 4. Implémentations comparées

### Version naïve
- **Principe** : triple boucle Python `for i in N`, `for j in M`, `for k in D`.
- **Traitement** : dot product et normes recalculés paire par paire.
- **Coût** : complexité \(O(NMD)\), avec constante très élevée due à l’overhead interpréteur (bytecode, objets Python, appels fréquents).

### Version optimisée
- **Principe** : conversion en `ndarray`, normalisation vectorisée, puis produit matriciel `E_unit @ A_unit.T`.
- **Traitement** : normes calculées une fois par vecteur, puis GEMM BLAS pour tous les couples.
- **Coût** : \(O(NMD)\) aussi en asymptotique, mais exécution en C/BLAS, accès mémoire contigu, meilleure localité cache, SIMD, et réduction drastique de l’overhead Python.
- **Impact machine concret** : moins d’instructions interprétées, meilleure utilisation L1/L2/L3, kernels BLAS optimisés (blocking/prefetch), vector units mieux exploitées.

## 5. Méthodologie de benchmark
1. **Environnement figé** : noter CPU, OS, version Python, version NumPy/BLAS.
2. **Configuration threads** : `OMP_NUM_THREADS=1`, `MKL_NUM_THREADS=1`, `OPENBLAS_NUM_THREADS=1`, affinité CPU (ex. `taskset -c 2`).
3. **Isolation calcul** : charger dataset une seule fois avant mesure; aucune lecture/écriture pendant le chrono.
4. **Warm-up** : 5 exécutions non mesurées par version.
5. **Runs mesurés** : 30 runs par version.
6. **Ordre d’exécution** : alterné (`ABAB...`) ou blocs randomisés pour limiter dérive thermique/scheduler.
7. **Métriques** : `wall_time_ns` (`perf_counter_ns`) et `cpu_time_ns` (`process_time_ns`).
8. **Stabilité temporelle** : si un run est trop court, appliquer un `repeat` interne (boucle K) et diviser le temps par K.
9. **Validation numérique** : `assert_allclose(naive, opt, rtol=1e-9, atol=1e-9)`.
10. **Sortie résultats** : CSV brut run-par-run + résumé statistique (médiane, IQR, speedup).

## 6. Résultats attendus et interprétation
- **Lecture des graphes** :  
  - Boxplot `wall_time` par version.  
  - Boxplot `cpu_time` par version.  
  - Barre de speedup médian (`naive / optimisée`).
- **Signal attendu** : la version vectorisée doit réduire fortement `wall_time` et `cpu_time`.
- **Interprétation CPU** : la baisse vient surtout de la suppression de l’overhead Python et de l’exécution BLAS/SIMD.
- **Proxy énergie** : baisse de `cpu_time` = moins de temps CPU actif, donc proxy raisonnable d’une baisse d’énergie dynamique (sans prétendre mesurer des Joules absolus).
- **Conclusion défendable** : “À workload égal, la vectorisation NumPy+BLAS améliore nettement la performance et le proxy énergie, avec un protocole reproductible.”
- **Limite à expliciter** : conclusion valide pour ce calcul, ce dataset et cette machine; pas généralisation universelle sans réplication.
