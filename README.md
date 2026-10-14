# Apprentissage de projections de correction

**Spécialiser a posteriori un espace d'embedding sur une base de données.**

Cette implémentation apprend des directions à retirer des embeddings du corpus pour améliorer les classements de recherche, sans modifier les embeddings des requêtes ni réentraîner l'encodeur. Elle propose une variante globale, une variante par clustering et deux références non supervisées : *whitening* et *all-but-the-top*.

## Installation

Depuis la racine du dépôt :

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r code/requirements-local.txt
```

Le premier téléchargement des données et du modèle nécessite Internet. Avec MiniLM, les embeddings et la correction sont ensuite calculés localement, sans clé d'API ni serveur de base de données.

## Premier essai

```bash
bash run_local_demo.sh --dataset scifact --seed 42 --epochs 1 --iterations 1
```

Cette commande télécharge SciFact, calcule ses embeddings avec MiniLM, les conserve dans `local_db_demo/` et exécute les deux variantes avec un budget réduit. Les exécutions suivantes réutilisent les embeddings enregistrés.

## Exécution

Le budget par défaut est de 30 *epochs* et 30 itérations externes.

```bash
bash run_local_demo.sh --dataset scifact --seed 42
bash run_local_demo.sh --dataset nfcorpus --variant requete --seed 42
bash run_local_demo.sh --dataset scifact --variant cluster --n-clusters 5 --seed 42
```

Les jeux de l'article sont `scifact`, `nfcorpus`, `arguana` et `scidocs`. Sans `--seed`, la répartition entraînement/validation/test est aléatoire. Il s'agit de découpages répétés, pas d'une rotation sur des plis fixes.

| Paramètre | Valeurs ou rôle |
|---|---|
| `--variant` | `requete` : globale ; `cluster` : par clustering ; `both` : les deux |
| `--embedder-backend` | `sentence_transformers` ou `openai` |
| `--embedder-model` | Nom du modèle |
| `--epochs`, `--iterations` | Budget d'apprentissage |
| `--n-clusters` | Nombre de clusters |
| `--seed` | Graine du découpage et de la recherche de directions |

Le nom technique `requete` désigne la variante globale, pas une projection des requêtes. Le script de démonstration utilise le critère d'acceptation strict et le poids de préservation intermédiaire.

Pour régler directement ces paramètres, après l'ingestion de SciFact :

```bash
PARASITIC_LOCAL_DB=local_db_demo python code/0_math_approach/debruitage_evalue_requete.py \
  --dataset "mteb/scifact__emb__sentence-transformers__sentence-transformers-all-minilm-l6-v2" \
  --epochs 30 --iterations 30 --val-fraction 0.2 --test-fraction 0.2 \
  --gate-acceptance-strategy strict --search-strictness strict --seed 42
```

`PARASITIC_LOCAL_DB` est un identifiant technique historique conservé pour ne pas changer le fonctionnement du code ; il désigne le dossier de stockage local des embeddings, sans signification particulière pour le nom actuel de la méthode.

Le critère d'acceptation peut être `strict` ou `adaptive`. Le poids de préservation est réglé par `--search-strictness` : `very_strict` correspond à *error-scaled*, `strict` à *mid* et `relaxed` à *relaxed*. Le script `debruitage_evalue_cluster.py` accepte en plus `--n-clusters`.

## Références non supervisées

Les transformations sont ajustées sur les embeddings du corpus et appliquées au corpus seul. Les requêtes restent inchangées.

- *Whitening* : centrage puis transformation calculée à partir de la covariance du corpus. Su et al. (2021), *Whitening Sentence Representations for Better Semantics and Faster Retrieval*.
- *All-but-the-top* : centrage puis retrait des composantes principales dominantes. Mu et al. (2018), *All-but-the-Top: Simple and Effective Postprocessing for Word Representations*.

Après l'ingestion de SciFact :

```bash
PARASITIC_LOCAL_DB=local_db_demo python code/0_math_approach/debruitage_baseline.py \
  --dataset "mteb/scifact__emb__sentence-transformers__sentence-transformers-all-minilm-l6-v2" \
  --baseline-method whitening --val-fraction 0.2 --test-fraction 0.2 --seed 42
```

Pour *all-but-the-top*, remplacer `whitening` par `abtt`. L'option `--abtt-components` règle le nombre de composantes retirées.

## Contenu et périmètre

- `code/0_math_approach/` : ingestion, variantes de correction et références.
- `code/local_store.py` : stockage des embeddings dans des fichiers TinyDB.
- `code/local_embedders.py` : encodeur local et client pour les API compatibles OpenAI.
- `supplement/complements.pdf` : complément expérimental de l'article (validation croisée par couple, comparaison aux références pour les cinq encodeurs, ablations, durées), avec sa source LaTeX.

Le complément distingue trois périmètres : résultats de test sur cinq découpages, résultats de test sur le découpage initial, et ablations calculées sur l'ensemble des requêtes du découpage initial.

Ce dépôt permet d'exécuter la méthode. Il ne contient ni les artefacts bruts de la campagne ni ses scripts d'agrégation : il ne permet donc pas de régénérer les tableaux à l'identique.

Les dépendances ne sont pas verrouillées. Une graine fixe ne garantit pas des résultats identiques entre machines ou versions de modèles. La préservation est vérifiée en validation, sans garantie d'absence de dégradation sur de nouvelles requêtes.

Gemini et Titan ne sont pas intégrés à cette version autonome. Une API compatible OpenAI nécessite un accès réseau et une clé `OPENAI_API_KEY` ; son utilisation peut être payante. Ne pas publier de clés, de fichiers `.env` ni de caches locaux.

## Licence

MIT ; voir `LICENSE`.
