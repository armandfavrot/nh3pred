# Validation de la migration vers 0.2.0

Validation locale effectuée le 19 octobre 2026, avant publication.

## Référence

Les résultats ont été calculés avec le code et les artefacts de
`deep-learning-for-ammonia-emissions/scripts/package/` avant migration.
Le code de calcul a été transféré ; seuls les imports, l'emplacement des
ressources et des ajustements de style ont changé. Les poids et les quatre
fichiers de paramètres sont identiques, avec leurs empreintes dans
`src/nh3pred/data/model_manifest.json`.

## Comparaison des prédictions

| Cas | Points | Écart absolu maximal sur les cumuls internes |
|---|---:|---:|
| Entrées minimales et météo par défaut | 29 | 0 |
| Trois applications et traitements distincts | 87 | 0 |
| Incorporation et météo personnalisée | 10 | 0 |
| Quatre essais de la thèse | 51 | 0 |
| Ces essais prolongés à 500 h | 185 | 0 |

Les sorties publiques, après arrondi, concordent également. Les comparaisons
internes acceptent `rtol=1e-6`, `atol=1e-7` ; les écarts mesurés ici sont nuls.
Les 126 points synthétiques sont conservés dans les tests distribués.
Les données expérimentales des deux derniers cas restent dans le dépôt de
recherche et ne figurent pas dans les fixtures publiées.

## Contrôles

- 39 tests réussis avec le wheel installé et exécutés depuis `/tmp`, sans
  import depuis `src/` ; les tests incluent les valeurs par défaut, le batching,
  les entrées catégorielles, les variables d'incorporation et les ressources.
- Comparaison complémentaire des cinq cas : 6 tests réussis, dont le contrôle
  d'intégrité des ressources. Ces cas recouvrent une partie des 39 tests.
- Distribution source et wheel construits ; contrôles `twine check` réussis.
- Wheel inspecté : pas de données d'entraînement, de tests, de documentation
  ni de code d'entraînement ; environ 1,2 Mo.
- Documentation Sphinx construite avec les avertissements traités comme erreurs.
- Contrôle Ruff réussi.

Environnement : Linux, Python 3.12.3, PyTorch 2.6.0+cu124 utilisé sur CPU,
pandas 3.0.3, NumPy 2.4.6. L'environnement temporaire possède sa propre
installation du wheel et réutilise les bibliothèques scientifiques locales.
Il ne constitue donc pas un test de résolution des dépendances depuis zéro.
Les autres systèmes et versions seront vérifiés par la CI configurée.

Un test existant attendait `till="unknown"` par défaut, contrairement aux
paramètres du modèle livré (`"no"`). Le test a été corrigé ; le défaut et
les prédictions du modèle n'ont pas été modifiés.

## Validation complémentaire en environnements isolés (20 octobre 2026)

Deux environnements Python 3.12.3 ont été créés avec
`include-system-site-packages = false`, sans réutiliser les bibliothèques
scientifiques du dépôt de recherche. Installation du wheel non éditable :

| Environnement | NumPy | pandas | PyTorch | Résultat |
|---|---|---|---|---|
| Résolution PyPI sans contraintes supplémentaires | 2.5.4 | 3.0.6 | 2.14.1 | 39 tests réussis |
| Versions basses vérifiées, PyTorch CPU | 1.26.4 | 3.0.0 | 2.5.0+cpu | 39 tests réussis |

`pip check` ne signale aucun conflit dans les deux environnements.
Les tests incluent les prédictions de référence et l'intégrité des artefacts.
Ces vérifications concernent Linux et Python 3.12.3 ; elles ne remplacent
pas la matrice CI Windows/Linux et Python 3.12/3.13.

Le premier essai avec pandas 2.2.3 a échoué au chargement de
`default_values.pkl` (`StringDtype`). La dépendance minimale a donc été
relevée à `pandas>=3.0.0`. Les poids, paramètres sérialisés et calculs
du modèle n'ont pas été modifiés.

Les quatre vignettes disposent maintenant de sources Quarto dans
`docs/vignettes/` et d'un lanceur `docs/render_vignettes.py`. Les exemples
utilisent `from nh3pred import predict` et des données synthétiques, sans
accès au dépôt de recherche. Les catégories publiques, les défauts,
les arrondis et la gestion des colonnes météo sont documentés pour 0.2.0.
Les statistiques de population avaient été retirées de la référence des
entrées à cette étape. Rectification après vérification du découpage :
les 2 128 applications et 51 499 observations d'origine correspondent bien
au sous-ensemble d'entraînement ; elles n'étaient pas incohérentes.

Les 19 cellules Python ont été exécutées lors du rendu HTML, puis de nouveau
pour le PDF combiné (25 pages), avec l'environnement récent. La construction
Sphinx avec `-W` et les contrôles `twine check` réussissent. L'archive source
inclut les quatre `.qmd`, le lanceur et les illustrations ; le wheel contient
uniquement le code d'inférence, les ressources du modèle et ses métadonnées.

Les graphiques d'interprétation restent des résultats précalculés :
leur empreinte a été vérifiée contre les sources de recherche. Ils portent
sur les 2 216 applications préparées, validation comprise, et ne constituent
pas une évaluation indépendante. Seuls les graphiques et leurs métadonnées
agrégées sont distribués ; leur recalcul nécessite le dépôt de recherche.

Aucun envoi sur PyPI ou TestPyPI n'a été effectué.

## Restauration des vignettes d'origine — étape 1

À la demande de l'auteur, `Getting started` et `Input reference` ont ensuite
été restaurées depuis `06_vignette/`, avec uniquement l'adaptation de l'import
dans `Getting started`. Les tableaux et textes d'origine sont conservés.
Les écarts avec l'API actuelle sont recensés, sans correction de fond,
dans `VIGNETTE_REVIEW.md`, pour décision de l'auteur avant publication.
Les rendus HTML et le PDF combiné ont été régénérés avec succès avec le
package installé ; les deux autres sources de vignettes n'ont pas été modifiées.
Les distributions de l'essai TestPyPI antérieur n'ont pas été reconstruites.

## Restauration des vignettes d'origine — étape 2

`Scenario comparisons` a été restaurée avec le seul changement d'import.
`Model architecture and interpretation` a été restaurée en conservant le
texte et les équations ; seules les lectures de ressources et de résultats
précalculés ainsi que les références de format des images sont adaptées
à un rendu indépendant. Les quatre figures PDF et les métadonnées sont
identiques aux résultats de recherche, dont l'empreinte a été vérifiée.
Les points de fond supplémentaires restent dans `VIGNETTE_REVIEW.md`,
sans modification des textes ou des valeurs des scénarios.
Le rendu complet a réussi avec le package installé dans l'environnement
isolé : exécution des exemples, HTML et PDF combiné. Les deux premières
sources restaurées à l'étape 1 restent inchangées.
