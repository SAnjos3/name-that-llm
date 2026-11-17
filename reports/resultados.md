# Registro de experimentos

Validação: `GroupKFold(n_splits=5)` por `prompt_id`. Métrica: balanced accuracy. Chute aleatório ≈ 5,6%.

## Experimentos manuais (notebook 02, seção 5)

| Data | Experimento | Média | Desvio |
|---|---|---|---|
| 2026-11-16 | TF-IDF palavras + LogisticRegression | 26,86% | 0,89% |
| 2026-11-16 | TF-IDF caracteres `char_wb` (2–5), sublinear + LogisticRegression | 29,63% | 0,53% |
| 2026-11-16 | Palavras + caracteres + LogisticRegression | 30,89% | 1,33% |
| 2026-11-16 | Palavras + caracteres + LinearSVC | 30,15% | 2,20% |
| 2026-11-16 | Palavras + caracteres + RidgeClassifier | 29,74% | 1,73% |
| 2026-11-16 | Palavras + caracteres + SGDClassifier | 28,15% | 1,76% |
| 2026-11-16 | Palavras + caracteres + ComplementNB | 24,56% | 1,31% |

## GridSearch (notebook 02, seção 6) — 2026-11-16

128 combinações × 5 folds = 640 treinos, 45,7 min em CPU local (`n_jobs=4`).

Grade: vetorizador de caracteres (`analyzer` ∈ {`char_wb`, `char`} × `lowercase` ∈ {True, False}) × modelo, função de custo e penalização:
LogisticRegression (`C` ∈ {1, 10}) · LinearSVC (`loss` ∈ {hinge, squared_hinge}, `C` ∈ {0.1, 0.5, 1}) · SGDClassifier (`loss` ∈ {hinge, log_loss, modified_huber}, `penalty` ∈ {l2, l1, elasticnet}, `alpha` ∈ {1e-5, 1e-4}) · RidgeClassifier (`alpha` ∈ {0.3, 1, 3}) · ComplementNB (`alpha` ∈ {0.1, 0.3, 1}).

| Modelo | Melhor configuração | Média | Desvio |
|---|---|---|---|
| **LogisticRegression** | `C=10`, `char`, `lowercase=False` | **32,52%** | 1,86% |
| SGDClassifier | `log_loss`, `l2`, `alpha=1e-4`, `char`, `lowercase=False` | 31,71% | 1,88% |
| LinearSVC | `squared_hinge`, `C=0.5`, `char`, `lowercase=False` | 31,23% | 1,37% |
| RidgeClassifier | `alpha=1`, `char`, `lowercase=False` | 31,01% | 1,16% |
| ComplementNB | `alpha=1`, `char`, `lowercase=False` | 25,82% | 1,89% |

Holdout 30% por prompt com a melhor configuração: **30,00%**; 55% dos erros dentro da mesma família de LLM (≈18% esperado ao acaso).

| Leaderboard público | — (submissão pendente) |
|---|---|
