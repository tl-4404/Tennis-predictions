# Pre-match tennis training data

`build_dataset.py` downloads historical singles results and builds one row per completed match.
It uses Python 3.10+ and the standard library; no packages are needed to build the CSVs.

```sh
python3 build_dataset.py --download
python3 -m unittest -v
```

Defaults: ATP 2010–2024, with 2010–2011 available as warm-up history and output beginning in
2012. Both players must have at least five eligible historical matches. Cached source CSVs
are reused. To rebuild without network access, omit `--download`.

## Output

Files in `Data/processed/` have identical row order:

| File | Purpose |
| --- | --- |
| `X.csv` | Numeric model inputs only, with categorical fields already one-hot encoded |
| `y.csv` | Outcomes for classification or regression |
| `metadata.csv` | Match IDs, player names, tournament date, split, cutoff and original score; not inputs |
| `training.csv` | All three tables combined for inspection; do not train on all its columns |
| `manifest.json` | Feature lists, settings, source checksums, row counts and missing-value counts |

Targets:

- `a_wins`: binary winner classification, appropriate for logistic regression, SVC or an NN classifier.
- `total_games`: numeric regression target, appropriate for linear regression, SVR or an NN regressor.
- `game_margin_a`: Player A's games minus Player B's games; can be negative even when A wins.
- `sets_a`, `sets_b`: sets won by each player.
- `set_score_class`: exact set outcome from A's perspective, e.g. `2-1` or `0-2`.

Original score is stored from the **winner's** perspective; numeric targets and set-score class
are from **Player A's** perspective. A/B assignment is reproducible and independent of the outcome.
Player IDs and names are metadata, not numeric model inputs.

## Features

Each player has ranking, ranking points, log ranking, age, height, seed, hand and entry category.
Match context includes surface, tournament level, round, best-of format and draw size.
Historical features include:

- Career, previous-20-match, previous-365-day, surface and head-to-head win rates and match counts.
- Previous-20-match ace rate, double-fault rate, first-serve-in rate, first- and second-serve
  points-won rates, service and return points-won rates, break-point save rate, hold rate and break rate.
- The same serving/return rates for same-surface matches **within the previous 20 overall matches**.
- Opportunity counts for each rate and counts of matches with serve data, so sparse histories are visible.
- A-minus-B differences for numeric player features, alongside each player's own values.

Ratios use pooled counts, not an unweighted mean of percentages. Double-fault rate here is
double faults per **service point**. Second-serve denominator is service points minus first serves in,
including double faults. Hold and break rates are derived from service games and break points converted.
Invalid/missing paired counts are excluded metric by metric. Empty cells mean missing or undefined,
not zero. No injury, exact rest-days, travel or weather features are inferred from unavailable data.
No opponent-adjusted ratings or Elo are included in this version.

## Time and leakage handling

The source's `tourney_date` is generally the tournament's start date, not the actual match date.
`match_num` is not guaranteed to be chronological. Consequently:

1. For a tournament starting on date D, the default history only includes matches in tournaments
   whose start is **strictly before D minus 28 days**.
2. Current tournament results never enter that tournament's features. Results of more recent
   tournaments are intentionally withheld, even if they have already finished.
3. The 28-day buffer is a conservative approximation for ordinary tour events, not a substitute for
   verified match/end timestamps. It cannot guarantee chronology for unusually long events or bad dates.
4. Within a historical tournament all its matches count, but their actual order is unknown.
   Date ties use deterministic match-ID ordering when selecting a last-20 history; this is approximate.
5. The 365-day window is measured from the target tournament start and also respects the history cutoff.
6. Pre-tournament ranking, age and other supplied player fields come from the current source row.
   Current match score, duration and performance statistics are used only as targets or in later histories.

The default split is training 2012–2021, validation 2022, test 2023–2024. Keep every tournament in
one split. This is a **rolling historical evaluation**: validation/test features can include earlier
results from those periods once the history cutoff allows them, as they would become available in use.
For cross-validation, use chronological folds rather than random shuffling.

Retirements, walkovers, defaults, incomplete/unparseable scores, and bracketed match tiebreaks
are excluded from both history and targets. This means the winner dataset describes **completed
conventional singles**, not retirement risk. Standard set tiebreaks count as games (7–6), while the
tiebreak points in parentheses do not. Missing stats remain missing, with no global imputation.

## Load for the three classifiers

This example requires pandas and scikit-learn, separately from the dependency-free parser:

```python
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, balanced_accuracy_score

X = pd.read_csv("Data/processed/X.csv")
y = pd.read_csv("Data/processed/y.csv")["a_wins"]
meta = pd.read_csv("Data/processed/metadata.csv")
train = meta["split"].eq("train")
validation = meta["split"].eq("validation")

models = {
    "logistic": LogisticRegression(max_iter=2000),
    "svm": SVC(kernel="rbf"),
    "nn": MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=300, random_state=42),
}
for name, model in models.items():
    pipeline = make_pipeline(
        SimpleImputer(strategy="median", add_indicator=True, keep_empty_features=True),
        StandardScaler(),
        model,
    )
    pipeline.fit(X.loc[train], y.loc[train])
    prediction = pipeline.predict(X.loc[validation])
    print(name, accuracy_score(y.loc[validation], prediction),
          balanced_accuracy_score(y.loc[validation], prediction))
```

Tune on validation only and reserve the test split for the final comparison. Fit preprocessing
on training data only. The example does not claim any achieved model performance.

## Options

```sh
# Alternate years and output directory (start early enough to supply warm-up history):
python3 build_dataset.py --download --start-year 2005 --end-year 2024 --output-start-year 2010 --output-dir Data/long_history

# Change the history window or retain players with no recorded history:
python3 build_dataset.py --window 30 --min-history 0

# Women's tour, kept separate from ATP:
python3 build_dataset.py --download --tour wta --raw-dir Data/wta_raw --output-dir Data/wta_processed
```

`--history-lag-days` changes the buffer; lowering it can admit matches not yet completed at
prediction time. `--train-end-year`, `--validation-end-year` and `--seed` control splits and A/B assignment.
For missing input years, download fails explicitly instead of silently constructing a partial history.
Duplicate match IDs with conflicting content fail explicitly.

## Data source and license

Data compiled by **Jeff Sackmann / Tennis Abstract**, retrieved from the
[Aneeshers archival mirror](https://github.com/Aneeshers/tennis-sackmann-archive).
[Column definitions](https://github.com/Aneeshers/tennis-sackmann-archive/blob/main/atp/matches_data_dictionary.txt).
Source and derived data are subject to **CC BY-NC-SA 4.0**: attribution, noncommercial use and
share-alike requirements. See the [source license](https://github.com/Aneeshers/tennis-sackmann-archive/blob/main/LICENSE).
Generated CSVs are ignored by Git to avoid committing large data files.
