# Tabular Classification

Predicting a discrete label from a row of a table: why gradient-boosted trees held the crown for a decade, what tabular foundation models changed in 2025-2026, the metrics that survive class imbalance, and runnable code that scores five models on one Adult census split.

Author

Benedict Thekkel

## 1. What is Tabular Classification?

Tabular classification maps **one row of a table** to **one of K discrete classes**. It is the least glamorous and most deployed task in machine learning: the row is a customer, a transaction, a patient, or a machine, and the label is churn / fraud / diagnosis / failure.

**Input.** A fixed set of columns, and the awkward part is that they are not one type:

- **Numeric** columns on wildly different scales (age in years, income in dollars).
- **Categorical** columns, some low-cardinality (`sex`), some high (`zip code`, `merchant_id`).
- **Ordinal** columns where the order is meaningful but the spacing is not (`education`).
- **Missing values** that are often informative - the fact that `income` is blank is itself a feature, so imputing it away destroys signal.

There is no spatial or temporal structure to exploit, which is exactly why convolutions and attention have so little to offer here: **permuting the columns changes nothing about the problem**, and a model that has to learn that invariance from data starts behind a tree that never assumed otherwise.

**Output.** A class label, but in practice you want the **probability vector** and then choose a threshold. Almost every real tabular deployment is a decision under asymmetric cost (a missed fraud costs more than a false alarm), so a model that ranks well but is badly calibrated is harder to use than a slightly worse model that is honest about its uncertainty.

**Neighbouring tasks:**

| Task | What it does | Typical tool |
|----|----|----|
| Tabular regression | Continuous target instead of a class | see `01_Tabular_Regression` |
| Time series forecasting | Rows are ordered and the future is the target | see `02_Time_Series_Forecasting` |
| Anomaly detection | No labels; find rows unlike the rest | Isolation Forest, LOF |
| Learning to rank | Order rows within a group | LambdaMART (XGBoost `rank:` objectives) |
| Uplift / causal | Effect of an intervention, not the outcome | EconML, CausalML |
| Survival analysis | Time until an event, with censoring | scikit-survival, XGBoost AFT |

------------------------------------------------------------------------

## 2. Real-World Use Cases

Tabular classification is the workhorse behind decisions that carry money and liability, which is why the constraints here are so rarely “accuracy”.

| Use case | Domain | Consumes / produces | Dominant constraint |
|----|----|----|----|
| Credit scoring and loan approval | Banking (FICO, Experian, every neobank) | Applicant features, bureau history -\> default probability | Regulatory explainability (adverse action notices), fairness audits, stability over years |
| Card fraud detection | Payments (Stripe Radar, Visa, Adyen) | Transaction + device + velocity features -\> fraud score | Sub-100 ms latency inline with authorisation; extreme imbalance (~0.1% positive); adversarial drift |
| Churn prediction | Telecom, SaaS | Usage, billing, support history -\> churn in next 90 days | Calibration (retention budget is finite); the ranking matters more than the label |
| Clinical risk scoring | Healthcare (sepsis, readmission, deterioration) | EHR vitals + labs + demographics -\> risk | Calibration and alarm fatigue; prospective validation; distribution shift between hospitals |
| Ad click-through prediction | Adtech | User + context + creative features -\> P(click) | Billions of rows, high-cardinality IDs, hourly retraining, cost per prediction |
| Predictive maintenance | Manufacturing, energy | Sensor aggregates + maintenance logs -\> failure in next N days | Very few positives; cost of a false negative is a plant outage |
| Insurance underwriting and claims triage | Insurance | Policy + claim features -\> fraud/severity class | Legal defensibility; monotonic constraints on price-relevant features |
| Lead scoring and propensity | B2B sales, marketing | CRM firmographics + behaviour -\> convert / not | Cheap to be wrong; retraining cadence and leakage discipline matter more than model class |

**What the benchmark number hides.** A tabular leaderboard reports accuracy or AUC on a random split of a static table, and almost every way a real project fails is invisible in that number.

- **Leakage is the number one cause of a great offline score and a useless model.** A column computed after the label event (a `closed_reason`, a `refund_flag`, an aggregate over the full history) leaks the answer. A random split leaks too when the rows are not independent: multiple rows per customer, or a table that spans time. Split by **entity and by time**, not at random.
- **Drift is the norm, not the exception.** Fraud is adversarial and moves in weeks; credit behaviour moves with the economy; a sensor gets recalibrated and its column shifts. Production tabular systems retrain on a schedule and monitor input distributions, not just output metrics.
- **The threshold is the model.** AUC is threshold-free, but a deployment picks one operating point, and that choice (driven by the cost ratio and the capacity of whatever acts on the alert) usually matters more than the last two points of AUC.
- **Explainability is a hard requirement in regulated domains.** Credit and insurance need a per-decision reason. That pushes towards monotonic-constrained GBDTs and scorecards, and it is why logistic regression is still shipping in 2026.
- **The cheap columns are the model.** Feature engineering, joins, and aggregation windows move metrics far more than swapping XGBoost for CatBoost. On the modelling side the honest advice is: pick a solid default, spend the time on the data.

------------------------------------------------------------------------

## 3. How Modern Tabular Classification Works

Five eras, and unusually for deep learning, the newest one did **not** simply replace the old.

**1. Linear and generalised linear models (pre-2000, still deployed).** Logistic regression on hand-built features, often binned into a scorecard. Weak on interactions, but monotone, auditable, and trivially fast. Still the default in credit and insurance because a regulator can read it.

**2. Bagged trees (Random Forest, 2001).** Many decorrelated deep trees, averaged. Handles mixed types, non-linearity and interactions with no tuning at all. Still the best “I have not looked at this data yet” baseline.

**3. Gradient boosting (2001 -\> 2018), the long-reigning champion.** Fit trees sequentially to the residual. The engineering wave made it dominant: **XGBoost** (2016, sparsity-aware split finding, regularisation), **LightGBM** (2017, histogram binning + leaf-wise growth, ~10x faster on wide data), **CatBoost** (2018, ordered target statistics for high-cardinality categoricals). scikit-learn’s `HistGradientBoosting*` is a LightGBM-style implementation in the standard library. For tables above ~10k rows this is still the accuracy/effort optimum in 2026.

**4. Deep tabular networks (2019-2023), mostly a dead end.** TabNet (2019), NODE, FT-Transformer (2021), SAINT and a long tail of variants applied attention to rows. Two systematic studies settled it: Shwartz-Ziv and Armon (2021) and **Grinsztajn et al. (2022)**, which showed GBDTs still win on medium-sized tabular data and explained *why* - neural nets are biased towards smooth functions, are hurt by uninformative features, and are not rotation-invariant in the way tabular data needs. Deep tabular models earn their place mainly when the table is **one modality among several** (a row plus an image plus free text), where the whole thing has to be differentiable end to end.

**5. Tabular foundation models and in-context learning (2022 -\> now), the real change.** **TabPFN** treats classification as a single forward pass: a transformer pretrained on millions of *synthetic* tabular tasks takes the whole training set as context and predicts the test rows with **no gradient steps at all**. TabPFN v2 (Hollmann et al., *Nature*, January 2025) extended it to ~10k rows and 500 features, mixed types and missing values, and beat a heavily tuned ensemble of GBDTs on small tables while fitting in seconds. Follow-ons - **TabICL** (2025, scaling in-context learning to ~100k rows), **TabDPT**, **CARTE** (transfers across tables with different schemas), and LLM-based **TabuLa-8B** (zero-shot from column *names*) - are pushing the row limit outward.

**Where that leaves you in 2026:**

| Situation | Pick |
|----|----|
| Under ~10k rows, tabular only | TabPFN v2 (or an ensemble of it with a GBDT) |
| 10k to hundreds of millions of rows | LightGBM / XGBoost / CatBoost |
| Regulated, needs per-decision reasons | Monotonic-constrained GBDT, or logistic regression |
| High-cardinality categoricals everywhere | CatBoost, or target/hash encoding + LightGBM |
| Table plus text plus images in one model | FT-Transformer style deep model, or fuse embeddings into a GBDT |
| “Just make it work, no tuning budget” | AutoGluon (stacked ensemble), or sklearn `HistGradientBoosting` |

------------------------------------------------------------------------

## 4. Evaluation Metrics

**Accuracy is the wrong default.** On a fraud table with 0.2% positives, predicting “never fraud” scores 99.8%. Two families of metric replace it: **ranking** metrics that ignore the threshold, and **probabilistic** metrics that check whether the score means anything.

Given TP / FP / FN / TN at a chosen threshold:

$$\text{precision} = \frac{TP}{TP + FP}, \qquad
  \text{recall} = \frac{TP}{TP + FN}, \qquad
  F_1 = \frac{2 \cdot P \cdot R}{P + R}$$

- **ROC-AUC** - probability that a random positive scores above a random negative. Threshold-free and prevalence-independent, which is also its weakness: with 0.2% positives an ROC-AUC of 0.98 can still mean most of your alerts are false.
- **PR-AUC (average precision)** - area under precision vs recall. **This is the metric for rare positives**, because its baseline is the positive rate, not 0.5, so it moves when the useful part of the ranking moves.
- **Log loss** $-\frac{1}{N}\sum [y\log \hat p + (1-y)\log(1-\hat p)]$ - punishes confident mistakes without bound. The standard training objective.
- **Brier score** $\frac{1}{N}\sum (\hat p - y)^2$ - mean squared error on probabilities. Bounded, and decomposes into calibration plus refinement.
- **Calibration** - of the rows scored 0.30, do 30% turn out positive? A model can rank perfectly (AUC 1.0) and be wildly miscalibrated. Measure with a reliability curve and **expected calibration error (ECE)**; fix with Platt scaling or isotonic regression on a held-out slice.
- **Expected cost** - the only metric that maps to the business: $\text{cost} = c_{FP} \cdot FP + c_{FN} \cdot FN$. Sweep the threshold, pick the minimum.

**Pitfalls.** Compare models on the *same* split with the *same* preprocessing, stratify the split when classes are imbalanced, and use time-based splits when the table has a time column. Never tune on the test set: fit and tune on train/validation (or cross-validation), and touch the test split once.

------------------------------------------------------------------------

``` python
import numpy as np
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    f1_score,
    log_loss,
    roc_auc_score,
)

# A toy imbalanced problem: 2% positives, and a ranker that is good but not perfect.
rng = np.random.default_rng(0)
n = 5000
y = (rng.random(n) < 0.02).astype(int)
score = np.clip(rng.normal(0.15, 0.10, n) + 0.45 * y, 1e-6, 1 - 1e-6)

always_negative = np.zeros(n)
print(f"accuracy of 'always negative'   {(always_negative == y).mean():.4f}   <- useless model")
print(f"ROC-AUC                         {roc_auc_score(y, score):.4f}")
print(f"PR-AUC  (baseline = {y.mean():.3f})     {average_precision_score(y, score):.4f}")
print(f"log loss                        {log_loss(y, score):.4f}")
print(f"Brier                           {brier_score_loss(y, score):.4f}")

# The threshold is a separate decision. Sweep it under an asymmetric cost.
C_FP, C_FN = 1.0, 20.0  # a missed positive costs 20 false alarms
grid = np.linspace(0.01, 0.99, 99)
costs = [(t, C_FP * ((score >= t) & (y == 0)).sum() + C_FN * ((score < t) & (y == 1)).sum())
         for t in grid]
best_t, best_cost = min(costs, key=lambda x: x[1])
cost_at_half = C_FP * ((score >= 0.5) & (y == 0)).sum() + C_FN * ((score < 0.5) & (y == 1)).sum()
print(f"\nbest threshold {best_t:.2f} at cost {best_cost:.0f} "
      f"(F1 there = {f1_score(y, (score >= best_t).astype(int)):.3f})")
print(f"the default threshold 0.50 costs {cost_at_half:.0f} - "
      f"{cost_at_half / best_cost:.1f}x more, on the same model")
```

    accuracy of 'always negative'   0.9788   <- useless model
    ROC-AUC                         0.9994
    PR-AUC  (baseline = 0.021)     0.9858
    log loss                        0.1823
    Brier                           0.0358

    best threshold 0.40 at cost 46 (F1 there = 0.886)
    the default threshold 0.50 costs 340 - 7.4x more, on the same model

## 5. Datasets

Tabular benchmarking has a reproducibility problem: for years every paper picked its own handful of UCI tables, so the curated **suites** in the last three rows matter more than any single dataset.

| Dataset | Contents | Size | Scope | License | Typical use |
|----|----|----|----|----|----|
| [Adult / Census Income](https://www.openml.org/d/1590) | 1994 US census records, 6 numeric + 8 categorical columns, label = income \> \$50k | 48,842 rows | US census, binary | CC BY 4.0 (UCI) | The canonical demo table - **used by this notebook** |
| [Covertype](https://www.openml.org/d/1596) | Cartographic features -\> forest cover type | 581,012 rows, 7 classes | US forestry | CC BY 4.0 | Large multiclass, tests scaling |
| [Credit Card Fraud](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) | Anonymised PCA components of card transactions | 284,807 rows, 0.17% positive | Payments | DbCL 1.0 | The standard extreme-imbalance table |
| [Bank Marketing](https://www.openml.org/d/1461) | Call-centre campaign outcomes | 45,211 rows | Portuguese bank | CC BY 4.0 | Imbalanced, many categoricals |
| [Higgs](https://www.openml.org/d/23512) | Simulated particle collisions | 11M rows, 28 numeric | Physics | CC BY 4.0 | Throughput benchmark for GBDTs |
| [Home Credit Default Risk](https://www.kaggle.com/competitions/home-credit-default-risk) | Multi-table loan application data | 300k applications + relational tables | Consumer credit | Competition terms | Realistic feature engineering / joins |
| [OpenML-CC18](https://www.openml.org/s/99) | Curated suite of 72 classification tasks | 500 to 100k rows each | Mixed | CC BY 4.0 | The standard “many tables” benchmark |
| [TabArena](https://tabarena.ai) | Living benchmark, curated tables + leaderboard | 51 tables (2025 release) | Mixed | Per-dataset | Current model-vs-model comparisons |
| [TALENT](https://github.com/LAMDA-Tabular/TALENT) | 300+ datasets, 30+ tabular methods, unified harness | Mixed | Mixed | Per-dataset | Large-scale method comparison |

**This notebook uses Adult** (OpenML id 1590, `version=2`), fetched through `sklearn.datasets.fetch_openml` with `data_home` pointed at `DL_tasks/datasets/`, which is gitignored. It is small (about 5 MB), has genuinely mixed column types, real missing values, and a 24% positive rate - imbalanced enough that accuracy misleads, balanced enough that a demo finishes in seconds. None of the datasets above are gated; the two Kaggle ones need a Kaggle account to download.

------------------------------------------------------------------------

## 6. The Model Landscape (mid-2026)

There is no single “SOTA tabular classifier” - the winner depends on the row count, and the crossover sits around 10k rows.

| Model | Params / size | License | Handles categoricals | Handles missing | Best for |
|----|----|----|----|----|----|
| Logistic regression | O(features) | BSD (sklearn) | via encoding | no (impute) | Auditability, a floor to beat |
| Random Forest | ~100s of trees | BSD (sklearn) | via encoding | no (impute) | Zero-tuning baseline |
| sklearn `HistGradientBoosting` | ~100s of trees | BSD | **native** | **native** | Strong default with no extra dependency |
| **LightGBM** | ~100s-1000s of trees | MIT | native | native | Speed on wide/large tables; the usual production pick |
| **XGBoost** | ~100s-1000s of trees | Apache 2.0 | native (1.5+) | native | Best-tuned accuracy, GPU training, monotone constraints |
| **CatBoost** | ~1000s of trees | Apache 2.0 | native, ordered TS | native | High-cardinality categoricals, least tuning |
| FT-Transformer | ~1-10M | MIT | embeddings | needs imputation | Multimodal fusion, differentiable end to end |
| **TabPFN v2** | 11M (prior-fitted transformer) | Custom (research-friendly) | native | **native** | **\<= 10k rows, ~500 features: current small-data SOTA, no training** |
| TabICL | ~100M | Apache 2.0 | native | native | In-context learning up to ~100k rows |
| AutoGluon (Tabular) | ensemble | Apache 2.0 | native | native | “Best number without thinking”: stacks GBDTs + NNs + TabPFN |

Leaderboards worth trusting: **[TabArena](https://tabarena.ai)** (living, well-controlled), the [AutoML Benchmark](https://openml.github.io/automlbenchmark/), and [TALENT](https://github.com/LAMDA-Tabular/TALENT). Be sceptical of any tabular claim measured on fewer than ~20 datasets: variance between tables dwarfs the difference between good models.

**What wins what.** On accuracy, TabPFN v2 leads below ~10k rows and tuned GBDT ensembles lead above it, with AutoGluon-style stacking on top of both. On speed, LightGBM trains fastest per unit of accuracy and TabPFN v2 has no training at all but pays it back at inference (it carries the training set in its context). On size and auditability, logistic regression is still unbeaten, which is exactly why credit and insurance still ship it.

**What runs below.** Everything down to the CatBoost row is a repo dependency, so the head-to-head in section 13 is a real seven-model comparison rather than a sklearn-only stand-in: logistic regression, random forest, sklearn `HistGradientBoosting`, an MLP, and all three production boosters. TabPFN v2 and TabICL are the notable absentees - they are vendor packages rather than general-purpose libraries (see the rule in `CLAUDE.md`), and section 16 says how to try them.

------------------------------------------------------------------------

## 7. Setup

Package roles:

- `scikit-learn` - the linear, forest, boosting and MLP baselines, the preprocessing pipeline, and every metric
- `xgboost` / `lightgbm` / `catboost` - the three production gradient boosters (section 12)
- `pandas` - the table itself
- `pyecharts` - all charts (repo rule: ECharts for graphing)
- `numpy` - array work
- `torch` - only for the `vram()` / `free_memory()` helpers shared with the rest of `DL_tasks`; no model here touches the GPU. Tabular models are CPU-bound, so **RAM is the budget that matters**, not VRAM.

The Adult download (about 5 MB) lands in `DL_tasks/datasets/sklearn/`, which is gitignored.

------------------------------------------------------------------------

``` python
# All of these are repo dependencies (see the root pyproject.toml):
# %pip install -q scikit-learn pandas pyecharts xgboost lightgbm catboost
```

``` python
import ctypes
import ctypes.util
import gc
import time
from pathlib import Path

import numpy as np
import pandas as pd
import psutil
import torch
from dotenv import find_dotenv, load_dotenv

# Knowledge/.env sets HF_TOKEN - authenticated Hub requests get higher rate limits
load_dotenv(find_dotenv(usecwd=True))

device = "cuda:0" if torch.cuda.is_available() else "cpu"
print("device:", device, "(sklearn models are CPU-only; RAM is the budget here)")


def vram(tag=""):
    "Report current GPU memory (allocated / reserved). No-op on CPU."
    if torch.cuda.is_available():
        alloc = torch.cuda.memory_allocated() / 1e9
        reserved = torch.cuda.memory_reserved() / 1e9
        print(f"VRAM {tag:16s} {alloc:5.2f} GB allocated / {reserved:5.2f} GB reserved")


def free_memory(*objs):
    "Delete objects, run GC, empty the CUDA cache, and return freed RAM to the OS.\n\n    Note the argument list only drops *this function's* references. A name bound in\n    the notebook still holds the object, so the working idiom is `del model;\n    free_memory()` at the call site - which is what every section below uses.\n    "
    for o in objs:
        del o
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
    # glibc keeps freed CPU allocations in its arenas instead of handing them back,
    # so RSS ratchets upward across sections. malloc_trim(0) returns them. On a
    # 20 GB box this is not optional - see dl-visualization-and-memory.md.
    try:
        ctypes.CDLL(ctypes.util.find_library("c") or "libc.so.6").malloc_trim(0)
    except Exception:
        pass


def memory_report(tag=""):
    "Print current system RAM (and VRAM if a GPU is present), in GB."
    vm = psutil.virtual_memory()
    print(f"RAM  {tag:16s} {(vm.total - vm.available) / 1e9:5.2f} / {vm.total / 1e9:5.2f} GB")
    vram(tag)


# All downloads go to DL_tasks/datasets/ (gitignored)
DATA_DIR = Path("../../datasets")
DATA_DIR.mkdir(exist_ok=True)
SKLEARN_HOME = str(DATA_DIR / "sklearn")

memory_report("baseline")
```

    device: cuda:0 (sklearn models are CPU-only; RAM is the budget here)
    RAM  baseline         16.75 / 20.97 GB
    VRAM baseline          0.00 GB allocated /  0.00 GB reserved

``` python
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split

# Adult / Census Income, OpenML id 1590. version=2 is the cleaned copy with real
# NaNs rather than the "?" string. ~5 MB, cached under DL_tasks/datasets/sklearn.
adult = fetch_openml("adult", version=2, as_frame=True, data_home=SKLEARN_HOME)
X_all, y_all = adult.data, (adult.target == ">50K").astype(int)

# Drop fnlwgt: it is a census *sampling weight*, not a property of the person. Leaving
# it in is a small, real example of the leakage-shaped mistakes tabular tables invite.
X_all = X_all.drop(columns=["fnlwgt"])

NUMERIC = X_all.select_dtypes(include="number").columns.tolist()
CATEGORICAL = [c for c in X_all.columns if c not in NUMERIC]

print(f"{len(X_all):,} rows x {X_all.shape[1]} columns   positive rate {y_all.mean():.3f}")
print(f"numeric    ({len(NUMERIC)}): {NUMERIC}")
print(f"categorical({len(CATEGORICAL)}): {CATEGORICAL}")
print("\nmissing values per column (top 5):")
print(X_all.isna().sum().sort_values(ascending=False).head(5).to_string())

# Stratified split so the 24% positive rate is preserved in both halves. A real
# deployment on a table with a time column would split by time instead.
X_tr, X_te, y_tr, y_te = train_test_split(
    X_all, y_all, test_size=0.25, stratify=y_all, random_state=0
)
print(f"\ntrain {X_tr.shape}  test {X_te.shape}  "
      f"positive rate {y_tr.mean():.3f} / {y_te.mean():.3f}")
X_all.head()
```

    48,842 rows x 13 columns   positive rate 0.239
    numeric    (5): ['age', 'education-num', 'capital-gain', 'capital-loss', 'hours-per-week']
    categorical(8): ['workclass', 'education', 'marital-status', 'occupation', 'relationship', 'race', 'sex', 'native-country']

    missing values per column (top 5):
    occupation        2809
    workclass         2799
    native-country     857
    age                  0
    education            0

    train (36631, 13)  test (12211, 13)  positive rate 0.239 / 0.239

|  | age | workclass | education | education-num | marital-status | occupation | relationship | race | sex | capital-gain | capital-loss | hours-per-week | native-country |
|----|----|----|----|----|----|----|----|----|----|----|----|----|----|
| 0 | 25 | Private | 11th | 7 | Never-married | Machine-op-inspct | Own-child | Black | Male | 0 | 0 | 40 | United-States |
| 1 | 38 | Private | HS-grad | 9 | Married-civ-spouse | Farming-fishing | Husband | White | Male | 0 | 0 | 50 | United-States |
| 2 | 28 | Local-gov | Assoc-acdm | 12 | Married-civ-spouse | Protective-serv | Husband | White | Male | 0 | 0 | 40 | United-States |
| 3 | 44 | Private | Some-college | 10 | Married-civ-spouse | Machine-op-inspct | Husband | Black | Male | 7688 | 0 | 40 | United-States |
| 4 | 18 | NaN | Some-college | 10 | Never-married | NaN | Own-child | White | Female | 0 | 0 | 30 | United-States |

``` python
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# Two preprocessing regimes, because trees and linear models want different things.
#
# For linear models and the MLP: impute, one-hot the categoricals, standardise the
# numerics (gradient-based fitting is scale-sensitive; trees are not).
preprocess_dense = ColumnTransformer([
    ("num", Pipeline([("impute", SimpleImputer(strategy="median")),
                      ("scale", StandardScaler())]), NUMERIC),
    ("cat", Pipeline([("impute", SimpleImputer(strategy="most_frequent")),
                      ("onehot", OneHotEncoder(handle_unknown="ignore",
                                               min_frequency=10))]), CATEGORICAL),
])

# For HistGradientBoosting: no encoding and no imputation at all. It splits on
# categories natively and routes NaN down whichever branch reduces loss, which is
# strictly better than pretending a missing value is the median.
CAT_MASK = [c in CATEGORICAL for c in X_all.columns]

# Random Forest cannot take NaN or strings, so it gets ordinal codes + imputation.
from sklearn.preprocessing import OrdinalEncoder

preprocess_ordinal = ColumnTransformer([
    ("num", SimpleImputer(strategy="median"), NUMERIC),
    ("cat", Pipeline([("impute", SimpleImputer(strategy="most_frequent")),
                      ("ord", OrdinalEncoder(handle_unknown="use_encoded_value",
                                             unknown_value=-1))]), CATEGORICAL),
])

n_dense = preprocess_dense.fit(X_tr).transform(X_tr[:5]).shape[1]
print(f"one-hot expands {X_tr.shape[1]} columns to {n_dense} features")
print(f"{sum(CAT_MASK)} columns flagged categorical for the native-categorical GBDT")
```

    one-hot expands 13 columns to 104 features
    8 columns flagged categorical for the native-categorical GBDT

## 8. Logistic Regression: the floor everything is measured against

A linear model on one-hot features. It cannot represent an interaction unless you build the interaction column yourself, so it should lose - but by how much is the single most useful number in a tabular project. If a boosted forest beats it by half a point of AUC, the extra complexity is not buying anything, and the auditable model wins.

The coefficients printed below are the reason this model is still shipping in regulated domains: each one is a log-odds contribution you can quote in an adverse-action notice. Note that they are only readable because the numeric columns were standardised first, so “per standard deviation” is a common unit; on raw columns the magnitudes compare income in dollars against age in years and mean nothing.

------------------------------------------------------------------------

``` python
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score

logreg = Pipeline([
    ("prep", preprocess_dense),
    ("clf", LogisticRegression(max_iter=2000, C=1.0)),
])

t0 = time.perf_counter()
logreg.fit(X_tr, y_tr)
fit_s = time.perf_counter() - t0

p = logreg.predict_proba(X_te)[:, 1]
print(f"logistic regression   fit {fit_s:5.1f}s   "
      f"ROC-AUC {roc_auc_score(y_te, p):.4f}   PR-AUC {average_precision_score(y_te, p):.4f}")

# The largest positive and negative coefficients, which is the whole point of this model.
names = logreg.named_steps["prep"].get_feature_names_out()
coefs = pd.Series(logreg.named_steps["clf"].coef_[0], index=names).sort_values()
print("\nstrongest negative / positive coefficients")
print(pd.concat([coefs.head(4), coefs.tail(4)]).to_string())

free_memory()
```

    logistic regression   fit   0.5s   ROC-AUC 0.9036   PR-AUC 0.7623

    strongest negative / positive coefficients
    cat__native-country_Columbia             -1.427429
    cat__occupation_Priv-house-serv          -1.387512
    cat__marital-status_Never-married        -1.176986
    cat__native-country_South                -1.131949
    cat__native-country_Ireland               1.183723
    cat__marital-status_Married-AF-spouse     1.208042
    cat__marital-status_Married-civ-spouse    1.590962
    num__capital-gain                         2.385514

## 9. Random Forest: the zero-tuning baseline

Hundreds of deep decorrelated trees, averaged. It needs no scaling, no distribution assumptions and essentially no tuning, which makes it the right first model on a table you have never seen. Its weakness is that averaging deep trees produces **over-smoothed, poorly calibrated probabilities** - the votes cluster away from 0 and 1 - so it usually ranks well and calibrates badly. Section 14 shows exactly that.

------------------------------------------------------------------------

``` python
from sklearn.ensemble import RandomForestClassifier

rf = Pipeline([
    ("prep", preprocess_ordinal),
    ("clf", RandomForestClassifier(n_estimators=300, min_samples_leaf=2,
                                   n_jobs=4, random_state=0)),  # 4 vCPU on this box
])

t0 = time.perf_counter()
rf.fit(X_tr, y_tr)
fit_s = time.perf_counter() - t0

p_rf = rf.predict_proba(X_te)[:, 1]
print(f"random forest         fit {fit_s:5.1f}s   "
      f"ROC-AUC {roc_auc_score(y_te, p_rf):.4f}   PR-AUC {average_precision_score(y_te, p_rf):.4f}")

memory_report("after RF")
```

    random forest         fit   2.1s   ROC-AUC 0.9157   PR-AUC 0.8069
    RAM  after RF         16.97 / 20.97 GB
    VRAM after RF          0.00 GB allocated /  0.00 GB reserved

## 10. HistGradientBoosting: the workhorse

scikit-learn’s histogram-based gradient booster, closely modelled on LightGBM: features are binned into 255 buckets once, then split finding is a histogram scan instead of a sort. Two things make it the right default here and not just a convenience:

- **`categorical_features` splits on categories directly**, with no one-hot blowup and no arbitrary ordinal ordering.
- **Missing values are learned, not imputed.** At each split NaN is routed to whichever side reduces the loss, so “this field was blank” stays a usable signal.

`early_stopping` holds out a validation slice internally and stops when it stops improving, so `max_iter=400` is a ceiling rather than a commitment.

------------------------------------------------------------------------

``` python
from sklearn.ensemble import HistGradientBoostingClassifier

hgb = HistGradientBoostingClassifier(
    max_iter=400,
    learning_rate=0.1,
    max_leaf_nodes=31,
    l2_regularization=1.0,
    categorical_features=CAT_MASK,   # native categorical splits, no one-hot
    early_stopping=True,
    validation_fraction=0.1,
    n_iter_no_change=20,
    random_state=0,
)

t0 = time.perf_counter()
hgb.fit(X_tr, y_tr)                  # raw frame: no imputation, no encoding
fit_s = time.perf_counter() - t0

p_hgb = hgb.predict_proba(X_te)[:, 1]
print(f"hist gradient boosting fit {fit_s:5.1f}s  "
      f"ROC-AUC {roc_auc_score(y_te, p_hgb):.4f}   PR-AUC {average_precision_score(y_te, p_hgb):.4f}")
print(f"early stopping used {hgb.n_iter_} of {hgb.max_iter} boosting rounds")

memory_report("after HGB")
```

    hist gradient boosting fit   8.5s  ROC-AUC 0.9292   PR-AUC 0.8303
    early stopping used 145 of 400 boosting rounds
    RAM  after HGB        17.00 / 20.97 GB
    VRAM after HGB         0.00 GB allocated /  0.00 GB reserved

## 11. MLP: the deep baseline that does not win

A two-hidden-layer network on the same one-hot features the logistic regression saw. This is the Grinsztajn et al. (2022) result in miniature: with the same budget of attention from you, the neural net lands **between** the linear model and the boosted trees, costs more to fit, and is far more sensitive to scaling and learning rate.

It is not that neural nets cannot fit tables. It is that their inductive bias (smooth functions, rotation-equivariant layers) is the wrong prior for a target that is piecewise constant in a few of many mostly-irrelevant columns, which is what tabular targets usually are.

------------------------------------------------------------------------

``` python
from sklearn.neural_network import MLPClassifier

mlp = Pipeline([
    ("prep", preprocess_dense),
    ("clf", MLPClassifier(hidden_layer_sizes=(128, 64), alpha=1e-3,
                          batch_size=256, learning_rate_init=1e-3,
                          early_stopping=True, n_iter_no_change=10,
                          max_iter=100, random_state=0)),
])

t0 = time.perf_counter()
mlp.fit(X_tr, y_tr)
fit_s = time.perf_counter() - t0

p_mlp = mlp.predict_proba(X_te)[:, 1]
print(f"MLP (128, 64)         fit {fit_s:5.1f}s   "
      f"ROC-AUC {roc_auc_score(y_te, p_mlp):.4f}   PR-AUC {average_precision_score(y_te, p_mlp):.4f}")
print(f"stopped after {mlp.named_steps['clf'].n_iter_} epochs")

free_memory()
memory_report("after MLP")
```

    MLP (128, 64)         fit   6.5s   ROC-AUC 0.9108   PR-AUC 0.7811
    stopped after 24 epochs
    RAM  after MLP        16.98 / 20.97 GB
    VRAM after MLP         0.00 GB allocated /  0.00 GB reserved

## 12. The production boosters: XGBoost, LightGBM, CatBoost

These three are what a production tabular stack actually runs. `HistGradientBoosting` above is a LightGBM-style implementation and gets you most of the way, but the differences below are real and they are the reason all three of these are still separate projects.

What each brings over `HistGradientBoosting`:

- **XGBoost** - the most tuning surface (and the most tuning literature), GPU training via `device="cuda"`, and `monotone_constraints`, which is how you make a credit model defensible.
- **LightGBM** - leaf-wise growth and EFB feature bundling; typically the fastest to train on wide tables, and the usual choice when retraining is hourly.
- **CatBoost** - ordered target statistics for categoricals, which is the one genuine algorithmic difference in the group and matters when you have `merchant_id`-style columns with tens of thousands of levels. It is also the slowest of the three here by an order of magnitude, which is the trade it makes.

Each takes categoricals differently, and the differences are exactly the kind of thing that silently costs you a point of AUC: XGBoost wants pandas `category` dtype with `enable_categorical=True`, LightGBM wants the column names passed to `fit`, and CatBoost wants strings with no NaNs at all. The cell handles each on its own terms rather than forcing one encoding on all three.

------------------------------------------------------------------------

``` python
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier

extra_results = {}

# XGBoost and LightGBM both read pandas "category" dtype. Setting the test frame's
# categories from the TRAINING frame matters: without it, a level absent from the test
# split shifts every code after it and the model reads the wrong category.
X_tr_cat, X_te_cat = X_tr.copy(), X_te.copy()
for c in CATEGORICAL:
    X_tr_cat[c] = X_tr_cat[c].astype("category")
    X_te_cat[c] = X_te_cat[c].astype("category").cat.set_categories(
        X_tr_cat[c].cat.categories)

xgb = XGBClassifier(n_estimators=400, learning_rate=0.1, max_depth=6,
                    subsample=0.8, colsample_bytree=0.8,
                    enable_categorical=True, tree_method="hist",   # device="cuda" to train on GPU
                    eval_metric="logloss", n_jobs=4, random_state=0)
t0 = time.perf_counter()
xgb.fit(X_tr_cat, y_tr)
extra_results["XGBoost"] = (xgb.predict_proba(X_te_cat)[:, 1], time.perf_counter() - t0)

lgbm = LGBMClassifier(n_estimators=400, learning_rate=0.1, num_leaves=31,
                      n_jobs=4, random_state=0, verbose=-1)
t0 = time.perf_counter()
lgbm.fit(X_tr_cat, y_tr, categorical_feature=CATEGORICAL)
extra_results["LightGBM"] = (lgbm.predict_proba(X_te_cat)[:, 1], time.perf_counter() - t0)

# CatBoost will not take NaN in a categorical column at all, and it wants strings rather
# than the category dtype - so "missing" becomes an explicit level, which is the right
# treatment anyway (section 10: a blank workclass is informative).
X_tr_cb, X_te_cb = X_tr.copy(), X_te.copy()
for c in CATEGORICAL:
    X_tr_cb[c] = X_tr_cb[c].astype(str).fillna("missing")
    X_te_cb[c] = X_te_cb[c].astype(str).fillna("missing")

cb = CatBoostClassifier(iterations=400, learning_rate=0.1, depth=6,
                        cat_features=CATEGORICAL, verbose=0,
                        thread_count=4, random_seed=0)
t0 = time.perf_counter()
cb.fit(X_tr_cb, y_tr)
extra_results["CatBoost"] = (cb.predict_proba(X_te_cb)[:, 1], time.perf_counter() - t0)

for name, (p, secs) in extra_results.items():
    print(f"{name:10s} fit {secs:5.1f}s  ROC-AUC {roc_auc_score(y_te, p):.4f}  "
          f"PR-AUC {average_precision_score(y_te, p):.4f}")

del xgb, lgbm, cb
free_memory()
```

    XGBoost    fit   1.1s  ROC-AUC 0.9250  PR-AUC 0.8226
    LightGBM   fit   0.4s  ROC-AUC 0.9279  PR-AUC 0.8292
    CatBoost   fit   5.4s  ROC-AUC 0.9300  PR-AUC 0.8326

## 13. Head-to-head Benchmark

Same split, same test rows, every metric computed the same way. The four sklearn models are refit here from scratch so the timings are comparable, and the three boosters from section 12 are folded in - seven models in total.

Read the table in this order: **PR-AUC** (does the ranking find the 24% positives), then **Brier / log loss** (are the probabilities usable), then **fit time** (what does a retrain cost). Accuracy is included only to show how little it separates the models.

Hardware: knowledge-lab (4 vCPU, 20 GB RAM); 36,631 training rows, 12,211 test rows. A single table is a smoke test - a real comparison runs 20+ tables, because between-table variance is larger than between-model variance.

------------------------------------------------------------------------

``` python
from sklearn.metrics import accuracy_score, brier_score_loss, f1_score, log_loss

MODELS = {
    "LogisticRegression": (lambda: Pipeline([
        ("prep", preprocess_dense),
        ("clf", LogisticRegression(max_iter=2000))]), "raw"),
    "RandomForest": (lambda: Pipeline([
        ("prep", preprocess_ordinal),
        ("clf", RandomForestClassifier(n_estimators=300, min_samples_leaf=2,
                                       n_jobs=4, random_state=0))]), "raw"),
    "HistGradientBoosting": (lambda: HistGradientBoostingClassifier(
        max_iter=400, learning_rate=0.1, categorical_features=CAT_MASK,
        early_stopping=True, n_iter_no_change=20, random_state=0), "raw"),
    "MLP (128, 64)": (lambda: Pipeline([
        ("prep", preprocess_dense),
        ("clf", MLPClassifier(hidden_layer_sizes=(128, 64), alpha=1e-3, batch_size=256,
                              early_stopping=True, max_iter=100, random_state=0))]), "raw"),
}

probs, rows = {}, []
for name, (make, _) in MODELS.items():
    model = make()
    t0 = time.perf_counter()
    model.fit(X_tr, y_tr)
    fit_s = time.perf_counter() - t0

    t0 = time.perf_counter()
    p = model.predict_proba(X_te)[:, 1]
    pred_s = time.perf_counter() - t0

    probs[name] = p
    rows.append(dict(model=name,
                     roc_auc=roc_auc_score(y_te, p),
                     pr_auc=average_precision_score(y_te, p),
                     log_loss=log_loss(y_te, p),
                     brier=brier_score_loss(y_te, p),
                     accuracy=accuracy_score(y_te, p >= 0.5),
                     f1=f1_score(y_te, p >= 0.5),
                     fit_s=fit_s, pred_ms=pred_s * 1000))
    del model                 # free each model before fitting the next
    free_memory()

for name, (p, s) in extra_results.items():   # the three boosters from section 12
    probs[name] = p
    rows.append(dict(model=name, roc_auc=roc_auc_score(y_te, p),
                     pr_auc=average_precision_score(y_te, p),
                     log_loss=log_loss(y_te, p), brier=brier_score_loss(y_te, p),
                     accuracy=accuracy_score(y_te, p >= 0.5),
                     f1=f1_score(y_te, p >= 0.5), fit_s=s, pred_ms=float("nan")))

bench = pd.DataFrame(rows).sort_values("pr_auc", ascending=False).reset_index(drop=True)
memory_report("after benchmark")
bench.round(4)
```

    RAM  after benchmark  17.16 / 20.97 GB
    VRAM after benchmark   0.00 GB allocated /  0.00 GB reserved

|  | model | roc_auc | pr_auc | log_loss | brier | accuracy | f1 | fit_s | pred_ms |
|----|----|----|----|----|----|----|----|----|----|
| 0 | CatBoost | 0.9300 | 0.8326 | 0.2735 | 0.0870 | 0.8731 | 0.7090 | 5.4054 | NaN |
| 1 | HistGradientBoosting | 0.9295 | 0.8304 | 0.2748 | 0.0875 | 0.8731 | 0.7125 | 0.4583 | 43.8145 |
| 2 | LightGBM | 0.9279 | 0.8292 | 0.2786 | 0.0882 | 0.8718 | 0.7097 | 0.4494 | NaN |
| 3 | XGBoost | 0.9250 | 0.8226 | 0.2855 | 0.0901 | 0.8723 | 0.7110 | 1.0614 | NaN |
| 4 | RandomForest | 0.9157 | 0.8069 | 0.3003 | 0.0944 | 0.8627 | 0.6816 | 1.5116 | 151.0576 |
| 5 | MLP (128, 64) | 0.9108 | 0.7811 | 0.3148 | 0.0994 | 0.8575 | 0.6768 | 2.7397 | 19.5984 |
| 6 | LogisticRegression | 0.9036 | 0.7623 | 0.3226 | 0.1030 | 0.8509 | 0.6558 | 0.2848 | 12.2775 |

``` python
from pyecharts.globals import CurrentConfig

# assets.pyecharts.org serves an expired TLS certificate from most nodes (2026-11-11),
# so charts failed to load. jsDelivr mirrors the same assets, pinned to one commit.
CurrentConfig.ONLINE_HOST = "https://cdn.jsdelivr.net/gh/pyecharts/pyecharts-assets@f2328d558550046fc5449672a6850e848e141590/assets/v6/"

from pyecharts import options as opts
from pyecharts.charts import Bar

names = bench["model"].tolist()
bar = (
    Bar()
    .add_xaxis(names)
    .add_yaxis("ROC-AUC", [round(v, 4) for v in bench["roc_auc"]])
    .add_yaxis("PR-AUC", [round(v, 4) for v in bench["pr_auc"]])
    .set_series_opts(label_opts=opts.LabelOpts(is_show=False))
    .set_global_opts(
        title_opts=opts.TitleOpts(title="Adult census: ranking quality",
                                  subtitle="12,211 held-out rows, 24% positive"),
        xaxis_opts=opts.AxisOpts(axislabel_opts=opts.LabelOpts(rotate=20)),
        yaxis_opts=opts.AxisOpts(name="score", min_=0.5, max_=1.0),
        tooltip_opts=opts.TooltipOpts(trigger="axis"),
        legend_opts=opts.LegendOpts(pos_top="8%"),
    )
)
bar.render_notebook()
```

``` python
from pyecharts.charts import Scatter

# The axis that actually decides a production pick: quality against retrain cost.
scatter = Scatter()
scatter.add_xaxis([round(float(s), 2) for s in bench["fit_s"]])
for name, prauc, fits in zip(bench["model"], bench["pr_auc"], bench["fit_s"]):
    scatter.add_yaxis(
        name, [[round(float(fits), 2), round(float(prauc), 4)]],
        symbol_size=18, label_opts=opts.LabelOpts(is_show=False),
    )
scatter.set_global_opts(
    title_opts=opts.TitleOpts(title="PR-AUC vs fit time", subtitle="up and to the left is better"),
    xaxis_opts=opts.AxisOpts(type_="value", name="fit seconds (4 vCPU)"),
    yaxis_opts=opts.AxisOpts(type_="value", name="PR-AUC", min_="dataMin"),
    tooltip_opts=opts.TooltipOpts(trigger="item", formatter="{a}: {c}"),
    legend_opts=opts.LegendOpts(pos_top="8%"),
)
scatter.render_notebook()
```

``` python
from pyecharts.charts import Line
from sklearn.metrics import precision_recall_curve, roc_curve

# ROC and precision-recall for the same models, subsampled to keep the chart light.
def thin(x, y, k=120):
    idx = np.linspace(0, len(x) - 1, min(k, len(x))).astype(int)
    return [round(float(v), 4) for v in np.asarray(x)[idx]], \
           [round(float(v), 4) for v in np.asarray(y)[idx]]

grid = [round(v, 3) for v in np.linspace(0, 1, 101)]
roc_line = Line().add_xaxis(grid)
pr_line = Line().add_xaxis(grid)
for name, p in probs.items():
    fpr, tpr, _ = roc_curve(y_te, p)
    roc_line.add_yaxis(name, [round(float(v), 4) for v in np.interp(grid, fpr, tpr)],
                       is_smooth=True, symbol="none",
                       label_opts=opts.LabelOpts(is_show=False))
    prec, rec, _ = precision_recall_curve(y_te, p)
    order = np.argsort(rec)
    pr_line.add_yaxis(name, [round(float(v), 4) for v in np.interp(grid, rec[order], prec[order])],
                      is_smooth=True, symbol="none",
                      label_opts=opts.LabelOpts(is_show=False))

roc_line.set_global_opts(
    title_opts=opts.TitleOpts(title="ROC curves", subtitle="diagonal = random"),
    xaxis_opts=opts.AxisOpts(type_="value", name="false positive rate"),
    yaxis_opts=opts.AxisOpts(name="true positive rate"),
    tooltip_opts=opts.TooltipOpts(trigger="axis"),
    legend_opts=opts.LegendOpts(pos_top="8%"),
)
roc_line.render_notebook()
```

``` python
pr_line.set_global_opts(
    title_opts=opts.TitleOpts(
        title="Precision-recall curves",
        subtitle=f"baseline (positive rate) = {y_te.mean():.3f} - this is the metric under imbalance"),
    xaxis_opts=opts.AxisOpts(type_="value", name="recall"),
    yaxis_opts=opts.AxisOpts(name="precision", min_=0, max_=1),
    tooltip_opts=opts.TooltipOpts(trigger="axis"),
    legend_opts=opts.LegendOpts(pos_top="8%"),
)
pr_line.render_notebook()
```

## 14. Calibration and the Threshold

The benchmark ranked the models. Deploying one needs two more things the ranking does not give you: **probabilities that mean what they say**, and **a threshold**.

A **reliability diagram** bins predictions by score and plots the observed positive rate in each bin. A perfectly calibrated model sits on the diagonal. The boosted trees land closest to it because they optimise log loss directly; the random forest is roughly twice as far off, since averaging hundreds of votes pulls probabilities towards the middle and leaves it under-confident at both extremes; the MLP is the worst of the four here. On this table all of them are decent in absolute terms - Adult is balanced enough (24% positive) that nothing is badly distorted. On a 0.2%-positive fraud table the same chart separates the models dramatically.

**Expected calibration error (ECE)** summarises the gap as a weighted mean absolute deviation from the diagonal. If it is large, wrap the model in `CalibratedClassifierCV` (isotonic for plenty of data, sigmoid/Platt for little) fitted on a held-out slice - never on the training rows, which are already fit.

Then the threshold: sweep it against your own cost ratio and take the minimum. The default of 0.5 is only optimal when false positives and false negatives cost the same and the classes are balanced, which is essentially never.

------------------------------------------------------------------------

``` python
N_BINS = 12
edges = np.linspace(0, 1, N_BINS + 1)
centres = [round(float(c), 3) for c in (edges[:-1] + edges[1:]) / 2]

rel = Line().add_xaxis(centres)
rel.add_yaxis("perfect calibration", centres, is_smooth=False, symbol="none",
              linestyle_opts=opts.LineStyleOpts(type_="dashed"),
              label_opts=opts.LabelOpts(is_show=False))

ece_rows = []
for name, p in probs.items():
    idx = np.clip(np.digitize(p, edges) - 1, 0, N_BINS - 1)
    obs, conf, weight = [], [], []
    for b in range(N_BINS):
        m = idx == b
        obs.append(float(y_te.to_numpy()[m].mean()) if m.sum() else None)
        conf.append(float(p[m].mean()) if m.sum() else None)
        weight.append(int(m.sum()))
    rel.add_yaxis(name, [None if o is None else round(o, 4) for o in obs],
                  is_smooth=True, symbol="circle", symbol_size=6,
                  label_opts=opts.LabelOpts(is_show=False))
    ece = sum(w * abs(o - c) for o, c, w in zip(obs, conf, weight) if o is not None) / len(p)
    ece_rows.append((name, ece, brier_score_loss(y_te, p)))

rel.set_global_opts(
    title_opts=opts.TitleOpts(title="Reliability diagram",
                              subtitle="below the diagonal = over-confident, above = under-confident"),
    xaxis_opts=opts.AxisOpts(type_="value", name="predicted probability", min_=0, max_=1),
    yaxis_opts=opts.AxisOpts(name="observed positive rate", min_=0, max_=1),
    tooltip_opts=opts.TooltipOpts(trigger="axis"),
    legend_opts=opts.LegendOpts(pos_top="8%"),
)
print(pd.DataFrame(ece_rows, columns=["model", "ECE", "Brier"])
        .sort_values("ECE").round(4).to_string(index=False))
rel.render_notebook()
```

                   model    ECE  Brier
                CatBoost 0.0047 0.0870
    HistGradientBoosting 0.0059 0.0875
      LogisticRegression 0.0074 0.1030
            RandomForest 0.0104 0.0944
                LightGBM 0.0127 0.0882
                 XGBoost 0.0157 0.0901
           MLP (128, 64) 0.0208 0.0994

``` python
# Threshold selection under an asymmetric cost. Say a missed high earner costs 5x a
# false alert (substitute your own ratio - this is the number the business owns).
C_FP, C_FN = 1.0, 5.0
grid = np.linspace(0.02, 0.98, 97)
y_te_np = y_te.to_numpy()

cost_line = Line().add_xaxis([round(float(t), 3) for t in grid])
summary = []
for name, p in probs.items():
    costs = [C_FP * ((p >= t) & (y_te_np == 0)).sum() + C_FN * ((p < t) & (y_te_np == 1)).sum()
             for t in grid]
    cost_line.add_yaxis(name, [int(c) for c in costs], is_smooth=True, symbol="none",
                        label_opts=opts.LabelOpts(is_show=False))
    best = int(np.argmin(costs))
    summary.append((name, float(grid[best]), int(costs[best]),
                    int(costs[np.argmin(np.abs(grid - 0.5))]),
                    f1_score(y_te, p >= grid[best])))

cost_line.set_global_opts(
    title_opts=opts.TitleOpts(title=f"Expected cost vs threshold (C_FN/C_FP = {C_FN / C_FP:.0f})",
                              subtitle="the minimum, not 0.5, is the operating point"),
    xaxis_opts=opts.AxisOpts(type_="value", name="threshold"),
    yaxis_opts=opts.AxisOpts(name="total cost", min_="dataMin"),
    tooltip_opts=opts.TooltipOpts(trigger="axis"),
    legend_opts=opts.LegendOpts(pos_top="8%"),
)
print(pd.DataFrame(summary, columns=["model", "best_threshold", "cost_at_best",
                                     "cost_at_0.5", "F1_at_best"]).round(3).to_string(index=False))
cost_line.render_notebook()
```

                   model  best_threshold  cost_at_best  cost_at_0.5  F1_at_best
      LogisticRegression            0.14          3974         6569       0.638
            RandomForest            0.17          3823         6188       0.659
    HistGradientBoosting            0.14          3416         5561       0.680
           MLP (128, 64)            0.12          3831         6140       0.652
                 XGBoost            0.17          3557         5575       0.690
                LightGBM            0.13          3417         5601       0.681
                CatBoost            0.19          3375         5689       0.698

## 15. Which Columns Actually Matter

Tree ensembles ship a built-in `feature_importances_`, and it is **biased**: impurity-based importance inflates high-cardinality and continuous columns, because they offer more places to split. **Permutation importance** measures the thing you actually care about - how much the held-out score drops when one column is shuffled - and works for any model, at the cost of one refit-free evaluation per column per repeat.

Its own trap is **correlated features**: shuffle one of two duplicated columns and the model leans on the other, so both look unimportant. Group correlated columns and permute the group when that matters.

------------------------------------------------------------------------

``` python
from sklearn.inspection import permutation_importance

hgb_final = HistGradientBoostingClassifier(
    max_iter=400, learning_rate=0.1, categorical_features=CAT_MASK,
    early_stopping=True, n_iter_no_change=20, random_state=0).fit(X_tr, y_tr)

# 2,000 test rows and 5 repeats keeps this near a minute on 4 vCPU.
sub = X_te.iloc[:2000]
perm = permutation_importance(hgb_final, sub, y_te.iloc[:2000], n_repeats=5,
                              scoring="average_precision", random_state=0, n_jobs=1)

imp = (pd.DataFrame({"feature": X_te.columns, "drop": perm.importances_mean,
                     "std": perm.importances_std})
         .sort_values("drop", ascending=True))

imp_bar = (
    Bar()
    .add_xaxis(imp["feature"].tolist())
    .add_yaxis("PR-AUC drop when shuffled", [round(float(v), 4) for v in imp["drop"]],
               label_opts=opts.LabelOpts(is_show=False))
    .reversal_axis()
    .set_global_opts(
        title_opts=opts.TitleOpts(title="Permutation importance (HistGradientBoosting)",
                                  subtitle="2,000 test rows, 5 repeats, scored on PR-AUC"),
        xaxis_opts=opts.AxisOpts(name="mean PR-AUC drop"),
        yaxis_opts=opts.AxisOpts(axislabel_opts=opts.LabelOpts(font_size=10)),
        tooltip_opts=opts.TooltipOpts(trigger="item"),
    )
)
del hgb_final
free_memory()
memory_report("after importance")
imp_bar.render_notebook()
```

    RAM  after importance 17.18 / 20.97 GB
    VRAM after importance  0.00 GB allocated /  0.00 GB reserved

## 16. Common Frameworks

Tabular is the one task in this folder where `transformers` does not appear at all. The ecosystem predates deep learning, and it has stayed separate for a good reason: gradient-boosted trees still win on tabular data, so the stack is scikit-learn and the three boosting libraries, surrounded by tooling for the things that actually decide a tabular project - the split, the threshold, the calibration, and the explanation someone will ask for.

| Framework | Layer | What it gives you | License | Reach for it when |
|----|----|----|----|----|
| [scikit-learn](https://github.com/scikit-learn/scikit-learn) | modelling | The estimators, `Pipeline`, `ColumnTransformer`, the cross-validation splitters, and `HistGradientBoosting` - a first-class booster with no extra dependency | BSD-3 | Default, and often sufficient. The `Pipeline` is what stops preprocessing leaking across the split |
| [XGBoost](https://github.com/dmlc/xgboost) / [LightGBM](https://github.com/microsoft/LightGBM) / [CatBoost](https://github.com/catboost/catboost) | modelling | The production boosters: categorical handling, missing values, monotone constraints, GPU training, and `scale_pos_weight` | Apache 2.0 / MIT | Serious work. Section 12 shows the gap to `HistGradientBoosting` is small - the differentiators are the constraints and the categorical handling |
| [AutoGluon](https://github.com/autogluon/autogluon) / [H2O AutoML](https://github.com/h2oai/h2o-3) | modelling | Multi-layer stacking over boosters, neural nets and (since 2025) TabPFN, for one line of code and a lot of CPU | Apache 2.0 | A strong baseline in an afternoon. Usually beats a hand-tuned single model by a point or two |
| [TabPFN](https://github.com/PriorLabs/TabPFN) | modelling | A pretrained transformer that classifies in a forward pass with no gradient steps, frequently best below ~10k rows | Apache 2.0 (v2 weights: see the model license) | Small tables. A vendor package rather than a general-purpose library, so it is prose here - but the results are real |
| [pandas](https://github.com/pandas-dev/pandas) / [polars](https://github.com/pola-rs/polars) | data | Loading, joins and feature construction - and in polars’ case, doing it on 4 cores without running out of memory | BSD-3 / MIT | Always. On this box polars’ lazy engine is the difference between a table that fits and one that does not |
| [imbalanced-learn](https://github.com/scikit-learn-contrib/imbalanced-learn) + [Optuna](https://github.com/optuna/optuna) | data | Resampling strategies inside a `Pipeline`, and successive-halving hyperparameter search over the booster | MIT | Tuning. Search `learning_rate`, `max_leaf_nodes`, `min_samples_leaf` and `l2_regularization` on cross-validated PR-AUC - and prefer class weights to resampling |
| [ONNX Runtime](https://onnxruntime.ai/) / [Treelite](https://github.com/dmlc/treelite) | inference runtime | A tree ensemble compiled to a portable artefact that scores in microseconds without a Python process | MIT / Apache 2.0 | Deployment. Tabular models usually serve inside an existing service, in a language that is not Python |
| [MLflow](https://github.com/mlflow/mlflow) / [Feast](https://github.com/feast-dev/feast) | serving | Model registry and versioning, and a feature store that computes a feature identically offline and online | Apache 2.0 | Production. Training/serving skew in feature computation is the classic tabular failure and a feature store is the structural fix |
| [SHAP](https://github.com/shap/shap) + `sklearn.inspection` | evaluation | Exact Shapley values for tree ensembles in near-linear time, partial dependence, and permutation importance | MIT / BSD-3 | Always, and mandatory in credit and insurance. Section 15 is the aggregate view; SHAP is the per-prediction one |

**The 2026 default stack** is polars to load, a scikit-learn `Pipeline` wrapping LightGBM or CatBoost, Optuna with successive halving to tune, calibration and a swept threshold at the end, SHAP for explanation, and ONNX or Treelite to deploy. AutoGluon when you want the strong baseline without the work.

The common wrong turn is switching model classes to chase accuracy. On a table this size the gap between LightGBM, XGBoost and CatBoost is far smaller than the gap between a tuned and untuned booster, and both are smaller than the gap made by getting the split right. If rows repeat per entity, use `GroupKFold`; if there is a time column, split on time. The second is fixing imbalance by resampling, which changes the base rate the model learns and breaks calibration - use class weights and a cost-swept threshold.

------------------------------------------------------------------------

## 17. Going Further

**Tune the boosting model, not the model class.** On a table this size, a 30-trial random search over `learning_rate`, `max_leaf_nodes`, `min_samples_leaf` and `l2_regularization` buys more than switching between LightGBM, XGBoost and CatBoost. Use `HalvingRandomSearchCV` (successive halving) rather than a full grid, and search on cross-validated PR-AUC, not accuracy.

**Try TabPFN v2 on your small tables.** `pip install tabpfn`, then `TabPFNClassifier().fit(X, y).predict_proba(X_test)` - no gradient steps, seconds to a result, and on tables under ~10k rows it is frequently the best single model available. It is a vendor package rather than a general-purpose library, so it is not a dependency here; see the general-purpose-library rule in `CLAUDE.md`. Its limits are real: ~10k rows, ~500 features, ~10 classes, and inference carries the whole training set, so it is slow at serving time.

**Get the split right before anything else.** If rows repeat per entity, use `GroupKFold`. If the table has a time column, use `TimeSeriesSplit` and evaluate on the future. Most “our model degraded in production” stories are a random split that never existed in production.

**Fix imbalance at the threshold, not by resampling.** SMOTE and random oversampling change the base rate the model learns and therefore break calibration. `class_weight` / `scale_pos_weight` and a cost-swept threshold are usually better, and always simpler to reason about.

**Explain per prediction when the domain demands it.** `shap.TreeExplainer` gives exact Shapley values for tree ensembles in near-linear time; partial dependence and ICE (`sklearn.inspection`) show the shape of a feature’s effect. For credit and insurance, add `monotone_constraints` (XGBoost/LightGBM) so “more income never lowers the score” is guaranteed rather than hoped for.

**Related notebooks in this repo:** `01_Tabular_Regression` (same pipeline, continuous target), `02_Time_Series_Forecasting` (when the rows are ordered), and `Natural_Language_Processing/00_Text_Classification` (when the informative column is free text).

------------------------------------------------------------------------

Back to top
