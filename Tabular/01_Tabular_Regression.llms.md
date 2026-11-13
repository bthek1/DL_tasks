# Tabular Regression

Predicting a continuous number from a row of a table: which error metric encodes which business decision, why a point forecast is usually the wrong deliverable, how quantile regression gives you an interval for free, and runnable code that scores five models on one California housing split.

Author

Benedict Thekkel

## 1. What is Tabular Regression?

Tabular regression maps **one row of a table** to **one real number**: a house price, a delivery time, a remaining useful life, a claim size. Structurally it is the same problem as tabular classification - the same mixed columns, the same missing values, the same models - and the differences that matter are all downstream of the target being continuous.

**Input.** Identical to classification: numeric columns on different scales, categorical and ordinal columns, informative missing values, no spatial or temporal structure to exploit.

**Output.** A real number, and this is where the two tasks diverge:

- **The loss encodes an assumption about the error distribution.** Squared error assumes symmetric, Gaussian-ish noise and is dominated by the worst rows. Absolute error targets the median and shrugs at outliers. Pinball loss targets an arbitrary quantile. Picking a loss is picking what “wrong” means.
- **Targets are often skewed and positive.** Prices, durations, counts and claim sizes have a long right tail. Fitting squared error on the raw scale lets the top 1% of rows write the model; fitting on `log1p(y)` usually does not.
- **A single number is rarely the deliverable.** “This house is worth \$412,000” is much less useful than “80% chance it is between \$360k and \$470k”. Quantile regression gets you there without a second modelling framework.
- **There is no threshold to tune**, but there is a **transform to undo**, and undoing it naively is biased (section 12).

**Neighbouring tasks:**

| Task | What it does | Typical tool |
|----|----|----|
| Tabular classification | Discrete label instead of a number | see `00_Tabular_Classification` |
| Time series forecasting | Rows are ordered; the target is the future of the same series | see `02_Time_Series_Forecasting` |
| Quantile / distributional regression | Predict an interval or full distribution | LightGBM `objective="quantile"`, NGBoost, CatBoost `RMSEWithUncertainty` |
| Survival regression | Time to an event, with censoring | scikit-survival, XGBoost AFT |
| Ordinal regression | Ordered categories (ratings) | `mord`, or regression + thresholds |
| Counting / rate models | Non-negative integer targets | Poisson / Tweedie objectives |

------------------------------------------------------------------------

## 2. Real-World Use Cases

| Use case | Domain | Consumes / produces | Dominant constraint |
|----|----|----|----|
| Automated property valuation (AVM) | Real estate (Zillow Zestimate, Redfin) | Property attributes, location, comparable sales -\> price | Interval width matters more than the point estimate; geographic drift; a public error number is reputational |
| Demand and price elasticity | Retail, e-commerce | SKU, price, promotion, calendar -\> units sold | Asymmetric cost (stockout vs overstock); needs to extrapolate to unseen prices |
| Delivery-time estimation (ETA) | Logistics, ride-hailing, food delivery | Route, traffic, courier, order features -\> minutes | Tail latency, not mean error: quantiles are the product; sub-100 ms serving |
| Insurance claim severity and pricing | Insurance | Policy + incident features -\> expected loss | Long-tailed target (Tweedie/gamma losses); regulatory monotonicity; legal defensibility |
| Remaining useful life | Manufacturing, aviation | Sensor aggregates + usage -\> hours until failure | Under-prediction is expensive, over-prediction is catastrophic - the loss must be asymmetric |
| Energy load and price forecasting | Utilities, trading | Weather, calendar, historical load -\> MW or price | Quantiles feed the bidding strategy; errors compound through the market |
| Cloud cost and capacity planning | Infrastructure | Workload metrics -\> resource need | Systematic over-provisioning is the cheap failure; needs high quantiles |
| Credit exposure and LGD | Banking | Loan and collateral features -\> loss given default | Bounded target in \[0, 1\], often bimodal; explainability required |

**What the RMSE hides.** A single held-out error number is a poor summary of a regression system for reasons that show up only in production.

- **The error distribution is not the error.** Two models with the same RMSE can differ by a factor of five in worst-case error. Look at the residual quantiles, not just their mean square.
- **Heteroscedasticity is the norm.** Error grows with the target for prices, durations and claims. A constant-width interval is wrong at both ends, which is the practical argument for quantile regression over “point estimate plus a global sigma”.
- **The cost of being wrong is usually asymmetric.** Under-estimating an ETA annoys a customer; over-estimating it loses the order. Fitting squared error and then apologising for the bias is strictly worse than fitting the quantile you actually want.
- **Trees cannot extrapolate.** A tree ensemble predicts a constant outside the range of the training data - move to a new price band or a new region and it flatlines. Linear models extrapolate (sometimes absurdly), which is one of the few places they still win.
- **Bounded and censored targets break plain regression.** A model that can predict a negative house price or a 130% loss ratio will do so, and it will do it on exactly the rows a reviewer reads first.

------------------------------------------------------------------------

## 3. How Modern Tabular Regression Works

The architecture story is the same as classification (linear -\> bagged trees -\> boosting -\> deep -\> foundation models), so this section covers the part that is specific to a continuous target: **the objective**.

**1. Least squares (Gauss, 1805).** Minimise $\sum (y - \hat y)^2$. Predicts the conditional **mean**, has a closed form for linear models, and is dominated by the largest residuals - which is a feature when errors really are Gaussian and a bug when the target is long-tailed.

**2. Robust and quantile losses (1970s onward).** Absolute error predicts the **median**; Huber loss is quadratic near zero and linear in the tails, so it keeps the smooth gradients of MSE without letting one outlier dominate; **pinball (quantile) loss** with parameter $\tau$ predicts the $\tau$-quantile:

$$L_\tau(y, \hat y) = \max\big(\tau (y - \hat y),\ (\tau - 1)(y - \hat y)\big)$$

Fitting $\tau = 0.1$ and $\tau = 0.9$ gives an 80% prediction interval directly.

**3. Distribution-matched objectives (GLM heritage, alive in boosting).** Poisson for counts, gamma for positive skewed magnitudes, **Tweedie** for the mass-at-zero-plus-positive-tail shape that insurance claims have. Every serious GBDT implements these, and choosing the right one beats tuning the wrong one.

**4. Gradient boosting (2001 -\> 2018) is still the default.** Same engineering wave as classification: XGBoost, LightGBM, CatBoost, scikit-learn’s `HistGradientBoosting`. All support squared, absolute, Huber, quantile and Poisson/Tweedie objectives, so the objective choice above is a one-line change.

**5. Deep tabular models (2019-2023).** TabNet, NODE, FT-Transformer. Same verdict as classification - Grinsztajn et al. (2022) found GBDTs still ahead on medium tabular data. Neural nets earn a place when the row is fused with text or images, or when you need one differentiable model end to end.

**6. Tabular foundation models (2022 -\> now).** TabPFN v2 (Hollmann et al., *Nature*, Jan 2025) covers regression as well as classification, and outputs a **full predictive distribution** rather than a point, which is unusually well matched to what regression deployments need. Under ~10k rows it is the current small-data leader; TabICL and TabDPT push the row limit further.

**7. Conformal prediction (orthogonal, and worth more than a model swap).** Wrap *any* regressor and get intervals with a finite-sample coverage guarantee, using only a calibration split. `MAPIE` implements it in a few lines. If you need “80% of the time the truth is inside this band” to actually be true, this is the tool, not a quantile model’s optimism.

------------------------------------------------------------------------

## 4. Evaluation Metrics

Every metric here is a different answer to “what counts as wrong”. Pick the one that matches the decision, then report a second one that is scale-free so a reader can judge it.

| Metric | Formula | Predicts | Use when |
|----|----|----|----|
| MSE / RMSE | $\sqrt{\frac{1}{N}\sum (y - \hat y)^2}$ | mean | Errors are symmetric and large errors hurt superlinearly |
| MAE | $\frac{1}{N}\sum \lvert y - \hat y \rvert$ | median | Outliers exist and should not dominate |
| MAPE | $\frac{100}{N}\sum \lvert \frac{y - \hat y}{y} \rvert$ | \- | Relative error is the business unit; **breaks near y = 0** and punishes over-prediction more than under |
| sMAPE / MASE | see `02_Time_Series_Forecasting` | \- | Comparing across series of different scale |
| $R^2$ | $1 - \frac{SS_{res}}{SS_{tot}}$ | \- | A scale-free sanity check; 0 means “no better than the mean” |
| Pinball loss $L_\tau$ | above | $\tau$-quantile | Grading an interval or a quantile forecast |
| Coverage | $\frac{1}{N}\sum \mathbb{1}[l \le y \le u]$ | \- | Checking a stated interval keeps its promise |

**Pitfalls.**

- **RMSE and MAE are in the units of the target**, so they are only interpretable next to the target’s own scale. Always report the target’s standard deviation or IQR alongside.
- $R^2$ on a test split can be negative. That is not a bug; it means the model is worse than predicting the training mean.
- **MAPE is asymmetric and undefined at zero.** A forecast of 0 against a truth of 1 gives 100%; a forecast of 2 against a truth of 1 gives 100% too, but you can never do worse than 100% by under-predicting and arbitrarily worse by over-predicting. Use it only on strictly positive targets, and prefer MAE on the log scale if the point is relative error.
- **Averaged metrics hide the tail.** Report the 90th and 99th percentile absolute error next to the mean, especially if downstream systems have SLAs.

------------------------------------------------------------------------

``` python
import numpy as np

rng = np.random.default_rng(0)
n = 2000
y = rng.gamma(shape=2.0, scale=50.0, size=n)          # positive, right-skewed, like a price
pred_mean = np.full(n, y.mean())                       # a mean-predicting model
pred_median = np.full(n, np.median(y))                 # a median-predicting model


def report(name, yhat):
    err = yhat - y
    rmse = float(np.sqrt((err ** 2).mean()))
    mae = float(np.abs(err).mean())
    mape = float((np.abs(err) / np.maximum(y, 1e-9)).mean() * 100)
    r2 = float(1 - (err ** 2).sum() / ((y - y.mean()) ** 2).sum())
    p90 = float(np.percentile(np.abs(err), 90))
    print(f"{name:22s} RMSE {rmse:7.2f}  MAE {mae:7.2f}  MAPE {mape:7.1f}%  "
          f"R2 {r2:6.3f}  p90|err| {p90:7.2f}")


print(f"target: mean {y.mean():.1f}  median {np.median(y):.1f}  sd {y.std():.1f}\n")
report("constant = mean", pred_mean)      # wins RMSE
report("constant = median", pred_median)  # wins MAE, and MAPE by a mile


def pinball(y_true, y_pred, tau):
    "Quantile (pinball) loss. tau=0.5 is MAE/2."
    d = y_true - y_pred
    return float(np.maximum(tau * d, (tau - 1) * d).mean())


for tau in (0.1, 0.5, 0.9):
    q = float(np.quantile(y, tau))
    print(f"\ntau={tau}: best constant is the empirical quantile {q:7.2f} "
          f"(pinball {pinball(y, np.full(n, q), tau):.3f}); "
          f"the mean scores {pinball(y, pred_mean, tau):.3f}")
```

    target: mean 100.6  median 83.6  sd 71.7

    constant = mean        RMSE   71.68  MAE   55.10  MAPE   127.4%  R2  0.000  p90|err|   96.79
    constant = median      RMSE   73.67  MAE   53.53  MAPE   104.4%  R2 -0.056  p90|err|  113.14

    tau=0.1: best constant is the empirical quantile   26.77 (pinball 8.356); the mean scores 27.551

    tau=0.5: best constant is the empirical quantile   83.59 (pinball 26.765); the mean scores 27.551

    tau=0.9: best constant is the empirical quantile  196.74 (pinball 15.770); the mean scores 27.551

## 5. Datasets

Classic regression tables are small, which is a problem the field has partly fixed with curated suites. Note that **Boston Housing is deprecated** - it was removed from scikit-learn in 1.2 because a feature encodes the proportion of Black residents in a way that makes the dataset unsuitable as a teaching example. California housing replaced it.

| Dataset | Contents | Size | Scope | License | Typical use |
|----|----|----|----|----|----|
| [California Housing](https://scikit-learn.org/stable/datasets/real_world.html#california-housing-dataset) | 1990 census block groups, 8 numeric columns -\> median house value | 20,640 rows | California, USA | Public domain (US census) | The standard regression demo - **used by this notebook** |
| [Ames Housing](https://www.openml.org/d/42165) | 79 mixed columns describing houses -\> sale price | 1,460 rows | Ames, Iowa | CC BY 4.0 | Feature engineering with real categoricals and missingness |
| [Bike Sharing](https://www.openml.org/d/42712) | Hourly rentals with weather and calendar | 17,379 rows | Washington DC | CC BY 4.0 | Count target (Poisson), strong seasonality |
| [Superconductivity](https://archive.ics.uci.edu/dataset/464) | Material composition -\> critical temperature | 21,263 rows, 81 features | Materials science | CC BY 4.0 | Wide numeric table, no categoricals |
| [Allstate Claims Severity](https://www.kaggle.com/c/allstate-claims-severity) | Anonymised insurance features -\> claim cost | 188,318 rows | Insurance | Competition terms | Long-tailed target, the canonical MAE/log-target problem |
| [NYC Taxi Trip Duration](https://www.kaggle.com/c/nyc-taxi-trip-duration) | Pickup/dropoff, time -\> seconds | 1.4M rows | NYC | Competition terms | Large, geospatial, heavy tail |
| [OpenML-CTR23](https://www.openml.org/s/353) | Curated suite of 35 regression tasks | 500 to 100k rows each | Mixed | CC BY 4.0 | The standard “many tables” regression benchmark |
| [TabArena](https://tabarena.ai) | Living benchmark with a regression track | 51 tables (2025) | Mixed | Per-dataset | Current model-vs-model comparisons |

**This notebook uses California Housing** (`sklearn.datasets.fetch_california_housing`, ~1.5 MB, cached in `DL_tasks/datasets/sklearn/`). It is all-numeric, which keeps the preprocessing short, and it has three properties that make it a good teaching table: the target is **capped at \$500,001** (an artefact that shows up as a visible band in the residuals), it is **heteroscedastic**, and `Latitude`/`Longitude` carry real interaction structure that a linear model cannot use and a tree can. None of the datasets above are gated; the two Kaggle ones need an account.

------------------------------------------------------------------------

## 6. The Model Landscape (mid-2026)

Same cast as classification, judged on regression-specific abilities.

| Model | Objectives available | Extrapolates | Native quantiles | License | Best for |
|----|----|----|----|----|----|
| Ridge / Lasso / ElasticNet | squared (+ L1/L2) | **yes** | no | BSD (sklearn) | Auditability, extrapolation, a floor to beat |
| Random Forest | squared, absolute | no | via `quantile-forest` | BSD | Zero-tuning baseline |
| sklearn `HistGradientBoosting` | squared, absolute, **quantile**, Poisson, gamma | no | **yes** (one fit per quantile) | BSD | Strong default with no extra dependency |
| **LightGBM** | \+ Huber, Tweedie, fair | no | yes | MIT | Speed; the usual production pick |
| **XGBoost** | \+ Tweedie, AFT, pseudo-Huber | no | yes (2.0+) | Apache 2.0 | Tuning surface, GPU, monotone constraints |
| **CatBoost** | \+ `RMSEWithUncertainty` (mean and variance in one fit) | no | yes | Apache 2.0 | High-cardinality categoricals, cheap uncertainty |
| MLP / FT-Transformer | any differentiable | yes (badly) | yes | MIT / BSD | Multimodal fusion |
| **TabPFN v2** | full predictive distribution | no | **yes, from one forward pass** | Custom | \<= 10k rows: current small-data leader, and it gives you the distribution |
| NGBoost | natural gradient boosting of distributions | no | yes | Apache 2.0 | When you want a parametric predictive distribution |
| MAPIE (wrapper) | \- | \- | **conformal intervals with coverage guarantees** | BSD | Making any model’s intervals honest |

Leaderboards: **[TabArena](https://tabarena.ai)**, the [AutoML Benchmark](https://openml.github.io/automlbenchmark/), and OpenML-CTR23.

**What wins what.** Below ~10k rows, TabPFN v2 leads on accuracy and hands you a distribution for free. Above it, a tuned LightGBM/XGBoost/CatBoost is the practical optimum. Ridge is the only model in the table that **extrapolates**, which decides the matter whenever the deployment sees feature ranges the training data did not. And for intervals that keep their promise, conformal prediction on top of the best point model beats any single model’s built-in optimism.

**What runs below.** Everything down to the CatBoost row, plus the MLP, is a repo dependency, so section 14 is a real eight-model comparison. TabPFN v2, NGBoost and MAPIE are the notable absentees - they are vendor or single-purpose packages rather than general-purpose libraries (see the rule in `CLAUDE.md`), and section 16 says how to try each of them.

------------------------------------------------------------------------

## 7. Setup

Package roles:

- `scikit-learn` - the ridge, forest, boosting and MLP models, metrics, and the California housing fetcher
- `xgboost` / `lightgbm` / `catboost` - the three production gradient boosters (section 13)
- `pandas` - the table
- `pyecharts` - all charts (repo rule)
- `numpy` - array work
- `torch` - only for the shared `vram()` / `free_memory()` helpers; nothing here uses the GPU. These models are CPU-bound, so **RAM is the budget**.

The download (~1.5 MB) lands in `DL_tasks/datasets/sklearn/`, which is gitignored.

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

load_dotenv(find_dotenv(usecwd=True))   # Knowledge/.env sets HF_TOKEN

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
    # glibc holds freed allocations in its arenas, so RSS ratchets up across
    # sections; malloc_trim(0) hands them back. Not optional on a 20 GB box.
    try:
        ctypes.CDLL(ctypes.util.find_library("c") or "libc.so.6").malloc_trim(0)
    except Exception:
        pass


def memory_report(tag=""):
    "Print current system RAM (and VRAM if a GPU is present), in GB."
    vm = psutil.virtual_memory()
    print(f"RAM  {tag:16s} {(vm.total - vm.available) / 1e9:5.2f} / {vm.total / 1e9:5.2f} GB")
    vram(tag)


DATA_DIR = Path("../../datasets")     # gitignored
DATA_DIR.mkdir(exist_ok=True)
SKLEARN_HOME = str(DATA_DIR / "sklearn")

memory_report("baseline")
```

    device: cuda:0 (sklearn models are CPU-only; RAM is the budget here)
    RAM  baseline         15.25 / 20.97 GB
    VRAM baseline          0.00 GB allocated /  0.00 GB reserved

``` python
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split

cal = fetch_california_housing(data_home=SKLEARN_HOME, as_frame=True)
X_all, y_all = cal.data, cal.target          # target is median house value in $100k

print(f"{len(X_all):,} rows x {X_all.shape[1]} columns, all numeric")
print(f"target: mean {y_all.mean():.3f}  median {y_all.median():.3f}  "
      f"sd {y_all.std():.3f}  min {y_all.min():.3f}  max {y_all.max():.3f}  (units of $100k)")

# The target is CAPPED: every block group above $500k was recorded as 5.00001.
capped = (y_all >= 5.0).mean()
print(f"{capped:.2%} of rows sit on the $500,001 cap - a censored target, not a real value")

X_tr, X_te, y_tr, y_te = train_test_split(X_all, y_all, test_size=0.25, random_state=0)
print(f"\ntrain {X_tr.shape}  test {X_te.shape}")
X_all.describe().T.round(3)
```

    20,640 rows x 8 columns, all numeric
    target: mean 2.069  median 1.797  sd 1.154  min 0.150  max 5.000  (units of $100k)
    4.81% of rows sit on the $500,001 cap - a censored target, not a real value

    train (15480, 8)  test (5160, 8)

|  | count | mean | std | min | 25% | 50% | 75% | max |
|----|----|----|----|----|----|----|----|----|
| MedInc | 20640.0 | 3.871 | 1.900 | 0.500 | 2.563 | 3.535 | 4.743 | 15.000 |
| HouseAge | 20640.0 | 28.639 | 12.586 | 1.000 | 18.000 | 29.000 | 37.000 | 52.000 |
| AveRooms | 20640.0 | 5.429 | 2.474 | 0.846 | 4.441 | 5.229 | 6.052 | 141.909 |
| AveBedrms | 20640.0 | 1.097 | 0.474 | 0.333 | 1.006 | 1.049 | 1.100 | 34.067 |
| Population | 20640.0 | 1425.477 | 1132.462 | 3.000 | 787.000 | 1166.000 | 1725.000 | 35682.000 |
| AveOccup | 20640.0 | 3.071 | 10.386 | 0.692 | 2.430 | 2.818 | 3.282 | 1243.333 |
| Latitude | 20640.0 | 35.632 | 2.136 | 32.540 | 33.930 | 34.260 | 37.710 | 41.950 |
| Longitude | 20640.0 | -119.570 | 2.004 | -124.350 | -121.800 | -118.490 | -118.010 | -114.310 |

``` python
from pyecharts.globals import CurrentConfig

# assets.pyecharts.org serves an expired TLS certificate from most nodes (2026-11-11),
# so charts failed to load. jsDelivr mirrors the same assets, pinned to one commit.
CurrentConfig.ONLINE_HOST = "https://cdn.jsdelivr.net/gh/pyecharts/pyecharts-assets@f2328d558550046fc5449672a6850e848e141590/assets/v6/"

from pyecharts import options as opts
from pyecharts.charts import Bar

# The target distribution, and the spike at the cap. Everything downstream - the choice
# of loss, the transform, the residual plot - is a reaction to this shape.
counts, edges = np.histogram(y_all, bins=50)
centres = [f"{c:.2f}" for c in (edges[:-1] + edges[1:]) / 2]

hist = (
    Bar()
    .add_xaxis(centres)
    .add_yaxis("block groups", [int(c) for c in counts],
               label_opts=opts.LabelOpts(is_show=False),
               category_gap="0%")
    .set_global_opts(
        title_opts=opts.TitleOpts(
            title="California housing: target distribution",
            subtitle="right-skewed, and the bar at 5.00 is the $500,001 census cap"),
        xaxis_opts=opts.AxisOpts(name="median house value ($100k)",
                                 axislabel_opts=opts.LabelOpts(rotate=60, font_size=9)),
        yaxis_opts=opts.AxisOpts(name="rows"),
        tooltip_opts=opts.TooltipOpts(trigger="axis"),
        datazoom_opts=[opts.DataZoomOpts(range_start=0, range_end=100)],
    )
)
hist.render_notebook()
```

## 8. Ridge Regression: the extrapolating floor

Linear least squares with an L2 penalty on the coefficients. On this table it should lose badly, because the signal is mostly in a `Latitude` x `Longitude` x `MedInc` interaction that a linear model cannot express - and that gap is exactly the number worth knowing before you reach for anything more complicated.

Ridge earns its place for a reason no tree can match: it **extrapolates**. Feed it a median income above anything in the training data and it keeps going up; a tree ensemble returns the constant it learned for its rightmost leaf. When the deployment will see feature ranges the training data did not, that difference outweighs several points of $R^2$.

Scaling is required here (the penalty is on the coefficients, so unscaled columns are penalised unequally); trees below need none.

------------------------------------------------------------------------

``` python
from sklearn.linear_model import RidgeCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ridge = Pipeline([
    ("scale", StandardScaler()),
    ("reg", RidgeCV(alphas=np.logspace(-3, 3, 13))),
])

t0 = time.perf_counter()
ridge.fit(X_tr, y_tr)
fit_s = time.perf_counter() - t0

p_ridge = ridge.predict(X_te)
print(f"ridge   fit {fit_s:5.2f}s   alpha {ridge.named_steps['reg'].alpha_:.3f}   "
      f"RMSE {np.sqrt(mean_squared_error(y_te, p_ridge)):.4f}   "
      f"MAE {mean_absolute_error(y_te, p_ridge):.4f}   R2 {r2_score(y_te, p_ridge):.4f}")

coefs = pd.Series(ridge.named_steps["reg"].coef_, index=X_tr.columns).sort_values()
print("\ncoefficients (on standardised features, units of $100k per sd):")
print(coefs.round(3).to_string())

# Trees cannot do this: ask for a block group with income beyond anything seen in training.
extreme = X_te.iloc[[0]].copy()
extreme["MedInc"] = X_tr["MedInc"].max() * 2
print(f"\nMedInc doubled past the training max: ridge says {ridge.predict(extreme)[0]:.2f} "
      f"(vs {ridge.predict(X_te.iloc[[0]])[0]:.2f} for the original row)")

free_memory()
```

    ridge   fit  0.06s   alpha 31.623   RMSE 0.7354   MAE 0.5369   R2 0.5909

    coefficients (on standardised features, units of $100k per sd):
    Latitude     -0.867
    Longitude    -0.838
    AveRooms     -0.256
    AveOccup     -0.030
    Population   -0.008
    HouseAge      0.124
    AveBedrms     0.297
    MedInc        0.831

    MedInc doubled past the training max: ridge says 13.61 (vs 2.27 for the original row)

## 9. Random Forest: the zero-tuning baseline

Deep decorrelated trees, averaged. It picks up the `Latitude` x `Longitude` interaction for free, needs no scaling and essentially no tuning, and on this table it jumps a long way past ridge.

The cost is that each prediction is an average over hundreds of piecewise-constant functions, so the model is **bounded by the training target range in every direction** - it can never predict above 5.0 or below 0.15 here, no matter what the features say. On a capped target that happens to be convenient; on an uncapped one it is a real limitation.

------------------------------------------------------------------------

``` python
from sklearn.ensemble import RandomForestRegressor

rf = RandomForestRegressor(n_estimators=300, min_samples_leaf=2,
                           n_jobs=4, random_state=0)     # 4 vCPU on this box
t0 = time.perf_counter()
rf.fit(X_tr, y_tr)
fit_s = time.perf_counter() - t0

p_rf = rf.predict(X_te)
print(f"random forest   fit {fit_s:5.1f}s   "
      f"RMSE {np.sqrt(mean_squared_error(y_te, p_rf)):.4f}   "
      f"MAE {mean_absolute_error(y_te, p_rf):.4f}   R2 {r2_score(y_te, p_rf):.4f}")
print(f"prediction range [{p_rf.min():.3f}, {p_rf.max():.3f}] vs "
      f"training target range [{y_tr.min():.3f}, {y_tr.max():.3f}] - trees cannot leave the box")

memory_report("after RF")
```

    random forest   fit   5.4s   RMSE 0.5195   MAE 0.3363   R2 0.7958
    prediction range [0.473, 5.000] vs training target range [0.150, 5.000] - trees cannot leave the box
    RAM  after RF         15.55 / 20.97 GB
    VRAM after RF          0.00 GB allocated /  0.00 GB reserved

## 10. HistGradientBoosting: the workhorse

The same histogram-based booster as in the classification notebook, with `loss` now selecting what the model predicts. Three objectives, one fit each, on identical data:

- **`squared_error`** targets the conditional **mean**. Best RMSE, pulled upward by the capped rows.
- **`absolute_error`** targets the conditional **median**. Best MAE, visibly more robust to the cap and the outliers, and slower to fit (the gradient carries less information per step).
- **`poisson`** would suit a count target; on a positive continuous target `gamma` is the usual choice. Neither is right for a capped price, but the one-line switch is the point.

That RMSE and MAE disagree about which fit is best is not a tie to be broken - it is the two metrics correctly reporting on two different estimands.

------------------------------------------------------------------------

``` python
from sklearn.ensemble import HistGradientBoostingRegressor

hgb_fits = {}
for loss in ("squared_error", "absolute_error"):
    m = HistGradientBoostingRegressor(
        loss=loss, max_iter=500, learning_rate=0.1, max_leaf_nodes=31,
        l2_regularization=1.0, early_stopping=True, validation_fraction=0.1,
        n_iter_no_change=25, random_state=0)
    t0 = time.perf_counter()
    m.fit(X_tr, y_tr)
    fit_s = time.perf_counter() - t0
    p = m.predict(X_te)
    hgb_fits[loss] = p
    print(f"HGB {loss:16s} fit {fit_s:5.1f}s ({m.n_iter_:3d} rounds)  "
          f"RMSE {np.sqrt(mean_squared_error(y_te, p)):.4f}  "
          f"MAE {mean_absolute_error(y_te, p):.4f}  R2 {r2_score(y_te, p):.4f}")
    del m
    free_memory()

print("\nsquared_error wins RMSE, absolute_error wins MAE - they are estimating "
      "different things (the mean and the median), not competing at the same thing.")
p_hgb = hgb_fits["squared_error"]
memory_report("after HGB")
```

    HGB squared_error    fit   3.6s (286 rounds)  RMSE 0.4488  MAE 0.2948  R2 0.8476
    HGB absolute_error   fit   5.5s (382 rounds)  RMSE 0.4708  MAE 0.2959  R2 0.8323

    squared_error wins RMSE, absolute_error wins MAE - they are estimating different things (the mean and the median), not competing at the same thing.
    RAM  after HGB        15.66 / 20.97 GB
    VRAM after HGB         0.00 GB allocated /  0.00 GB reserved

## 11. Quantile Regression: shipping an interval instead of a number

Fit the same booster three times with `loss="quantile"` at $\tau = 0.1, 0.5, 0.9$ and you have an 80% prediction interval and a median, with no extra machinery. The interval **widens where the model is unsure**, which is the entire advantage over “point estimate plus a global sigma”: this target is heteroscedastic, so a constant-width band is wrong at both ends.

Two things to check on the result, both computed below:

- **Coverage.** Do ~80% of held-out rows actually land inside? Quantile boosting is fit, not guaranteed, so coverage drifts - typically low, because the quantile fit overfits the training quantiles.
- **Width.** A band wide enough to always cover is useless. Coverage and mean width have to be read together.

When coverage has to be a promise rather than a hope, wrap the point model in **conformal prediction** (`MAPIE`), which converts any regressor’s residuals into intervals with a finite-sample coverage guarantee under exchangeability.

------------------------------------------------------------------------

``` python
QUANTILES = (0.1, 0.5, 0.9)
qpred = {}
for tau in QUANTILES:
    m = HistGradientBoostingRegressor(
        loss="quantile", quantile=tau, max_iter=400, learning_rate=0.1,
        early_stopping=True, n_iter_no_change=25, random_state=0)
    m.fit(X_tr, y_tr)
    qpred[tau] = m.predict(X_te)
    del m
    free_memory()

lo, mid, hi = qpred[0.1], qpred[0.5], qpred[0.9]
# Quantile fits are independent, so they can cross; enforce monotonicity after the fact.
lo, hi = np.minimum(lo, hi), np.maximum(lo, hi)

inside = ((y_te >= lo) & (y_te <= hi)).mean()
print(f"nominal coverage 80%   empirical {inside:.1%}   mean width {np.mean(hi - lo):.3f} "
      f"($100k)  vs target sd {y_te.std():.3f}")


def pinball(y_true, y_pred, tau):
    d = np.asarray(y_true) - np.asarray(y_pred)
    return float(np.maximum(tau * d, (tau - 1) * d).mean())


for tau in QUANTILES:
    print(f"  tau={tau}: pinball {pinball(y_te, qpred[tau], tau):.4f}  "
          f"(the squared-error model scores {pinball(y_te, p_hgb, tau):.4f} on the same loss)")
```

    nominal coverage 80%   empirical 69.6%   mean width 0.834 ($100k)  vs target sd 1.150
      tau=0.1: pinball 0.0635  (the squared-error model scores 0.1513 on the same loss)
      tau=0.5: pinball 0.1480  (the squared-error model scores 0.1474 on the same loss)
      tau=0.9: pinball 0.0862  (the squared-error model scores 0.1435 on the same loss)

``` python
from pyecharts.charts import Line

# 120 test rows sorted by predicted median: the band should breathe, not sit at
# constant width. Where it is wide, the model is telling you it does not know.
order = np.argsort(mid)[:: max(1, len(mid) // 120)][:120]
x = list(range(len(order)))

band = (
    Line()
    .add_xaxis(x)
    .add_yaxis("actual", [round(float(v), 3) for v in np.asarray(y_te)[order]],
               symbol="circle", symbol_size=5, is_smooth=False,
               linestyle_opts=opts.LineStyleOpts(opacity=0),
               label_opts=opts.LabelOpts(is_show=False))
    .add_yaxis("q0.10", [round(float(v), 3) for v in lo[order]], is_smooth=True, symbol="none",
               linestyle_opts=opts.LineStyleOpts(type_="dashed"),
               label_opts=opts.LabelOpts(is_show=False))
    .add_yaxis("q0.50 (median)", [round(float(v), 3) for v in mid[order]], is_smooth=True,
               symbol="none", label_opts=opts.LabelOpts(is_show=False))
    .add_yaxis("q0.90", [round(float(v), 3) for v in hi[order]], is_smooth=True, symbol="none",
               linestyle_opts=opts.LineStyleOpts(type_="dashed"),
               label_opts=opts.LabelOpts(is_show=False))
    .set_global_opts(
        title_opts=opts.TitleOpts(
            title="80% prediction interval, test rows sorted by predicted median",
            subtitle=f"empirical coverage {inside:.1%} of the nominal 80%"),
        xaxis_opts=opts.AxisOpts(type_="value", name="test row (sorted)"),
        yaxis_opts=opts.AxisOpts(name="median house value ($100k)"),
        tooltip_opts=opts.TooltipOpts(trigger="axis"),
        legend_opts=opts.LegendOpts(pos_top="8%"),
    )
)
band.render_notebook()
```

## 12. The Log-Transform Trap

Right-skewed positive targets are usually easier to model on the log scale: `log1p(y)` makes the residuals roughly symmetric, stops the top 1% of rows from writing the model, and turns multiplicative error into additive error.

The trap is coming back. If a model is unbiased on the log scale, then $\mathbb{E}[\log y] = \hat\mu$, and $\exp(\hat\mu) \ne \mathbb{E}[y]$ - by Jensen’s inequality, `expm1` of the mean prediction systematically **under-estimates** the mean. It is a correct estimate of the **median**, which is often what you wanted anyway; if you actually need the mean, apply a smearing correction (Duan 1983) or fit a gamma/Tweedie objective on the raw scale instead.

The cell measures the bias on this table.

------------------------------------------------------------------------

``` python
from sklearn.compose import TransformedTargetRegressor

base = dict(max_iter=500, learning_rate=0.1, early_stopping=True,
            n_iter_no_change=25, random_state=0)

raw = HistGradientBoostingRegressor(**base).fit(X_tr, y_tr)
logged = TransformedTargetRegressor(
    regressor=HistGradientBoostingRegressor(**base),
    func=np.log1p, inverse_func=np.expm1).fit(X_tr, y_tr)

p_raw, p_log = raw.predict(X_te), logged.predict(X_te)
for name, p in (("raw target", p_raw), ("log1p target", p_log)):
    print(f"{name:14s} RMSE {np.sqrt(mean_squared_error(y_te, p)):.4f}  "
          f"MAE {mean_absolute_error(y_te, p):.4f}  "
          f"mean bias {np.mean(p - y_te):+.4f}  "
          f"median bias {np.median(p - y_te):+.4f}")

# Duan's smearing estimator: scale by the mean of exp(residual) on the log scale.
resid_log = np.log1p(y_tr) - logged.regressor_.predict(X_tr)
smear = float(np.mean(np.exp(resid_log)))
p_smeared = np.expm1(logged.regressor_.predict(X_te)) * smear
print(f"\nsmearing factor {smear:.4f}")
print(f"{'log1p + smearing':14s} RMSE {np.sqrt(mean_squared_error(y_te, p_smeared)):.4f}  "
      f"MAE {mean_absolute_error(y_te, p_smeared):.4f}  "
      f"mean bias {np.mean(p_smeared - y_te):+.4f}")
print("\nThe log model is better centred on the median and worse on the mean; smearing "
      "trades some MAE back for a smaller mean bias. Which you want is a business question.")

del raw, logged
free_memory()
```

    raw target     RMSE 0.4508  MAE 0.2976  mean bias +0.0130  median bias +0.0396
    log1p target   RMSE 0.4473  MAE 0.2903  mean bias -0.0163  median bias +0.0261

    smearing factor 1.0043
    log1p + smearing RMSE 0.4470  MAE 0.2909  mean bias -0.0075

    The log model is better centred on the median and worse on the mean; smearing trades some MAE back for a smaller mean bias. Which you want is a business question.

## 13. The production boosters: XGBoost, LightGBM, CatBoost

The production stack. What each adds over `HistGradientBoosting` for regression specifically:

- **XGBoost** - `reg:tweedie` and `survival:aft` objectives, GPU training, and `monotone_constraints` for defensible pricing models.
- **LightGBM** - `huber`, `fair` and `tweedie` objectives, and the fastest fit per unit of accuracy on wide tables.
- **CatBoost** - `RMSEWithUncertainty`, which returns a **mean and a variance from a single fit**, rather than one fit per quantile (section 11 needed three fits to get an interval).

California housing is all-numeric, so unlike the classification notebook there is no categorical encoding to get wrong here and all three take the raw frame.

------------------------------------------------------------------------

``` python
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor

extra_results = {}

xgb = XGBRegressor(n_estimators=500, learning_rate=0.1, max_depth=6,
                   subsample=0.8, colsample_bytree=0.8, tree_method="hist",
                   n_jobs=4, random_state=0)          # device="cuda" to train on the GPU
t0 = time.perf_counter()
xgb.fit(X_tr, y_tr)
extra_results["XGBoost"] = (xgb.predict(X_te), time.perf_counter() - t0)

lgbm = LGBMRegressor(n_estimators=500, learning_rate=0.1, num_leaves=31,
                     n_jobs=4, random_state=0, verbose=-1)
t0 = time.perf_counter()
lgbm.fit(X_tr, y_tr)
extra_results["LightGBM"] = (lgbm.predict(X_te), time.perf_counter() - t0)

cb = CatBoostRegressor(iterations=500, learning_rate=0.1, depth=6,
                       verbose=0, thread_count=4, random_seed=0)
t0 = time.perf_counter()
cb.fit(X_tr, y_tr)
extra_results["CatBoost"] = (cb.predict(X_te), time.perf_counter() - t0)

for name, (p, secs) in extra_results.items():
    print(f"{name:10s} fit {secs:5.1f}s  RMSE {np.sqrt(mean_squared_error(y_te, p)):.4f}  "
          f"MAE {mean_absolute_error(y_te, p):.4f}  R2 {r2_score(y_te, p):.4f}")

# CatBoost gets a mean AND a variance from one fit - compare with the three separate
# quantile fits section 11 needed for the same job.
cb_unc = CatBoostRegressor(iterations=500, learning_rate=0.1, depth=6,
                           loss_function="RMSEWithUncertainty", verbose=0,
                           thread_count=4, random_seed=0).fit(X_tr, y_tr)
mean_var = cb_unc.predict(X_te)
sigma = np.sqrt(np.maximum(mean_var[:, 1], 0))
covered = float(np.mean(np.abs(y_te - mean_var[:, 0]) <= 1.2816 * sigma))
cb_width = float(np.mean(2 * 1.2816 * sigma))
print(f"\nnominal 80% intervals, two ways of getting them:")
print(f"  CatBoost RMSEWithUncertainty (1 fit)  coverage {covered:5.1%}  "
      f"mean width {cb_width:.3f}")
print(f"  HGB quantile regression (3 fits)      coverage {inside:5.1%}  "
      f"mean width {np.mean(hi - lo):.3f}")
print("\nComparable width, comparable coverage, a third of the fitting - and both fall short "
      "of\nthe nominal 80%, which is the standing argument for conformal prediction "
      "(section 16)\nover trusting any model's own uncertainty.")

del xgb, lgbm, cb, cb_unc
free_memory()
```

    XGBoost    fit   0.8s  RMSE 0.4443  MAE 0.2898  R2 0.8507
    LightGBM   fit   0.3s  RMSE 0.4447  MAE 0.2919  R2 0.8504
    CatBoost   fit   0.6s  RMSE 0.4503  MAE 0.2984  R2 0.8466

    nominal 80% intervals, two ways of getting them:
      CatBoost RMSEWithUncertainty (1 fit)  coverage 73.6%  mean width 0.839
      HGB quantile regression (3 fits)      coverage 69.6%  mean width 0.834

    Comparable width, comparable coverage, a third of the fitting - and both fall short of
    the nominal 80%, which is the standing argument for conformal prediction (section 16)
    over trusting any model's own uncertainty.

## 14. Head-to-head Benchmark

Same split, same test rows, every metric computed the same way - five sklearn models refit here plus the three boosters from section 13, eight in total. Read it as: **RMSE** if large errors hurt superlinearly, **MAE** if they do not, **p90 absolute error** because that is what an SLA is written against, and **fit time** because that is what a retrain costs.

Hardware: knowledge-lab (4 vCPU, 20 GB RAM); 15,480 training rows, 5,160 test rows. The target is in units of \$100k, so an MAE of 0.30 is about \$30,000 - always translate the metric back into the target’s units before deciding a model is good.

One table is a smoke test. A real comparison runs the whole OpenML-CTR23 suite, because between-table variance dwarfs between-model variance.

------------------------------------------------------------------------

``` python
from sklearn.neural_network import MLPRegressor

MODELS = {
    "Ridge": lambda: Pipeline([("scale", StandardScaler()),
                               ("reg", RidgeCV(alphas=np.logspace(-3, 3, 13)))]),
    "RandomForest": lambda: RandomForestRegressor(n_estimators=300, min_samples_leaf=2,
                                                  n_jobs=4, random_state=0),
    "HGB (squared)": lambda: HistGradientBoostingRegressor(
        max_iter=500, learning_rate=0.1, early_stopping=True,
        n_iter_no_change=25, random_state=0),
    "HGB (absolute)": lambda: HistGradientBoostingRegressor(
        loss="absolute_error", max_iter=500, learning_rate=0.1,
        early_stopping=True, n_iter_no_change=25, random_state=0),
    "MLP (128, 64)": lambda: Pipeline([
        ("scale", StandardScaler()),
        ("reg", MLPRegressor(hidden_layer_sizes=(128, 64), alpha=1e-3, batch_size=256,
                             early_stopping=True, max_iter=200, random_state=0))]),
}

preds, rows = {}, []
for name, make in MODELS.items():
    model = make()
    t0 = time.perf_counter()
    model.fit(X_tr, y_tr)
    fit_s = time.perf_counter() - t0
    p = model.predict(X_te)
    preds[name] = p
    err = np.abs(p - y_te)
    rows.append(dict(model=name,
                     rmse=float(np.sqrt(mean_squared_error(y_te, p))),
                     mae=float(mean_absolute_error(y_te, p)),
                     r2=float(r2_score(y_te, p)),
                     p90_abs_err=float(np.percentile(err, 90)),
                     max_abs_err=float(err.max()),
                     fit_s=fit_s))
    del model                 # free each model before fitting the next
    free_memory()

for name, (p, s) in extra_results.items():
    preds[name] = p
    err = np.abs(p - y_te)
    rows.append(dict(model=name, rmse=float(np.sqrt(mean_squared_error(y_te, p))),
                     mae=float(mean_absolute_error(y_te, p)), r2=float(r2_score(y_te, p)),
                     p90_abs_err=float(np.percentile(err, 90)),
                     max_abs_err=float(err.max()), fit_s=s))

bench = pd.DataFrame(rows).sort_values("rmse").reset_index(drop=True)
memory_report("after benchmark")
bench.round(4)
```

    RAM  after benchmark  15.77 / 20.97 GB
    VRAM after benchmark   0.00 GB allocated /  0.00 GB reserved

|     | model          | rmse   | mae    | r2     | p90_abs_err | max_abs_err | fit_s  |
|-----|----------------|--------|--------|--------|-------------|-------------|--------|
| 0   | XGBoost        | 0.4443 | 0.2898 | 0.8507 | 0.6636      | 3.4892      | 0.8294 |
| 1   | LightGBM       | 0.4447 | 0.2919 | 0.8504 | 0.6831      | 3.2500      | 0.2941 |
| 2   | CatBoost       | 0.4503 | 0.2984 | 0.8466 | 0.6636      | 3.4181      | 0.6472 |
| 3   | HGB (squared)  | 0.4508 | 0.2976 | 0.8463 | 0.6775      | 3.3666      | 0.6342 |
| 4   | HGB (absolute) | 0.4710 | 0.2950 | 0.8322 | 0.6916      | 3.8750      | 0.8998 |
| 5   | RandomForest   | 0.5195 | 0.3363 | 0.7958 | 0.7898      | 3.5245      | 3.6860 |
| 6   | MLP (128, 64)  | 0.5390 | 0.3671 | 0.7802 | 0.8202      | 5.0887      | 2.2512 |
| 7   | Ridge          | 0.7354 | 0.5369 | 0.5909 | 1.0862      | 5.8691      | 0.0534 |

``` python
names = bench["model"].tolist()
bar = (
    Bar()
    .add_xaxis(names)
    .add_yaxis("RMSE", [round(float(v), 4) for v in bench["rmse"]])
    .add_yaxis("MAE", [round(float(v), 4) for v in bench["mae"]])
    .add_yaxis("p90 |error|", [round(float(v), 4) for v in bench["p90_abs_err"]])
    .set_series_opts(label_opts=opts.LabelOpts(is_show=False))
    .set_global_opts(
        title_opts=opts.TitleOpts(title="California housing: error by model",
                                  subtitle="5,160 held-out rows, units of $100k (lower is better)"),
        xaxis_opts=opts.AxisOpts(axislabel_opts=opts.LabelOpts(rotate=20)),
        yaxis_opts=opts.AxisOpts(name="error ($100k)"),
        tooltip_opts=opts.TooltipOpts(trigger="axis"),
        legend_opts=opts.LegendOpts(pos_top="8%"),
    )
)
bar.render_notebook()
```

``` python
from pyecharts.charts import Scatter

# Accuracy against retrain cost - the axis a production pick is actually made on.
scatter = Scatter()
scatter.add_xaxis([round(float(s), 2) for s in bench["fit_s"]])
for name, rmse, fits in zip(bench["model"], bench["rmse"], bench["fit_s"]):
    scatter.add_yaxis(name, [[round(float(fits), 2), round(float(rmse), 4)]],
                      symbol_size=18, label_opts=opts.LabelOpts(is_show=False))
scatter.set_global_opts(
    title_opts=opts.TitleOpts(title="RMSE vs fit time",
                              subtitle="down and to the left is better"),
    xaxis_opts=opts.AxisOpts(type_="value", name="fit seconds (4 vCPU)"),
    yaxis_opts=opts.AxisOpts(type_="value", name="RMSE ($100k)", min_="dataMin"),
    tooltip_opts=opts.TooltipOpts(trigger="item", formatter="{a}: {c}"),
    legend_opts=opts.LegendOpts(pos_top="8%"),
)
scatter.render_notebook()
```

## 15. Residuals: where the aggregate number hides the failure

RMSE is one number over 5,160 rows. The residual plot is the diagnostic that tells you *which* rows, and on this table it shows three things at once:

- **A hard diagonal edge at the top.** Every row capped at 5.00001 can only be under-predicted, so the residuals for those rows lie exactly on a line. That is a data artefact, not a model failure, and no amount of tuning removes it.
- **A funnel.** Error grows with the predicted value - textbook heteroscedasticity, and the justification for the quantile intervals in section 11 over a constant-width band.
- **A floor effect** at the bottom of the range, where the model cannot go below the cheapest training block group.

The second chart bins absolute error by predicted value, which makes the funnel legible as a curve rather than a cloud.

------------------------------------------------------------------------

``` python
best_name = bench.iloc[0]["model"]
p_best = preds[best_name]
resid = np.asarray(y_te) - p_best

# Thin to ~2,000 points: ECharts is happy, the shape is unchanged.
sel = np.linspace(0, len(p_best) - 1, min(2000, len(p_best))).astype(int)
resid_scatter = (
    Scatter()
    .add_xaxis([round(float(v), 3) for v in p_best[sel]])
    .add_yaxis("residual", [[round(float(p_best[i]), 3), round(float(resid[i]), 3)] for i in sel],
               symbol_size=4, label_opts=opts.LabelOpts(is_show=False))
    .set_global_opts(
        title_opts=opts.TitleOpts(
            title=f"Residuals vs prediction ({best_name})",
            subtitle="the straight upper edge is the $500,001 cap; the funnel is heteroscedasticity"),
        xaxis_opts=opts.AxisOpts(type_="value", name="predicted ($100k)"),
        yaxis_opts=opts.AxisOpts(type_="value", name="actual - predicted ($100k)"),
        tooltip_opts=opts.TooltipOpts(trigger="item"),
    )
)
print(f"{best_name}: residual mean {resid.mean():+.4f} (bias), sd {resid.std():.4f}")
print(f"rows on the cap: mean residual {resid[np.asarray(y_te) >= 5.0].mean():+.4f} "
      f"- systematically under-predicted, by construction")
resid_scatter.render_notebook()
```

    XGBoost: residual mean -0.0099 (bias), sd 0.4442
    rows on the cap: mean residual +0.5034 - systematically under-predicted, by construction

``` python
# The funnel as a curve: mean absolute error inside deciles of the prediction.
bins = np.quantile(p_best, np.linspace(0, 1, 11))
bins[-1] += 1e-9
idx = np.clip(np.digitize(p_best, bins) - 1, 0, 9)

funnel = Line().add_xaxis([f"{(bins[b] + bins[b + 1]) / 2:.2f}" for b in range(10)])
for name, p in preds.items():
    e = np.abs(np.asarray(y_te) - p)
    funnel.add_yaxis(name, [round(float(e[idx == b].mean()), 4) for b in range(10)],
                     is_smooth=True, symbol="circle", symbol_size=6,
                     label_opts=opts.LabelOpts(is_show=False))
funnel.set_global_opts(
    title_opts=opts.TitleOpts(title="MAE by decile of predicted value",
                              subtitle="error grows with the target - a constant-width interval is wrong at both ends"),
    xaxis_opts=opts.AxisOpts(name="predicted value ($100k)"),
    yaxis_opts=opts.AxisOpts(name="mean absolute error ($100k)"),
    tooltip_opts=opts.TooltipOpts(trigger="axis"),
    legend_opts=opts.LegendOpts(pos_top="8%"),
)
funnel.render_notebook()
```

## 16. Common Frameworks

Regression shares the tabular stack with classification, so the rows below emphasise what changes when the target is continuous: the objective has to match the target’s shape, the output should usually be an interval rather than a number, and extrapolation becomes a live failure mode that no classification model has. Those three concerns account for most of the specialised tooling here.

| Framework | Layer | What it gives you | License | Reach for it when |
|----|----|----|----|----|
| [scikit-learn](https://github.com/scikit-learn/scikit-learn) | modelling | Ridge, forests, `HistGradientBoostingRegressor` with `loss="quantile"`/`"poisson"`, `Pipeline`, and the CV splitters | BSD-3 | Default. The objective switch of section 8 is a one-line change here and moves the metric more than tuning does |
| [XGBoost](https://github.com/dmlc/xgboost) / [LightGBM](https://github.com/microsoft/LightGBM) / [CatBoost](https://github.com/catboost/catboost) | modelling | Tweedie, gamma, Poisson and Huber objectives, monotone constraints, and LightGBM’s `linear_tree=True` | Apache 2.0 / MIT | Counts, insurance-style zero-inflated targets, or anywhere large errors are catastrophic and MSE is the wrong loss |
| [MAPIE](https://github.com/scikit-learn-contrib/MAPIE) | modelling | Conformal prediction on top of any fitted estimator - intervals with a coverage guarantee under exchangeability | BSD-3 | Coverage has to be a promise rather than a hope. Section 11’s quantiles are the cheap version; this is the one with a proof |
| [AutoGluon](https://github.com/autogluon/autogluon) / [TabPFN](https://github.com/PriorLabs/TabPFN) | modelling | Stacked ensembles for one line of code, and - below ~10k rows - a forward-pass regressor that returns a full predictive distribution | Apache 2.0 | A strong baseline fast. TabPFN is a vendor package, so it is prose here, but the small-table results are real |
| [pandas](https://github.com/pandas-dev/pandas) / [polars](https://github.com/pola-rs/polars) | data | Loading, joins, and feature construction that fits in 20 GB | BSD-3 / MIT | Always |
| [Optuna](https://github.com/optuna/optuna) | data | Successive-halving search over the booster’s hyperparameters, scored by cross-validated MAE or pinball loss | MIT | Tuning. Search the regularisation and leaf size before anything else |
| [ONNX Runtime](https://onnxruntime.ai/) / [Treelite](https://github.com/dmlc/treelite) | inference runtime | The ensemble compiled to a portable, microsecond-latency artefact | MIT / Apache 2.0 | Deployment inside an existing service, which is where tabular models usually live |
| [MLflow](https://github.com/mlflow/mlflow) / [Feast](https://github.com/feast-dev/feast) | serving | Registry, versioning, and identical feature computation offline and online | Apache 2.0 | Production. Feature skew between training and serving is the classic silent regression |
| [SHAP](https://github.com/shap/shap) + `sklearn.inspection` | evaluation | Per-prediction attributions, partial dependence and ICE curves, and `monotone_constraints` to turn “more income never lowers the estimate” into a guarantee | MIT / BSD-3 | Always in regulated domains, and whenever a residual plot (section 15) shows structure you need to explain |

**The 2026 default stack** is polars plus a scikit-learn `Pipeline` around LightGBM with an objective chosen to match the target, quantile heads or MAPIE for intervals, Optuna to tune, SHAP to explain, and ONNX to deploy. Nothing here needs a GPU.

The common wrong turn is optimising a point prediction when the consumer needs an interval. Inventory, capacity, pricing and staffing all consume a quantile, and a model that reports only a mean has thrown away the information the decision depends on. The second is ignoring extrapolation: a tree ensemble silently flatlines outside its training range, so either blend in a linear component or monitor for out-of-range inputs and refuse to score them.

------------------------------------------------------------------------

## 17. Going Further

**Match the objective to the target’s shape before tuning anything.** Counts want Poisson; positive skewed magnitudes want gamma; insurance-style mass-at-zero-plus-tail wants Tweedie; “large errors are catastrophic” wants Huber, not MSE. This is a one-line change and it moves the metric more than a hyperparameter search.

**Ship intervals, not points.** Fit quantiles as in section 11, then check coverage. When coverage has to be a guarantee, use conformal prediction: `pip install mapie`, then `MapieRegressor(estimator=model).fit(X, y).predict(X_test, alpha=0.2)` gives 80% intervals that keep their promise under exchangeability, on top of whatever point model you already trust.

**Try TabPFN v2 on small tables.** `pip install tabpfn`, then `TabPFNRegressor().fit(X, y)` - no gradient steps, and it returns a full predictive distribution rather than a point. Below ~10k rows it is often the best single model available. It is a vendor package rather than a general-purpose library, so it is not a dependency here (see the general-purpose-library rule in `CLAUDE.md`).

**Watch for extrapolation.** If the deployment will see feature values outside the training range, a tree ensemble silently flatlines. Either add a linear component (a two-model blend, or LightGBM’s `linear_tree=True`), or monitor for out-of-range inputs and refuse to score them.

**Get the split right.** Group by entity when rows repeat; split by time when the table has a time column. A random split on temporally ordered rows is the most common way a regression model looks excellent offline and fails in production.

**Explain the predictions.** `shap.TreeExplainer` is exact and fast for tree ensembles; partial dependence and ICE curves (`sklearn.inspection`) show the shape of each feature’s effect; `monotone_constraints` (XGBoost/LightGBM) turn “more income should never lower the estimate” from a hope into a guarantee.

**Related notebooks in this repo:** `00_Tabular_Classification` (the same pipeline with a discrete target), `02_Time_Series_Forecasting` (when the rows are ordered in time), and `Other/00_Graph_Machine_Learning` (when the rows are connected to each other).

------------------------------------------------------------------------

Back to top
