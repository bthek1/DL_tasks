# Time Series Forecasting

Predicting the future of an ordered series: why a seasonal-naive baseline beats most deep models, what the 2024-2026 time-series foundation models actually changed, the metrics that survive scale differences, and runnable code that puts TimesFM zero-shot against a PatchTST trained from scratch on the same ETTh1 windows.

Author

Benedict Thekkel

## 1. What is Time Series Forecasting?

Time series forecasting maps a **history** of one or more ordered numeric series to their **future values**. The ordering is the whole point: rows are not exchangeable, the test set is always later than the training set, and the same value means something different depending on where it sits.

**Input.** A **context window** of $L$ past observations, usually at a fixed frequency (hourly, daily), often multivariate ($C$ channels observed together), and often with side information:

- **Past covariates** - things you only know retrospectively (actual temperature, realised traffic).
- **Future covariates** - things you know ahead of time (calendar, holidays, scheduled promotions, published price). These are usually the highest-value features in the whole model.
- **Static covariates** - properties of the series itself (store size, product category, region).

**Output.** A **horizon** of $H$ future values. The interesting question is what shape:

| Output | What it is | When you need it |
|----|----|----|
| Point forecast | One number per future step | Simple reporting |
| Quantile forecast | e.g. q10 / q50 / q90 per step | Inventory, capacity, bidding - the usual real deliverable |
| Full distribution | Parametric or sample paths | Risk, simulation, anything that composes forecasts |
| Direct multi-step | All $H$ steps in one forward pass | The modern default: no error accumulation |
| Recursive / autoregressive | Feed each prediction back in | Flexible horizon, but errors compound |

**Neighbouring tasks:**

| Task | What it does | Typical tool |
|----|----|----|
| Time series classification | Label a whole window (activity, fault type) | ROCKET, InceptionTime, PatchTST encoder |
| Anomaly detection | Flag windows that do not fit the pattern | Matrix profile, forecast residuals |
| Changepoint detection | Find where the regime shifted | `ruptures`, BOCPD |
| Imputation | Fill missing observations inside a series | PatchTST/TimesNet masked reconstruction |
| Tabular regression | Same target type, but rows are exchangeable | see `01_Tabular_Regression` |
| Panel / hierarchical forecasting | Thousands of related series that must sum consistently | `hierarchicalforecast`, reconciliation |

------------------------------------------------------------------------

## 2. Real-World Use Cases

| Use case | Domain | Consumes / produces | Dominant constraint |
|----|----|----|----|
| Retail demand planning | Retail, e-commerce (Amazon, Walmart) | Millions of SKU x store series + promotions -\> weekly units | Scale (millions of series), intermittency (mostly zeros), hierarchy that must reconcile |
| Electricity load and price | Utilities, energy trading | Historical load, weather forecast, calendar -\> MW / price per hour | Quantiles feed the bid; weather forecast error dominates; hard operational deadline |
| Capacity and autoscaling | Cloud infrastructure | Request rate, CPU, queue depth -\> next 30-60 min | Asymmetric cost (under-provision is an outage); minutes-fresh retraining |
| Financial forecasting | Trading, treasury | Prices, volumes, order flow -\> return or volatility | Signal-to-noise near zero; non-stationarity; a naive baseline is genuinely hard to beat |
| Inventory and supply chain | Manufacturing, logistics | Shipments, lead times, orders -\> demand over lead time | The *distribution* is the product (safety stock is a quantile), not the mean |
| Healthcare capacity | Hospitals, public health | Admissions, ED arrivals, epidemic counts -\> beds, staff | Regime changes (a new variant) break every model trained before them |
| Predictive maintenance | Industry, aviation | Sensor streams -\> degradation trajectory | Very long horizons, few failure examples, multivariate |
| Web and app traffic | Any online product | Sessions per hour -\> capacity, ad inventory | Multiple seasonalities at once (daily + weekly + annual) plus events |

**What the benchmark number hides.**

- **A naive baseline is not a formality, it is the bar.** Seasonal naive (repeat last week) beats a surprising number of published deep models on real data. The M4 and M5 competitions were both won by statistical/hybrid approaches, and the “are transformers effective for time series?” paper (Zeng et al., 2023) showed a **single linear layer** (DLinear) matching or beating a whole generation of transformer forecasters on the standard long-horizon benchmarks. Report the naive baseline or the number means nothing.
- **Non-stationarity is the actual problem.** Real series shift level, variance and seasonality. A model validated on a fixed split silently assumes tomorrow resembles last year. Use rolling origin evaluation (backtesting), not a single split.
- **Leakage is easy and invisible.** Scaling with statistics computed over the whole series, imputing with future values, or using a covariate that will not be available at forecast time - each of these produces a beautiful backtest and a broken deployment.
- **Point forecasts are usually the wrong deliverable.** Safety stock is a quantile; capacity is a quantile; a bid curve is a distribution. A model that gives a good mean and no uncertainty cannot be used for the decisions people actually make.
- **Aggregation changes everything.** Hourly is noisy, daily is manageable, weekly is easy. A lot of impressive accuracy comes from forecasting a more aggregated series than the decision needs.
- **Intermittency breaks the standard metrics.** Retail demand is mostly zeros; MAPE is undefined, RMSE rewards forecasting zero, and specialist methods (Croston, Tweedie objectives) exist for precisely this.

------------------------------------------------------------------------

## 3. How Modern Forecasting Works

**1. Classical statistical models (1970-2010, still competitive).** ARIMA (Box-Jenkins), exponential smoothing / ETS, and Theta. Per-series, interpretable, tiny, and genuinely hard to beat on short univariate series. ETS and Theta variants have won or nearly won competitions as recently as M4 (2018).

**2. Gradient-boosted trees on lag features (2015-2020).** Turn the series into a table (lags, rolling means, calendar dummies) and fit LightGBM. **This won M5 (2020)** and is still the default in most retail companies: it handles thousands of series in one model, absorbs covariates naturally, and trains in minutes. Its limit is that trees cannot extrapolate a trend, so the series usually has to be differenced or detrended first.

**3. Deep global models (2017-2021).** DeepAR (2017, autoregressive RNN with a probabilistic head), N-BEATS (2019, deep residual stack of basis expansions), TFT (2019, attention plus gating plus covariate selection). The change here was **global** training: one model over thousands of series instead of one model per series, which is where the deep approach starts paying.

**4. Transformers, then the backlash, then patching (2021-2023).** Informer, Autoformer and FEDformer attacked the quadratic attention cost for long horizons. Then Zeng et al. (2023) showed a one-layer linear model (DLinear) beat all of them on the same benchmarks, which was a genuine crisis. The resolution was **patching**: **PatchTST** (2023) splits the series into subsequence patches (as ViT does with images) and keeps channels independent, which fixed both the compute cost and the overfitting. **PatchTSMixer** (2023) replaced attention with MLP-mixing and matched it more cheaply. Both are transformers-native and used below.

**5. Time series foundation models (2023 -\> now), the current frontier.** Pretrain one model on billions of time points from many domains, then forecast a series it has never seen **zero-shot**:

- **TimesFM** (Google, 2024; v2.0 500M, v2.5 200M) - decoder-only patched transformer, 100B+ real-world time points, transformers-native. **Used below.**
- **Chronos / Chronos-Bolt** (Amazon, 2024-2025) - tokenise scaled values and reuse a T5 language model; Bolt is the faster direct-multi-step successor.
- **Moirai** (Salesforce, 2024) - any-variate attention, multiple patch sizes, probabilistic.
- **Lag-Llama**, **TimeGPT** (Nixtla, commercial API), **Tiny Time Mixers** (IBM, ~1M params and competitive, which is the interesting part).

**Where that leaves you in 2026.** A foundation model gives a strong forecast with no training at all, which is a real change for cold starts and for the long tail of series nobody was going to model by hand. It does **not** yet reliably beat a well-fit local model on a series you have plenty of history for, and it cannot use your covariates as naturally as LightGBM can. The honest default is: **baseline first, then LightGBM on lags if you have covariates, then a foundation model zero-shot as a fast second opinion, then a trained PatchTST if you have many long series and the horizon is long.**

------------------------------------------------------------------------

## 4. Evaluation Metrics

Forecasting metrics exist mostly to solve one problem: **series live on different scales**, so an error of 5 is excellent for one and catastrophic for another.

**Scale-dependent** (fine within one series, meaningless across series):

$$\text{MAE} = \frac{1}{H}\sum_{t} \lvert y_t - \hat y_t \rvert, \qquad
  \text{RMSE} = \sqrt{\frac{1}{H}\sum_{t} (y_t - \hat y_t)^2}$$

**Percentage-based** (comparable, but fragile):

- **MAPE** breaks at $y = 0$ and punishes over-prediction more than under-prediction.
- **sMAPE** is bounded but still unstable near zero, and its asymmetry is subtler rather than gone.

**Scaled** (the ones the competitions settled on):

$$\text{MASE} = \frac{\text{MAE of the forecast}}{\text{MAE of a seasonal naive forecast on the training data}}$$

**MASE \< 1 means you beat seasonal naive; MASE \> 1 means you did not.** It is defined at zero, it is symmetric, and it has a built-in sanity check, which is why M4/M5 used it (with **RMSSE**, the squared version, in M5).

**Probabilistic** (for the deliverable that is usually wanted):

- **Pinball / quantile loss** at each required quantile - the same loss used to fit them.
- **Weighted quantile loss (WQL)** - the mean pinball loss over a quantile grid, normalised by total demand. The standard for foundation-model benchmarks.
- **CRPS** - the continuous ranked probability score, the limit of averaged pinball loss over all quantiles. The right scoring rule for a full predictive distribution.
- **Coverage** - do 80% intervals contain the truth 80% of the time?

**Pitfalls.** Average metrics **across windows and series**, not over a single window. Use **rolling-origin backtesting** rather than one split. Never compute normalisation statistics on the test period. And always print the seasonal-naive number next to yours.

------------------------------------------------------------------------

``` python
import numpy as np


def mase(y_true, y_pred, y_train, season=24):
    "Mean absolute scaled error. < 1 beats seasonal naive on the training history."
    naive_mae = np.mean(np.abs(y_train[season:] - y_train[:-season]))
    return float(np.mean(np.abs(y_true - y_pred)) / naive_mae)


def smape(y_true, y_pred):
    "Symmetric MAPE, in percent. Still unstable when y is near zero."
    denom = (np.abs(y_true) + np.abs(y_pred)) / 2
    return float(np.mean(np.abs(y_true - y_pred) / np.maximum(denom, 1e-9)) * 100)


def pinball(y_true, y_pred, tau):
    "Quantile loss at level tau."
    d = np.asarray(y_true) - np.asarray(y_pred)
    return float(np.mean(np.maximum(tau * d, (tau - 1) * d)))


def wql(y_true, quantile_preds, taus):
    "Weighted quantile loss: mean pinball over a quantile grid, normalised by total actual."
    total = np.sum(np.abs(y_true))
    return float(2 * sum(pinball(y_true, q, t) for q, t in zip(quantile_preds, taus))
                 * len(y_true) / (len(taus) * total))


# A toy daily series: weekly seasonality + trend + noise.
rng = np.random.default_rng(0)
t = np.arange(400)
y = 100 + 0.1 * t + 10 * np.sin(2 * np.pi * t / 7) + rng.normal(0, 3, 400)
train, test = y[:365], y[365:]

seasonal_naive = np.tile(train[-7:], 5)[: len(test)]
flat_naive = np.full(len(test), train[-1])

for name, pred in [("seasonal naive", seasonal_naive), ("last value", flat_naive)]:
    print(f"{name:16s} MAE {np.mean(np.abs(test - pred)):6.2f}  "
          f"sMAPE {smape(test, pred):5.2f}%  MASE {mase(test, pred, train, season=7):5.3f}")

print("\nMASE is the useful one: it says directly whether the model earned its complexity.")
```

    seasonal naive   MAE   3.25  sMAPE  2.37%  MASE 0.921
    last value       MAE   6.68  sMAPE  4.85%  MASE 1.892

    MASE is the useful one: it says directly whether the model earned its complexity.

## 5. Datasets

Long-horizon forecasting research converged on a small set of tables, which is both convenient and a known weakness (they are few, they are all infrastructure/weather, and results on them do not always transfer).

| Dataset | Contents | Size | Frequency | License | Typical use |
|----|----|----|----|----|----|
| [ETT (ETTh1/h2, ETTm1/m2)](https://github.com/zhouhaoyi/ETDataset) | Electricity transformer load + oil temperature, 7 channels | 17,420 hourly rows (h1) | hourly / 15 min | CC BY-NC 4.0 (research use) | The standard long-horizon benchmark - **used by this notebook** |
| [Electricity (LD2011_2014)](https://archive.ics.uci.edu/dataset/321) | Consumption for 321 clients | 26k x 321 | hourly | CC BY 4.0 | Multivariate scaling |
| [Traffic](https://pems.dot.ca.gov/) | Road occupancy, 862 sensors | 17k x 862 | hourly | Public | Very wide multivariate |
| [Weather](https://www.bgc-jena.mpg.de/wetter/) | 21 meteorological channels | 52k rows | 10 min | CC BY 4.0 | High frequency, many channels |
| [M4](https://github.com/Mcompetitions/M4-methods) | 100,000 series across domains | 100k series | yearly to hourly | Open | The classic many-series competition benchmark |
| [M5](https://www.kaggle.com/c/m5-forecasting-accuracy) | Walmart unit sales, hierarchical | 42,840 series | daily | Competition terms | Intermittent demand, hierarchy, covariates |
| [Monash TSF Archive](https://forecastingdata.org/) | ~30 curated datasets in one format | Mixed | Mixed | Per-dataset | The standard “many datasets” evaluation |
| [GIFT-Eval](https://huggingface.co/spaces/Salesforce/GIFT-Eval) | 24 datasets, 144k series, 7 domains, unified | Mixed | Mixed | Per-dataset | The current foundation-model leaderboard |

**This notebook uses ETTh1** - the `OT` (oil temperature) channel for the univariate sections and all 7 channels for the multivariate PatchTST. It is 2.5 MB of CSV, downloaded once into `DL_tasks/datasets/` (gitignored), covering July 2016 to June 2018 hourly. It has daily and weekly seasonality, a visible trend, and enough non-stationarity to make the naive baseline respectable. Note the **CC BY-NC** license: research and teaching are fine, commercial use is not. Nothing here is gated.

------------------------------------------------------------------------

## 6. The Model Landscape (mid-2026)

| Model | Params | License | Trained per series? | Probabilistic | Covariates | Best for |
|----|----|----|----|----|----|----|
| Seasonal naive | 0 | \- | \- | no | no | **The bar every model must clear** |
| ETS / ARIMA (`statsforecast`) | ~10 | Apache 2.0 | per series | yes | limited | Short univariate series, few of them |
| LightGBM on lag features | ~100s of trees | MIT | one global model | via quantile objectives | **excellent** | Many series with rich covariates (won M5) |
| DeepAR | ~1M | Apache 2.0 (GluonTS) | global | yes | yes | Probabilistic, many related series |
| N-BEATS / N-HiTS | ~10-30M | Apache 2.0 | global | via ensembling | limited | Pure univariate accuracy |
| DLinear | ~1k | Apache 2.0 | global | no | no | **The baseline that embarrassed the transformers** |
| **PatchTST** | ~1-10M | Apache 2.0 | global | via quantile head | limited | Long-horizon multivariate - **trained below** |
| PatchTSMixer | ~0.5-5M | Apache 2.0 | global | via quantile head | yes | Same, cheaper, no attention |
| TFT | ~10M | Apache 2.0 | global | yes | **excellent** | Interpretable, covariate-heavy |
| **TimesFM 2.0** | 500M | Apache 2.0 | **zero-shot** | quantiles | no (v2.0) | **Cold start, no training - run below** |
| Chronos-Bolt | 9M-200M | Apache 2.0 | zero-shot | quantiles | no | Fast zero-shot, small footprint |
| Moirai / Moirai-MoE | 14M-311M | Apache 2.0 | zero-shot | yes | past covariates | Any-variate zero-shot |
| Tiny Time Mixers (TTM) | ~1M | Apache 2.0 | zero-shot + fine-tune | no | yes | Extremely cheap zero-shot; needs `tsfm_public` |
| TimeGPT | undisclosed | commercial API | zero-shot | yes | yes | Managed service |

Leaderboards: **[GIFT-Eval](https://huggingface.co/spaces/Salesforce/GIFT-Eval)** (the current foundation-model standard), the [Monash archive](https://forecastingdata.org/), and the long-horizon ETT/Electricity/Traffic tables reported in the PatchTST line of papers.

**What wins what.** On zero-shot accuracy over unseen series, TimesFM 2.x and Chronos-Bolt lead. On a series you have years of history for, a trained local or global model still usually wins. On throughput, DLinear and TTM are orders of magnitude cheaper than anything else in the table and are frequently within a few percent. On covariates - which is what most business forecasting actually needs - LightGBM and TFT are still ahead of every foundation model.

**Download budget.** `google/timesfm-2.0-500m-pytorch` is a 4 GB repo (2 GB safetensors plus a 2 GB legacy `.ckpt`); only the safetensors are fetched, so about 2 GB lands on disk - inside the ~8 GB cap in `CLAUDE.md`. `google/timesfm-2.5-200m-pytorch` is smaller if you are tight on space.

------------------------------------------------------------------------

## 7. Setup

Package roles:

- `transformers` (\>=5.13) + `torch` - TimesFM (zero-shot) and PatchTST (trained here), both transformers-native; no vendor forecasting packages
- `pandas` / `numpy` - the series and the windowing
- `pyecharts` - all charts (repo rule)
- `accelerate` - device placement for the 500M checkpoint

The ETTh1 CSV (2.5 MB) and the Hugging Face cache both land in `DL_tasks/datasets/`, which is gitignored. TimesFM 2.0 in fp32 is about 2 GB of VRAM, and PatchTST as configured here is 0.1M parameters, so the whole notebook fits inside the 12 GB card with room to spare - but the models are still freed between sections, because that is the house rule.

------------------------------------------------------------------------

``` python
# TimesFM and PatchTST are both transformers-native - no vendor forecasting packages.
# %pip install -q torch transformers accelerate pandas pyecharts
```

``` python
import ctypes
import ctypes.util
import gc
import time
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd
import psutil
import torch
from dotenv import find_dotenv, load_dotenv

# Knowledge/.env sets HF_TOKEN - authenticated Hub requests get higher rate limits
load_dotenv(find_dotenv(usecwd=True))

device = "cuda:0" if torch.cuda.is_available() else "cpu"
if device != "cpu":
    print(torch.cuda.get_device_name(0))
print("device:", device)


def vram(tag=""):
    "Report current GPU memory (allocated / reserved). No-op on CPU."
    if torch.cuda.is_available():
        alloc = torch.cuda.memory_allocated() / 1e9
        reserved = torch.cuda.memory_reserved() / 1e9
        print(f"VRAM {tag:18s} {alloc:5.2f} GB allocated / {reserved:5.2f} GB reserved")


def free_memory(*objs):
    "Delete objects, run GC, empty the CUDA cache, and return freed RAM to the OS.\n\n    Note the argument list only drops *this function's* references. A name bound in\n    the notebook still holds the object, so the working idiom is `del model;\n    free_memory()` at the call site - which is what every section below uses.\n    "
    for o in objs:
        del o
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
    # glibc keeps freed CPU allocations in its arenas, so RSS ratchets upward across
    # sections; malloc_trim(0) hands them back. Not optional on a 20 GB box - see
    # dl-visualization-and-memory.md.
    try:
        ctypes.CDLL(ctypes.util.find_library("c") or "libc.so.6").malloc_trim(0)
    except Exception:
        pass


def memory_report(tag=""):
    "Print current system RAM (and VRAM if a GPU is present), in GB."
    vm = psutil.virtual_memory()
    print(f"RAM  {tag:18s} {(vm.total - vm.available) / 1e9:5.2f} / {vm.total / 1e9:5.2f} GB")
    vram(tag)


DATA_DIR = Path("../../datasets")          # gitignored
DATA_DIR.mkdir(exist_ok=True)
HF_CACHE = str(DATA_DIR / "hf_cache")

ETT_CSV = DATA_DIR / "ETTh1.csv"
if not ETT_CSV.exists():
    urllib.request.urlretrieve(
        "https://raw.githubusercontent.com/zhouhaoyi/ETDataset/main/ETT-small/ETTh1.csv",
        ETT_CSV,
    )

memory_report("baseline")
```

    NVIDIA GeForce RTX 3060
    device: cuda:0
    RAM  baseline           15.31 / 20.97 GB
    VRAM baseline            0.00 GB allocated /  0.00 GB reserved

``` python
CHANNELS = ["HUFL", "HULL", "MUFL", "MULL", "LUFL", "LULL", "OT"]
TARGET = "OT"                      # oil temperature, the channel the benchmarks forecast

df = pd.read_csv(ETT_CSV, parse_dates=["date"])
series = df[CHANNELS].to_numpy(dtype="float32")
y_all = df[TARGET].to_numpy(dtype="float32")
n = len(df)

# The canonical ETT split is chronological 60/20/20 - never random. Everything after
# the boundary is the future and must stay untouched until evaluation.
TRAIN_END, VAL_END = int(n * 0.6), int(n * 0.8)
CTX, HORIZON, SEASON = 512, 96, 24     # 512 h of context, 96 h ahead, daily seasonality

print(f"{n:,} hourly rows, {df.date.min().date()} to {df.date.max().date()}")
print(f"train [0:{TRAIN_END}]  val [{TRAIN_END}:{VAL_END}]  test [{VAL_END}:{n}]")
print(f"context {CTX} h ({CTX / 24:.0f} days) -> horizon {HORIZON} h ({HORIZON / 24:.0f} days)")
print(f"\n{TARGET}: mean {y_all.mean():.2f}  sd {y_all.std():.2f}  "
      f"min {y_all.min():.2f}  max {y_all.max():.2f}")
df.head()
```

    17,420 hourly rows, 2016-07-01 to 2018-06-26
    train [0:10452]  val [10452:13936]  test [13936:17420]
    context 512 h (21 days) -> horizon 96 h (4 days)

    OT: mean 13.32  sd 8.57  min -4.08  max 46.01

|     | date                | HUFL  | HULL  | MUFL  | MULL  | LUFL  | LULL  | OT        |
|-----|---------------------|-------|-------|-------|-------|-------|-------|-----------|
| 0   | 2016-07-01 00:00:00 | 5.827 | 2.009 | 1.599 | 0.462 | 4.203 | 1.340 | 30.531000 |
| 1   | 2016-07-01 01:00:00 | 5.693 | 2.076 | 1.492 | 0.426 | 4.142 | 1.371 | 27.787001 |
| 2   | 2016-07-01 02:00:00 | 5.157 | 1.741 | 1.279 | 0.355 | 3.777 | 1.218 | 27.787001 |
| 3   | 2016-07-01 03:00:00 | 5.090 | 1.942 | 1.279 | 0.391 | 3.807 | 1.279 | 25.044001 |
| 4   | 2016-07-01 04:00:00 | 5.358 | 1.942 | 1.492 | 0.462 | 3.868 | 1.279 | 21.948000 |

``` python
from pyecharts.globals import CurrentConfig

# assets.pyecharts.org serves an expired TLS certificate from most nodes (2026-11-11),
# so charts failed to load. jsDelivr mirrors the same assets, pinned to one commit.
CurrentConfig.ONLINE_HOST = "https://cdn.jsdelivr.net/gh/pyecharts/pyecharts-assets@f2328d558550046fc5449672a6850e848e141590/assets/v6/"

from pyecharts import options as opts
from pyecharts.charts import Line

# The whole OT series, thinned to ~1,500 points. Look for what makes this hard:
# a level shift around the middle, seasonality that changes amplitude, and a test
# period (the last 20%) that does not look much like the training period.
step = max(1, n // 1500)
idx = np.arange(0, n, step)
dates = df["date"].dt.strftime("%Y-%m-%d").to_numpy()[idx]

overview = (
    Line()
    .add_xaxis(list(dates))
    .add_yaxis("OT (oil temperature)", [round(float(v), 2) for v in y_all[idx]],
               is_smooth=True, symbol="none", label_opts=opts.LabelOpts(is_show=False))
    .set_global_opts(
        title_opts=opts.TitleOpts(title="ETTh1: oil temperature, 2016-07 to 2018-06",
                                  subtitle="the last 20% (right) is the test period - it does not resemble the training period"),
        xaxis_opts=opts.AxisOpts(axislabel_opts=opts.LabelOpts(rotate=45, font_size=9)),
        yaxis_opts=opts.AxisOpts(name="temperature"),
        tooltip_opts=opts.TooltipOpts(trigger="axis"),
        datazoom_opts=[opts.DataZoomOpts(range_start=0, range_end=100)],
    )
)
overview.render_notebook()
```

``` python
# The evaluation harness every model below is scored with: fixed test windows,
# rolling origin, one week apart so they do not overlap much.
STRIDE = 168                                    # one week between window starts
STARTS = list(range(VAL_END, n - CTX - HORIZON, STRIDE))
print(f"{len(STARTS)} test windows, {CTX} h context -> {HORIZON} h horizon, {STRIDE} h apart")

Y_TRUE = np.stack([y_all[s + CTX: s + CTX + HORIZON] for s in STARTS])       # (W, H)
CONTEXTS = np.stack([y_all[s: s + CTX] for s in STARTS])                     # (W, L)
TRAIN_HIST = y_all[:TRAIN_END]

# One scale factor for MASE, computed on the TRAINING history only - computing it on
# the test period would be leakage of exactly the kind section 2 warns about.
NAIVE_MAE = float(np.mean(np.abs(TRAIN_HIST[SEASON:] - TRAIN_HIST[:-SEASON])))
print(f"seasonal-naive MAE on the training history = {NAIVE_MAE:.3f}  (the MASE denominator)")


def score(name, preds, seconds=None):
    "Score a (W, H) forecast array against Y_TRUE with the shared metrics."
    err = preds - Y_TRUE
    out = dict(model=name,
               mae=float(np.abs(err).mean()),
               rmse=float(np.sqrt((err ** 2).mean())),
               mase=float(np.abs(err).mean() / NAIVE_MAE),
               smape=float(np.mean(np.abs(err) / np.maximum(
                   (np.abs(Y_TRUE) + np.abs(preds)) / 2, 1e-9)) * 100),
               seconds=seconds)
    print(f"{name:26s} MAE {out['mae']:6.3f}  RMSE {out['rmse']:6.3f}  "
          f"MASE {out['mase']:5.3f}  sMAPE {out['smape']:6.2f}%"
          + (f"  {seconds:6.2f}s" if seconds is not None else ""))
    return out
```

    18 test windows, 512 h context -> 96 h horizon, 168 h apart
    seasonal-naive MAE on the training history = 2.436  (the MASE denominator)

## 8. The Baselines You Have To Beat

Three models with no parameters, and on this dataset they are not a formality.

- **Last value (random walk).** Repeat the final observation for all 96 hours. For a series with a strong trend and weak seasonality this is genuinely hard to beat at short horizons.
- **Seasonal naive.** Repeat the last 24 hours, four times. This encodes the daily cycle for free.
- **Seasonal mean.** Average the last 7 days hour-by-hour. Same seasonal structure, but the noise is averaged down.

Whichever of these wins is the number every trained model below is judged against, and it is what MASE is normalised by. If a 500M-parameter foundation model cannot beat “repeat last week”, the honest conclusion is to ship “repeat last week”.

Watch which one wins here. Oil temperature drifts more than it cycles over four days, so the *non*-seasonal baseline comes out ahead of both seasonal ones - a reminder that “seasonal naive” is not automatically the right naive, and that picking the baseline is itself a modelling decision about the series.

------------------------------------------------------------------------

``` python
results = []

t0 = time.perf_counter()
pred_last = np.repeat(CONTEXTS[:, -1:], HORIZON, axis=1)
results.append(score("naive (last value)", pred_last, time.perf_counter() - t0))

t0 = time.perf_counter()
pred_snaive = np.tile(CONTEXTS[:, -SEASON:], (1, HORIZON // SEASON + 1))[:, :HORIZON]
results.append(score("seasonal naive (24 h)", pred_snaive, time.perf_counter() - t0))

t0 = time.perf_counter()
last_week = CONTEXTS[:, -7 * SEASON:].reshape(len(STARTS), 7, SEASON).mean(axis=1)
pred_smean = np.tile(last_week, (1, HORIZON // SEASON + 1))[:, :HORIZON]
results.append(score("seasonal mean (7 days)", pred_smean, time.perf_counter() - t0))

print(f"\nMASE is 1.0 by construction for a seasonal-naive forecast measured on the "
      f"training history;\nthe values above differ because the test period is harder "
      f"than the training period.")
```

    naive (last value)         MAE  2.306  RMSE  3.025  MASE 0.947  sMAPE  35.47%    0.00s
    seasonal naive (24 h)      MAE  2.545  RMSE  3.292  MASE 1.045  sMAPE  36.72%    0.00s
    seasonal mean (7 days)     MAE  2.524  RMSE  3.187  MASE 1.036  sMAPE  34.56%    0.00s

    MASE is 1.0 by construction for a seasonal-naive forecast measured on the training history;
    the values above differ because the test period is harder than the training period.

## 9. TimesFM 2.0: forecasting with no training at all

TimesFM is a decoder-only transformer pretrained on over 100 billion real-world time points. It takes a raw context window and emits the horizon **zero-shot** - no fitting, no validation split, no hyperparameters. The 500M checkpoint is transformers-native as `TimesFmModelForPrediction`, so it drops in with `from_pretrained`.

Three mechanics worth knowing:

- **It patches the input** (32 steps per patch) and predicts 128 steps per output patch, so the 96-hour horizon comes out of a single forward pass rather than 96 autoregressive steps.
- **`past_values` takes a list of 1-D tensors**, not a padded batch, so windows of different lengths batch together naturally. All 18 test windows go through in one call below.
- **`freq` is a coarse frequency hint**: 0 for high frequency (hourly and finer), 1 for medium (weekly, monthly), 2 for low (quarterly, yearly). It is not the sampling rate.

It also returns `full_predictions`, which carries the quantile heads - the model is probabilistic, not just a point forecaster.

------------------------------------------------------------------------

``` python
from transformers import TimesFmModelForPrediction

TIMESFM_ID = "google/timesfm-2.0-500m-pytorch"   # ~2 GB of safetensors fetched
timesfm = TimesFmModelForPrediction.from_pretrained(
    TIMESFM_ID, cache_dir=HF_CACHE, dtype=torch.float32).to(device).eval()
print(f"{TIMESFM_ID}: {sum(p.numel() for p in timesfm.parameters()) / 1e6:.0f}M params, "
      f"context up to {timesfm.config.context_length}, "
      f"output patch {timesfm.config.horizon_length}")
vram("timesfm loaded")

contexts = [torch.tensor(c, device=device) for c in CONTEXTS]
t0 = time.perf_counter()
with torch.inference_mode():
    out = timesfm(past_values=contexts, freq=[0] * len(contexts), return_dict=True)
elapsed = time.perf_counter() - t0

pred_tfm = out.mean_predictions[:, :HORIZON].float().cpu().numpy()
results.append(score("TimesFM 2.0 (zero-shot)", pred_tfm, elapsed))
print(f"\nall {len(contexts)} windows in one forward pass, {elapsed:.2f}s total "
      f"({elapsed / len(contexts) * 1000:.0f} ms per window)")
print(f"full_predictions carries the quantile heads: shape {tuple(out.full_predictions.shape)}")
```

    google/timesfm-2.0-500m-pytorch: 499M params, context up to 2048, output patch 128
    VRAM timesfm loaded      2.00 GB allocated /  2.14 GB reserved
    TimesFM 2.0 (zero-shot)    MAE  1.999  RMSE  2.665  MASE 0.821  sMAPE  26.42%    0.35s

    all 18 windows in one forward pass, 0.35s total (20 ms per window)
    full_predictions carries the quantile heads: shape (18, 128, 10)

``` python
# The quantile heads, on one window. full_predictions is (batch, horizon, 1 + n_quantiles):
# channel 0 is the mean, channels 1..9 are deciles q10..q90.
w = 0
fp = out.full_predictions[w, :HORIZON].float().cpu().numpy()
q10, q50, q90 = fp[:, 1], fp[:, 5], fp[:, 9]

hours = list(range(HORIZON))
band = (
    Line()
    .add_xaxis(hours)
    .add_yaxis("actual", [round(float(v), 3) for v in Y_TRUE[w]],
               is_smooth=True, symbol="none", label_opts=opts.LabelOpts(is_show=False))
    .add_yaxis("TimesFM q50", [round(float(v), 3) for v in q50],
               is_smooth=True, symbol="none", label_opts=opts.LabelOpts(is_show=False))
    .add_yaxis("q10", [round(float(v), 3) for v in q10], is_smooth=True, symbol="none",
               linestyle_opts=opts.LineStyleOpts(type_="dashed"),
               label_opts=opts.LabelOpts(is_show=False))
    .add_yaxis("q90", [round(float(v), 3) for v in q90], is_smooth=True, symbol="none",
               linestyle_opts=opts.LineStyleOpts(type_="dashed"),
               label_opts=opts.LabelOpts(is_show=False))
    .add_yaxis("seasonal naive", [round(float(v), 3) for v in pred_snaive[w]],
               is_smooth=True, symbol="none",
               linestyle_opts=opts.LineStyleOpts(type_="dotted"),
               label_opts=opts.LabelOpts(is_show=False))
    .set_global_opts(
        title_opts=opts.TitleOpts(title="TimesFM zero-shot: 96 h ahead on one test window",
                                  subtitle="the q10-q90 band widens with horizon, which is the honest behaviour"),
        xaxis_opts=opts.AxisOpts(type_="value", name="hours ahead"),
        yaxis_opts=opts.AxisOpts(name="oil temperature", min_="dataMin"),
        tooltip_opts=opts.TooltipOpts(trigger="axis"),
        legend_opts=opts.LegendOpts(pos_top="8%"),
    )
)

cover = float(np.mean((Y_TRUE[w] >= q10) & (Y_TRUE[w] <= q90)))
print(f"window 0: nominal 80% interval covers {cover:.0%} of the 96 actual hours, "
      f"mean width {np.mean(q90 - q10):.2f}")
del timesfm, out, contexts
free_memory()
vram("after timesfm")
band.render_notebook()
```

    window 0: nominal 80% interval covers 81% of the 96 actual hours, mean width 4.10
    VRAM after timesfm       0.01 GB allocated /  0.02 GB reserved

## 10. PatchTST: training a transformer from scratch

PatchTST (Nie et al., ICLR 2023) is the design that made transformers work on long-horizon forecasting, and it did it by taking two things away:

- **Patching.** Instead of one token per timestep, split the series into subsequence patches (16 steps here). A 336-step context becomes 21 tokens instead of 336, so attention cost drops by ~250x and each token carries local shape rather than a single scalar.
- **Channel independence.** Every channel goes through the same backbone separately, with no cross-channel attention. Counter-intuitively this **improves** accuracy on these benchmarks - cross-channel attention mostly overfits with 7 channels and 10k rows.

It also uses **RevIN**-style instance normalisation (`scaling="std"`): each window is normalised by its own mean and standard deviation and denormalised on the way out, which is what lets one global model handle a non-stationary series whose level shifts.

The config below is deliberately small (0.11M parameters, `d_model=64`, 3 layers). It trains in under a minute on the RTX 3060 and gets most of the way to the published numbers; the paper’s configuration is roughly 10x larger and trains for 100 epochs.

------------------------------------------------------------------------

``` python
from torch.utils.data import DataLoader, TensorDataset
from transformers import PatchTSTConfig, PatchTSTForPrediction

PT_CTX = 336                    # the context length the PatchTST paper uses on ETTh1

# Normalise with TRAINING statistics only. Using the full-series mean here would leak
# the test period into every window - the classic invisible forecasting bug.
mu, sd = series[:TRAIN_END].mean(0), series[:TRAIN_END].std(0)
Z = (series - mu) / sd


def make_windows(lo, hi, stride=1):
    "Sliding (past, future) windows from Z[lo:hi]."
    starts = range(lo, hi - PT_CTX - HORIZON, stride)
    past = np.stack([Z[i: i + PT_CTX] for i in starts])
    future = np.stack([Z[i + PT_CTX: i + PT_CTX + HORIZON] for i in starts])
    return torch.tensor(past), torch.tensor(future)


past_tr, fut_tr = make_windows(0, TRAIN_END)
past_va, fut_va = make_windows(TRAIN_END, VAL_END, stride=4)
print(f"train windows {tuple(past_tr.shape)}   val windows {tuple(past_va.shape)}")

patchtst = PatchTSTForPrediction(PatchTSTConfig(
    num_input_channels=len(CHANNELS),
    context_length=PT_CTX, prediction_length=HORIZON,
    patch_length=16, patch_stride=16,          # 336 / 16 = 21 tokens per channel
    d_model=64, num_hidden_layers=3, num_attention_heads=4, ffn_dim=128,
    dropout=0.2, head_dropout=0.2,
    scaling="std",                             # RevIN-style per-window normalisation
    loss="mse",
)).to(device)
print(f"PatchTST: {sum(p.numel() for p in patchtst.parameters()) / 1e6:.3f}M params, "
      f"{PT_CTX // 16} patches per channel")
vram("patchtst init")
```

    train windows (10020, 336, 7)   val windows (763, 336, 7)
    PatchTST: 0.109M params, 21 patches per channel
    VRAM patchtst init       0.01 GB allocated /  0.02 GB reserved

``` python
EPOCHS = 12
opt = torch.optim.AdamW(patchtst.parameters(), lr=1e-3, weight_decay=1e-4)
sched = torch.optim.lr_scheduler.OneCycleLR(
    opt, max_lr=1e-3, total_steps=EPOCHS * (len(past_tr) // 64 + 1))

# num_workers=0: only 4 vCPU, and these tensors are already in RAM.
loader = DataLoader(TensorDataset(past_tr, fut_tr), batch_size=64, shuffle=True, num_workers=0)

train_curve, val_curve = [], []
best_val, best_state = float("inf"), None
t0 = time.perf_counter()
for epoch in range(EPOCHS):
    patchtst.train()
    total = 0.0
    for past, future in loader:
        loss = patchtst(past_values=past.to(device), future_values=future.to(device)).loss
        opt.zero_grad()
        loss.backward()
        opt.step()
        sched.step()
        total += loss.item()
    train_curve.append(total / len(loader))

    patchtst.eval()
    with torch.inference_mode():
        val_loss = patchtst(past_values=past_va.to(device),
                            future_values=fut_va.to(device)).loss.item()
    val_curve.append(val_loss)

    # Validation bottoms out before training does, so keep the best state rather than
    # whatever the last epoch happens to leave behind - the standard early-stopping
    # discipline, done by hand because this loop is deliberately explicit.
    if val_loss < best_val:
        best_val, best_state = val_loss, {k: v.detach().clone()
                                          for k, v in patchtst.state_dict().items()}
    print(f"epoch {epoch + 1:2d}  train {train_curve[-1]:.4f}  val {val_loss:.4f}"
          + ("  <- best" if val_loss == best_val else ""))

train_seconds = time.perf_counter() - t0
patchtst.load_state_dict(best_state)
print(f"\n{train_seconds:.0f}s total on {device}; restored the epoch with val {best_val:.4f}")
del loader, past_tr, fut_tr, past_va, fut_va, best_state   # keep the model, drop the rest
free_memory()
memory_report("after training")
```

    epoch  1  train 0.6259  val 0.6265  <- best
    epoch  2  train 0.5462  val 0.6042  <- best
    epoch  3  train 0.5132  val 0.5344  <- best
    epoch  4  train 0.4069  val 0.4069  <- best
    epoch  5  train 0.3733  val 0.3973  <- best
    epoch  6  train 0.3568  val 0.4017
    epoch  7  train 0.3463  val 0.3915  <- best
    epoch  8  train 0.3335  val 0.4056
    epoch  9  train 0.3255  val 0.3967
    epoch 10  train 0.3185  val 0.3991
    epoch 11  train 0.3140  val 0.4021
    epoch 12  train 0.3113  val 0.4027

    18s total on cuda:0; restored the epoch with val 0.3915
    RAM  after training     16.08 / 20.97 GB
    VRAM after training      0.02 GB allocated /  0.05 GB reserved

``` python
curve = (
    Line()
    .add_xaxis([str(e) for e in range(1, EPOCHS + 1)])
    .add_yaxis("train MSE", [round(v, 4) for v in train_curve],
               is_smooth=True, label_opts=opts.LabelOpts(is_show=False))
    .add_yaxis("val MSE", [round(v, 4) for v in val_curve],
               is_smooth=True, label_opts=opts.LabelOpts(is_show=False))
    .set_global_opts(
        title_opts=opts.TitleOpts(title="PatchTST training on ETTh1",
                                  subtitle="loss is on standardised units, so it is not comparable to the MAE table"),
        xaxis_opts=opts.AxisOpts(name="epoch"),
        yaxis_opts=opts.AxisOpts(name="MSE (standardised)"),
        tooltip_opts=opts.TooltipOpts(trigger="axis"),
        legend_opts=opts.LegendOpts(pos_top="8%"),
    )
)
curve.render_notebook()
```

``` python
# Score PatchTST on the SAME test windows as everything else. It needs PT_CTX=336
# hours of context ending at the same point, so the windows line up exactly.
past_test = torch.tensor(np.stack([Z[s + CTX - PT_CTX: s + CTX] for s in STARTS]))
target_ch = CHANNELS.index(TARGET)

patchtst.eval()
t0 = time.perf_counter()
with torch.inference_mode():
    z_pred = patchtst(past_values=past_test.to(device)).prediction_outputs.float().cpu().numpy()
elapsed = time.perf_counter() - t0

# Undo the training-set standardisation to get back to degrees.
pred_patchtst = z_pred[:, :, target_ch] * sd[target_ch] + mu[target_ch]
results.append(score("PatchTST (trained here)", pred_patchtst, elapsed))
print(f"\ntraining cost {train_seconds:.0f}s; inference {elapsed * 1000:.0f}ms for "
      f"{len(STARTS)} windows")

del patchtst, past_test
free_memory()
vram("after patchtst")
```

    PatchTST (trained here)    MAE  2.274  RMSE  2.914  MASE 0.933  sMAPE  31.08%    0.00s

    training cost 18s; inference 2ms for 18 windows
    VRAM after patchtst      0.02 GB allocated /  0.05 GB reserved

## 11. Head-to-head Benchmark

Same 18 test windows, same 512-hour context ending at the same timestamp, same horizon, same metrics. Two caveats that matter for reading the table honestly:

- **The comparison is not apples to apples on training data.** TimesFM never saw ETTh1 in this notebook (though ETT-like data may well be in its pretraining corpus - that is the standing caveat with every foundation model benchmark). PatchTST was trained on the first 60% of this exact series. That is the realistic choice you face, not a flaw in the experiment.
- **18 windows on one channel of one dataset is a smoke test.** GIFT-Eval runs 24 datasets and 144k series for a reason. Treat the ordering here as indicative, and read MASE rather than MAE if you want to compare against anything outside this notebook.

**And the result is exactly the warning from section 2.** Measured on this box: TimesFM zero-shot lands at MASE 0.82, and the strongest naive baseline - **repeat the last value** - lands at 0.95. The small PatchTST trained here comes in at 0.94: it clears the baseline, by about one percent. Sixteen seconds of training, 10,020 windows, and a purpose-built long-horizon architecture, to beat “assume nothing changes” by one percent.

That is not an argument against PatchTST; it is an argument for always printing the baseline. The paper’s configuration is roughly 10x larger and trains for 100 epochs, and 12 epochs of a 0.11M-parameter model is not that. What the number does show is how easy it would be to report “we trained a transformer forecaster, MAE 2.28” as a success and never discover that repeating the last observation scored 2.31 - and that the 500M model nobody trained scored 2.00.

Hardware: knowledge-lab, RTX 3060 (12 GB) and 4 vCPU.

------------------------------------------------------------------------

``` python
bench = pd.DataFrame(results).sort_values("mase").reset_index(drop=True)

BASELINES = ["naive (last value)", "seasonal naive (24 h)", "seasonal mean (7 days)"]
best_baseline = bench[bench["model"].isin(BASELINES)].iloc[0]
bench["beats_baselines"] = bench["mase"] < best_baseline["mase"]

print(f"best baseline: {best_baseline['model']} at MASE {best_baseline['mase']:.3f} - "
      "any model that does not clear this line has not earned its complexity")
memory_report("after benchmark")
bench.round(4)
```

    best baseline: naive (last value) at MASE 0.947 - any model that does not clear this line has not earned its complexity
    RAM  after benchmark    16.08 / 20.97 GB
    VRAM after benchmark     0.02 GB allocated /  0.05 GB reserved

|  | model | mae | rmse | mase | smape | seconds | beats_baselines |
|----|----|----|----|----|----|----|----|
| 0 | TimesFM 2.0 (zero-shot) | 1.9986 | 2.6654 | 0.8205 | 26.4215 | 0.3518 | True |
| 1 | PatchTST (trained here) | 2.2736 | 2.9138 | 0.9335 | 31.0761 | 0.0023 | True |
| 2 | naive (last value) | 2.3065 | 3.0250 | 0.9470 | 35.4684 | 0.0001 | False |
| 3 | seasonal mean (7 days) | 2.5243 | 3.1866 | 1.0364 | 34.5552 | 0.0001 | False |
| 4 | seasonal naive (24 h) | 2.5452 | 3.2917 | 1.0449 | 36.7166 | 0.0001 | False |

``` python
from pyecharts.charts import Bar

names = bench["model"].tolist()
metric_bar = (
    Bar()
    .add_xaxis(names)
    .add_yaxis("MASE", [round(float(v), 3) for v in bench["mase"]])
    .add_yaxis("MAE", [round(float(v), 3) for v in bench["mae"]])
    .set_series_opts(label_opts=opts.LabelOpts(is_show=False))
    .set_global_opts(
        title_opts=opts.TitleOpts(
            title="ETTh1 OT, 96 h horizon: error by model",
            subtitle=f"{len(STARTS)} rolling test windows - lower is better"),
        xaxis_opts=opts.AxisOpts(axislabel_opts=opts.LabelOpts(rotate=20, font_size=10)),
        yaxis_opts=opts.AxisOpts(name="error"),
        tooltip_opts=opts.TooltipOpts(trigger="axis"),
        legend_opts=opts.LegendOpts(pos_top="8%"),
    )
)
metric_bar.render_notebook()
```

``` python
# Error against horizon: every model is good at hour 1 and the question is how fast
# it decays. A flat curve means the model found real structure; a steep one means it
# is mostly extrapolating the last observation.
by_h = Line().add_xaxis([str(h + 1) for h in range(HORIZON)])
for name, preds in [("naive (last value)", pred_last),
                    ("seasonal naive (24 h)", pred_snaive),
                    ("seasonal mean (7 days)", pred_smean),
                    ("TimesFM 2.0 (zero-shot)", pred_tfm),
                    ("PatchTST (trained here)", pred_patchtst)]:
    mae_h = np.abs(preds - Y_TRUE).mean(axis=0)
    by_h.add_yaxis(name, [round(float(v), 3) for v in mae_h], is_smooth=True,
                   symbol="none", label_opts=opts.LabelOpts(is_show=False))

by_h.set_global_opts(
    title_opts=opts.TitleOpts(title="MAE by forecast horizon",
                              subtitle="how fast each model decays as it looks further ahead"),
    xaxis_opts=opts.AxisOpts(name="hours ahead",
                             axislabel_opts=opts.LabelOpts(interval=11)),
    yaxis_opts=opts.AxisOpts(name="MAE"),
    tooltip_opts=opts.TooltipOpts(trigger="axis"),
    legend_opts=opts.LegendOpts(pos_top="8%"),
)
by_h.render_notebook()
```

``` python
# The forecasts themselves, on the window where the models disagree most.
spread = np.array([np.abs(pred_tfm[i] - pred_patchtst[i]).mean() for i in range(len(STARTS))])
w = int(np.argmax(spread))

compare = (
    Line()
    .add_xaxis(list(range(-72, HORIZON)))
    .add_yaxis("context + actual",
               [round(float(v), 3) for v in np.concatenate([CONTEXTS[w, -72:], Y_TRUE[w]])],
               is_smooth=True, symbol="none",
               linestyle_opts=opts.LineStyleOpts(width=2),
               label_opts=opts.LabelOpts(is_show=False))
)
for name, preds in [("seasonal naive", pred_snaive), ("TimesFM 2.0", pred_tfm),
                    ("PatchTST", pred_patchtst)]:
    compare.add_yaxis(name, [None] * 72 + [round(float(v), 3) for v in preds[w]],
                      is_smooth=True, symbol="none", label_opts=opts.LabelOpts(is_show=False))

compare.set_global_opts(
    title_opts=opts.TitleOpts(
        title=f"Test window {w}: the 72 h before, then 96 h of forecast",
        subtitle="chosen as the window where TimesFM and PatchTST disagree most"),
    xaxis_opts=opts.AxisOpts(type_="value", name="hours (0 = forecast origin)"),
    yaxis_opts=opts.AxisOpts(name="oil temperature", min_="dataMin"),
    tooltip_opts=opts.TooltipOpts(trigger="axis"),
    legend_opts=opts.LegendOpts(pos_top="8%"),
)
compare.render_notebook()
```

## 12. Backtesting: one split is not an evaluation

Everything above used a fixed set of windows in the last 20% of the series. That is already better than a single train/test cut, but a real evaluation uses **rolling-origin backtesting**: advance the forecast origin through time, refit (or at least re-context) at each origin, and average the metric over origins. It is the only way to see that a model was excellent in spring and useless in autumn.

The cell measures the seasonal-naive and TimesFM error **per month of test period**, which is the cheapest version of the same idea. Watch how much the error moves between months - that spread is the honest uncertainty on any single number in the benchmark table above, and it is usually larger than the gap between the models.

------------------------------------------------------------------------

``` python
months = pd.to_datetime(df["date"]).dt.to_period("M").to_numpy()
window_month = np.array([str(months[s + CTX]) for s in STARTS])

rows = []
for name, preds in [("seasonal naive (24 h)", pred_snaive),
                    ("TimesFM 2.0 (zero-shot)", pred_tfm),
                    ("PatchTST (trained here)", pred_patchtst)]:
    per_window = np.abs(preds - Y_TRUE).mean(axis=1)
    for m in sorted(set(window_month)):
        sel = window_month == m
        rows.append(dict(model=name, month=m, mae=float(per_window[sel].mean()),
                         windows=int(sel.sum())))

per_month = pd.DataFrame(rows)
pivot = per_month.pivot(index="month", columns="model", values="mae")

drift = Line().add_xaxis(list(pivot.index))
for col in pivot.columns:
    drift.add_yaxis(col, [round(float(v), 3) for v in pivot[col]], is_smooth=True,
                    symbol="circle", symbol_size=7, label_opts=opts.LabelOpts(is_show=False))
drift.set_global_opts(
    title_opts=opts.TitleOpts(title="MAE by month of the test period",
                              subtitle="the month-to-month spread is bigger than the model-to-model gap"),
    xaxis_opts=opts.AxisOpts(name="month", axislabel_opts=opts.LabelOpts(rotate=30)),
    yaxis_opts=opts.AxisOpts(name="MAE"),
    tooltip_opts=opts.TooltipOpts(trigger="axis"),
    legend_opts=opts.LegendOpts(pos_top="8%"),
)
print(pivot.round(3).to_string())
print("\nspread within a model across months:")
print((pivot.max() - pivot.min()).round(3).to_string())
drift.render_notebook()
```

    model    PatchTST (trained here)  TimesFM 2.0 (zero-shot)  seasonal naive (24 h)
    month                                                                           
    2018-02                    1.915                    1.412                  0.947
    2018-03                    2.691                    2.288                  3.114
    2018-04                    2.327                    1.756                  2.565
    2018-05                    2.733                    2.658                  3.072
    2018-06                    1.328                    1.366                  1.687

    spread within a model across months:
    model
    PatchTST (trained here)    1.405
    TimesFM 2.0 (zero-shot)    1.292
    seasonal naive (24 h)      2.167

## 13. Common Frameworks

Forecasting has two ecosystems that only recently started talking to each other. The established one - statsmodels, Nixtla, sktime, Darts - is built around fitting many series in parallel, covariates, hierarchies and rolling backtests, and it is what production forecasting actually runs on. The new one is foundation models in `transformers`. The second is more exciting; the first is where the accuracy on business data still comes from, mostly because it handles covariates and the other does not.

| Framework | Layer | What it gives you | License | Reach for it when |
|----|----|----|----|----|
| [statsforecast](https://github.com/Nixtla/statsforecast) + [mlforecast](https://github.com/Nixtla/mlforecast) | modelling | AutoARIMA, AutoETS and Theta fitted over thousands of series in parallel, and LightGBM-on-lags - which won M5 and runs most retail forecasting | Apache 2.0 | Start here. The classical baselines done properly are what section 8 says you have to beat |
| [transformers](https://github.com/huggingface/transformers) | modelling | PatchTST, TimesFM and the Granite time-series checkpoints, with `distribution_output` for probabilistic heads | Apache 2.0 | Zero-shot forecasting, or fine-tuning a foundation model - which beats both zero-shot and from-scratch for far less compute |
| [sktime](https://github.com/sktime/sktime) / [Darts](https://github.com/unit8co/darts) | modelling | A scikit-learn-shaped API over classical, ML and deep forecasters, with pipelines and cross-validation built in | BSD-3 / Apache 2.0 | You want to compare across model families without rewriting the harness three times |
| [statsmodels](https://github.com/statsmodels/statsmodels) | modelling | SARIMAX, state-space models, seasonal decomposition, and the diagnostics that explain *why* a series behaves as it does | BSD-3 | Interpretation, or a series where the structure matters more than the point forecast |
| [pandas](https://github.com/pandas-dev/pandas) / [polars](https://github.com/pola-rs/polars) | data | Resampling, gap filling, lag and rolling features, and calendar joins | BSD-3 / MIT | Always. Most forecasting bugs are index bugs - missing timestamps, duplicated origins, silent timezone shifts |
| [hierarchicalforecast](https://github.com/Nixtla/hierarchicalforecast) | data | MinT and other reconciliation methods, so store forecasts sum to region and national totals | Apache 2.0 | Any hierarchy. Forecasting the bottom level and adding up is measurably worse than reconciling |
| [ONNX Runtime](https://onnxruntime.ai/) / [Treelite](https://github.com/dmlc/treelite) | inference runtime | The LightGBM path compiled for fast batch scoring of many series | MIT / Apache 2.0 | Deployment. Forecasting is usually a scheduled batch job, so throughput over many series is the metric |
| [MLflow](https://github.com/mlflow/mlflow) / [Feast](https://github.com/feast-dev/feast) | serving | Versioned models and a feature store that guarantees a covariate is computed the same way at train and predict time | Apache 2.0 | Production. Future covariates are the easiest place to leak information and the hardest to notice |
| [utilsforecast](https://github.com/Nixtla/utilsforecast) / [Darts](https://github.com/unit8co/darts) backtesting | evaluation | Rolling-origin evaluation, MASE, pinball loss and CRPS, averaged over origins and series | Apache 2.0 | Always. Section 12 is the point: one split is not an evaluation |

**The 2026 default stack** is `statsforecast` for the baselines, `mlforecast` with LightGBM once you have covariates, a fine-tuned foundation model from `transformers` when you have many related series and few covariates, reconciliation if there is a hierarchy, and rolling-origin backtesting deciding between them.

The common wrong turn is reaching for a foundation model on a business series with strong known drivers. No model in section 6 takes future covariates as gracefully as LightGBM or TFT, and for promotions, holidays, prices and weather the covariate is worth more than the architecture. The second is forecasting the mean: inventory, capacity and bidding all consume a quantile, so predict the distribution and score it with pinball loss or CRPS. For mostly-zero demand, handle intermittency explicitly - and stop using MAPE, which is undefined there.

------------------------------------------------------------------------

## 14. Going Further

**Use your covariates.** No foundation model in section 6 takes future covariates as gracefully as LightGBM or TFT do, and for most business series (promotions, holidays, published prices, weather forecasts) the covariate is worth more than the architecture. If you have them, that is where the accuracy is.

**Fine-tune the foundation model instead of training from scratch.** TimesFM, Chronos and Moirai all fine-tune on a target dataset in far less compute than training a PatchTST, and usually beat both zero-shot and from-scratch. `PatchTSTForPrediction.from_pretrained("ibm-granite/granite-timeseries-patchtst")` is a transformers-native pretrained starting point if you want to stay inside `transformers`.

**Forecast the distribution, not the mean.** Set `loss="nll"` with a `distribution_output` on PatchTST, use the quantile heads TimesFM already returns, or fit LightGBM at several quantiles. Then score with pinball loss / CRPS and check coverage. Inventory, capacity and bidding all consume a quantile, never a mean.

**Reconcile hierarchies.** If store-level forecasts must sum to region and national totals, forecast every level and reconcile (MinT, `hierarchicalforecast`). Forecasting only the bottom level and adding up is measurably worse.

**Handle intermittency separately.** For mostly-zero demand, Croston’s method, ADIDA, or a Tweedie objective in LightGBM beat anything that assumes a continuous target - and MAPE/sMAPE stop being meaningful entirely.

**Backtest properly before believing any of this.** Rolling origin, multiple horizons, metrics averaged over origins and series, and the naive baseline printed next to every number.

**Related notebooks in this repo:** `01_Tabular_Regression` (the same target type when the rows are exchangeable), `00_Tabular_Classification` (turning a forecast into a decision), and `Audio/04_Audio_Classification` (the other place sliding windows over a 1-D signal show up).

------------------------------------------------------------------------

Back to top
