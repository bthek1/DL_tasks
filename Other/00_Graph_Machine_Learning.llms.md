# Graph Machine Learning

Learning on data whose structure is the relationships: what message passing actually computes, why homophily decides whether a GNN helps at all, the three task levels (node, edge, graph), and runnable code that builds GCN, GraphSAGE, GAT and GIN from scratch in ~20 lines of PyTorch each.

Author

Benedict Thekkel

## 1. What is Graph Machine Learning?

Graph machine learning is what you do when the **relationships between examples are part of the input**. A tabular model sees rows as independent; a graph model sees a paper and the papers it cites, a user and the users they follow, an atom and the atoms it is bonded to. The edges are not metadata - they carry most of the signal.

**Input.** A graph $G = (V, E)$ with:

- **Node features** $X \in \mathbb{R}^{N \times F}$ - what each node is on its own.
- **Edges**, as an index list, an adjacency matrix, or a sparse structure. Directed or not, weighted or not, typed or not.
- Optionally **edge features** (bond type, transaction amount, timestamp) and **graph-level features**.

The defining property, and the one every architecture is designed around, is **permutation invariance**: relabelling the nodes must not change the answer. That rules out feeding an adjacency matrix into an MLP (row order would matter) and is why message passing looks the way it does.

**Output.** Three task levels, and almost everything in the field is one of them:

| Level | Predicts | Examples |
|----|----|----|
| **Node** | A label or value per node | Fraud accounts, paper topics, protein function |
| **Edge / link** | Whether an edge exists, or its type | Recommendation, knowledge-graph completion, drug interaction |
| **Graph** | One label per whole graph | Molecule toxicity, program classification, mesh category |

There is a fourth setting that changes the engineering completely: **transductive** (one fixed graph, predict unlabelled nodes in it - the Cora setup below) versus **inductive** (generalise to nodes or graphs never seen at training time - what production systems need).

**Neighbouring tasks:**

| Task | What it does | Typical tool |
|----|----|----|
| Tabular classification | Same labels, but rows are independent | see `Tabular/00_Tabular_Classification` |
| Community detection | Unsupervised node grouping | Louvain, Leiden (`networkx`, `igraph`) |
| Node embedding (unsupervised) | Vector per node from structure alone | DeepWalk, node2vec |
| Knowledge-graph embedding | Score (head, relation, tail) triples | TransE, ComplEx, RotatE |
| Combinatorial optimisation | Routing, matching, scheduling on graphs | GNN + search, learned heuristics |
| Geometric deep learning | The umbrella: grids, groups, graphs, manifolds | Bronstein et al., 2021 |

------------------------------------------------------------------------

## 2. Real-World Use Cases

| Use case | Domain | Consumes / produces | Dominant constraint |
|----|----|----|----|
| Payment and account fraud rings | Fintech, payments (PayPal, Stripe) | Account/device/transaction graph -\> risk per node or subgraph | Adversarial and evolving; needs inference in milliseconds on a graph with billions of edges |
| Product and content recommendation | E-commerce, social (Pinterest PinSage, Alibaba, UberEats) | User-item interaction graph -\> ranked items | Inductive: new users and items arrive constantly; sampling is the whole engineering problem |
| Molecular property prediction | Drug discovery, materials (DeepMind GNoME, AlphaFold’s graph pieces) | Molecule as atoms + bonds -\> property, energy, toxicity | Data scarcity; 3D geometry and physical symmetries must be respected |
| Traffic and ETA prediction | Maps (Google Maps used a GNN for ETA from 2020) | Road network + live speeds -\> travel time per segment | Spatio-temporal: the graph is fixed, the signal is not; hard latency budget |
| Knowledge graph completion | Search, enterprise data | Entity-relation triples -\> missing facts | Very high cardinality; many relation types; incompleteness is the norm |
| Circuit and chip design | EDA (Google’s chip floorplanning) | Netlist as a graph -\> placement, congestion | Enormous graphs; the objective is a simulator, not a label |
| Social network abuse and spam | Platforms | Follow/interact graph -\> account classification | Label sparsity; adversaries deliberately break homophily |
| Supply-chain and financial risk | Banking, logistics | Counterparty/shipment graph -\> exposure, contagion | Explainability; small labelled sets; multi-hop effects matter |
| Code and program analysis | Developer tools | AST/dataflow graph -\> bug, type, completion | Heterogeneous node/edge types; graphs are large and deep |

**What the benchmark number hides.**

- **Homophily decides whether a GNN helps at all.** Message passing averages a node with its neighbours, which is only useful if neighbours tend to share the label. Cora is ~81% homophilous, so a GCN beats a feature-only MLP by 20 points (measured in section 9). On a **heterophilous** graph - fraudsters deliberately transact with legitimate accounts, a protein bonds to *different* residue types - the same averaging destroys signal, and a plain MLP can beat a GCN. Measure homophily before choosing an architecture.
- **The standard benchmarks are tiny and leaky.** Cora, CiteSeer and PubMed have a few thousand nodes, and a decade of papers tuned on the same 20-labels-per-class split. Shchur et al. (2018) showed the reported ranking largely dissolves under a fair evaluation with random splits. OGB exists to fix this.
- **Scale is the engineering problem, not accuracy.** A billion-edge graph does not fit on a GPU, so production systems are built out of **neighbour sampling** (GraphSAGE), cluster partitioning (Cluster-GCN), or historical embeddings. The paper’s model is usually not the bottleneck; the sampler is.
- **Transductive results do not transfer.** Almost every classic benchmark trains and tests on one fixed graph. Production is inductive - new users, new molecules, new accounts every hour - and a method that needs to see the whole graph at training time cannot be deployed that way.
- **Depth does not help.** Stack more than 2-4 message-passing layers and every node’s receptive field covers the graph, representations converge, and accuracy collapses. This is **oversmoothing**, and it is why almost every deployed GNN is shallow.
- **A good tabular baseline is often enough.** Aggregate neighbour features by hand (mean of neighbours, degree, triangle count, PageRank), put them in a table, fit LightGBM. This wins more often than the GNN literature suggests, trains in seconds, and is the baseline any GNN project should have to beat.

------------------------------------------------------------------------

## 3. How Modern Graph Learning Works

**1. Hand-built structural features (pre-2014).** Degree, PageRank, clustering coefficient, triangle counts, shortest-path features - computed with `networkx`, fed to a normal classifier. Still an excellent baseline, and still how a lot of production fraud scoring works.

**2. Shallow node embeddings (2014-2016).** DeepWalk and node2vec run random walks over the graph and apply word2vec to the sequences. Unsupervised, transductive, and blind to node features - a new node has no embedding until you retrain.

**3. Spectral graph convolution (2013-2016).** Define convolution through the graph Laplacian’s eigenbasis. Elegant, and expensive: eigendecomposition is $O(N^3)$ and the filters do not transfer between graphs. ChebNet (2016) approximated it with Chebyshev polynomials, which made it local and cheap.

**4. The GCN simplification (Kipf and Welling, 2017), where the field turned.** Truncate ChebNet to first order and you get one line:

$$H^{(l+1)} = \sigma\!\left(\hat{A} H^{(l)} W^{(l)}\right), \qquad
  \hat{A} = \tilde{D}^{-1/2} (A + I) \tilde{D}^{-1/2}$$

Add self-loops, symmetrically normalise, multiply, transform, activate. That is the whole model, and it is what section 9 implements in eight lines.

**5. Message Passing Neural Networks (Gilmer et al., 2017), the general frame.** Every GNN in common use is an instance of:

$$h_v^{(l+1)} = \text{UPDATE}\Big(h_v^{(l)},\ \ \text{AGGREGATE}\big(\{ \text{MESSAGE}(h_v^{(l)}, h_u^{(l)}, e_{uv}) : u \in \mathcal{N}(v) \}\big)\Big)$$

The architectures differ only in those three functions:

| Model | Year | AGGREGATE | The idea |
|----|----|----|----|
| **GCN** | 2017 | degree-normalised sum | Simplest possible spectral filter |
| **GraphSAGE** | 2017 | mean / max / LSTM, **plus a separate self term** | **Inductive** via neighbour sampling; scales |
| **GAT** | 2018 | attention-weighted sum | Learn which neighbours matter |
| **GIN** | 2019 | sum + MLP | Provably as expressive as the 1-WL test - the max for message passing |

**6. The expressiveness ceiling (Xu et al., 2019).** Message-passing GNNs cannot distinguish any pair of graphs that the **1-dimensional Weisfeiler-Lehman colour-refinement test** cannot - so they cannot count triangles or tell certain regular graphs apart. GIN hits that ceiling exactly. Getting past it needs something extra: positional or structural encodings, subgraph counts, or higher-order message passing.

**7. Graph transformers (2021 -\> now).** Attend over all node pairs and inject the structure as a bias instead of a constraint: **Graphormer** (2021, degree/spatial/edge encodings; won OGB-LSC), SAN, GraphGPS (2022, local message passing plus global attention). They dodge oversmoothing and the WL ceiling at quadratic cost, which is affordable for molecules (tens of atoms) and not for a social network.

**8. Where the field is in 2026.** Geometric and equivariant GNNs dominate the science applications - AlphaFold 3 (2024) and the diffusion-based structure predictors, GNoME’s 2.2 million new crystals (2023), and the machine-learned interatomic potentials (MACE, NequIP) that now do molecular dynamics at DFT accuracy. On the industrial side the story is scaling: sampling, partitioning and serving billion-edge graphs. And there is a persistent, healthy counter-current showing that on many public benchmarks a well-tuned MLP with good structural features gets uncomfortably close to the fancy model.

------------------------------------------------------------------------

## 4. Evaluation Metrics

The metric follows the task level, and the **split** is where graph evaluation goes wrong.

**Node classification.** Accuracy, or macro-F1 when classes are imbalanced. The subtlety is the split: in the transductive setting the whole graph (including test nodes and their edges) is visible during training, and only the *labels* are hidden. That is standard and legitimate, but it is not what production looks like.

**Link prediction.** ROC-AUC and average precision over a set of held-out positive edges and sampled negatives, or ranking metrics (Hits@K, MRR) when the task is “which of these candidates”. Two traps:

- **Negative sampling defines the difficulty.** Uniformly sampled non-edges are trivially separable (most random pairs are far apart); hard negatives from the 2-hop neighbourhood give a number that means something.
- **Held-out edges must be removed from the message-passing graph**, not just from the label set. Leaving them in lets the model read the answer off the adjacency it was given. This is the single most common link-prediction bug.

**Graph classification.** Accuracy or ROC-AUC over graphs, with **10-fold cross-validation** - the standard TU datasets have ~1,000 graphs, so a single split has an error bar of several points. Errico et al. (2020) showed a large fraction of the reported GNN improvements on these datasets vanish under a proper evaluation protocol.

**Scale-free sanity checks.** Always report:

- A **feature-only MLP** (ignores all edges). If the GNN does not beat it, the structure was not helping.
- A **structure-only** baseline (label propagation, or node2vec + logistic regression). If it matches the GNN, the features were not helping.
- **Edge homophily** $h = \frac{1}{|E|}\lvert \{(u,v) \in E : y_u = y_v\} \rvert$ - the single number that predicts whether message passing will work at all.

------------------------------------------------------------------------

``` python
import numpy as np

# Edge homophily on two toy graphs with the same shape and opposite structure.
rng = np.random.default_rng(0)


def edge_homophily(edges, labels):
    "Fraction of edges connecting two nodes with the same label."
    return float(np.mean(labels[edges[:, 0]] == labels[edges[:, 1]]))


N = 400
labels = rng.integers(0, 4, N)

# Homophilous: 90% of edges connect same-label nodes (a citation network).
same = [(i, rng.choice(np.where(labels == labels[i])[0])) for i in range(N) for _ in range(3)]
cross = [(i, rng.integers(0, N)) for i in range(N)]
homo = np.array(same + cross[: len(cross) // 8])

# Heterophilous: edges deliberately join different labels (a fraud ring hiding in
# legitimate traffic, or a bipartite-ish transaction graph).
diff = [(i, rng.choice(np.where(labels != labels[i])[0])) for i in range(N) for _ in range(3)]
hetero = np.array(diff + cross[: len(cross) // 8])

print(f"homophilous graph   h = {edge_homophily(homo, labels):.3f}  -> averaging neighbours helps")
print(f"heterophilous graph h = {edge_homophily(hetero, labels):.3f}  -> averaging destroys signal")
print(f"random baseline     h ~ {1 / 4:.3f}  (4 classes, edges independent of labels)")
print("\nA GCN on the second graph will lose to a plain MLP. Measure this before you model.")
```

    homophilous graph   h = 0.966  -> averaging neighbours helps
    heterophilous graph h = 0.006  -> averaging destroys signal
    random baseline     h ~ 0.250  (4 classes, edges independent of labels)

    A GCN on the second graph will lose to a plain MLP. Measure this before you model.

## 5. Datasets

The classic citation trio (Cora, CiteSeer, PubMed) is what almost every tutorial uses and is also the field’s known weak point: they are tiny, homophilous, and over-tuned. OGB (2020) exists because of that.

| Dataset | Contents | Size | Task | License | Typical use |
|----|----|----|----|----|----|
| [Cora](https://linqs.soe.ucsc.edu/data) | Citation network of ML papers; bag-of-words features, 7 topics | 2,708 nodes / 5,429 edges / 1,433 features | Node classification | Research use (LINQS) | The canonical demo - **used by this notebook** |
| [CiteSeer / PubMed](https://linqs.soe.ucsc.edu/data) | Same shape, different corpora | 3.3k / 19.7k nodes | Node classification | Research use | The rest of the classic trio |
| [PROTEINS](https://huggingface.co/datasets/graphs-datasets/PROTEINS) | Proteins as graphs of secondary-structure elements | 1,113 graphs, ~39 nodes each | Graph classification | Research use (TU Dortmund) | **Used by this notebook** (section 12) |
| [TUDataset collection](https://chrsmrrs.github.io/datasets/) | ~120 graph-classification datasets (MUTAG, ENZYMES, IMDB-B, …) | 200 to 200k graphs | Graph classification | Per-dataset | The standard small-graph suite |
| [OGB (`ogbn-arxiv`, `ogbn-products`, `ogbl-citation2`, `ogbg-molhiv`)](https://ogb.stanford.edu/) | Large, standardised, with **fixed realistic splits** | 169k to 2.4M nodes | All three levels | MIT | **The benchmark to use for anything serious** |
| [ZINC / QM9](https://www.kaggle.com/datasets/basu369victor/zinc-dataset) | Molecules with quantum/physicochemical targets | 250k / 134k molecules | Graph regression | Open | Molecular property prediction |
| [Heterophilous suite](https://github.com/yandex-research/heterophilous-graphs) | Roman-empire, amazon-ratings, minesweeper, … | 10k to 2.9M nodes | Node classification | MIT | Testing whether your model survives low homophily |
| [MalNet, ogbn-papers100M](https://ogb.stanford.edu/docs/lsc/) | Very large graphs | 100M+ nodes | Node classification | MIT | Sampling and systems research |

**This notebook uses Cora** (downloaded once as a 164 KB tarball into `DL_tasks/datasets/`, which is gitignored) for node classification and link prediction, and **PROTEINS** from the Hugging Face Hub for graph classification. Both are small enough that the whole notebook runs in about a minute on a 12 GB GPU - and small enough that you should not draw strong conclusions from the numbers. Nothing here is gated.

**On `torch_geometric` and DGL.** The two standard graph libraries are not repo dependencies, and this notebook does not need them: at 2,708 nodes a **dense** adjacency matrix is a 2708 x 2708 float32 array (29 MB), so every GNN below is written in plain PyTorch as a matrix multiply. That is not a workaround, it is the clearest way to see what message passing actually computes. Section 13 says what to switch to when the graph outgrows it.

------------------------------------------------------------------------

## 6. The Model Landscape (mid-2026)

| Model | Year | Aggregation | Inductive | Scales to | License | Best for |
|----|----|----|----|----|----|----|
| MLP on node features | \- | none | yes | anything | \- | **The baseline that tells you if edges matter** |
| node2vec / DeepWalk | 2014-16 | random walks | no | ~1M nodes | MIT | Unsupervised embeddings, no features needed |
| **GCN** | 2017 | degree-normalised sum | no (as published) | ~100k nodes full-batch | MIT | The default first GNN - **built below** |
| **GraphSAGE** | 2017 | mean/max + self term | **yes** | billions (sampling) | Apache 2.0 | **Production node tasks** - built below |
| **GAT / GATv2** | 2018/2021 | attention | yes | ~1M nodes | MIT | Neighbours of unequal importance - built below |
| **GIN** | 2019 | sum + MLP | yes | graph-level | MIT | **Graph classification** (max WL expressiveness) - built below |
| Cluster-GCN / GraphSAINT | 2019-20 | subgraph sampling | yes | 100M+ nodes | MIT | Training on graphs that do not fit in memory |
| SGC / correct-and-smooth | 2019-21 | precomputed propagation | partly | very large | MIT | Astonishingly strong for the compute; a real baseline |
| Graphormer | 2021 | global attention + structural bias | yes | small graphs | MIT | Molecules; won OGB-LSC PCQM4M |
| GraphGPS | 2022 | local MPNN + global attention | yes | medium | MIT | Best-of-both hybrid |
| MACE / NequIP / Allegro | 2022-23 | equivariant, geometric | yes | atoms | MIT | Interatomic potentials at DFT accuracy |
| Relational GCN / HGT | 2018/2020 | per-relation weights | yes | large | MIT | Heterogeneous graphs (many node/edge types) |

Leaderboards worth trusting: **[OGB](https://ogb.stanford.edu/docs/leader_overview/)** (fixed splits, realistic scale) and the [Long Range Graph Benchmark](https://github.com/vijaydwivedi75/lrgb). Treat any Cora/CiteSeer number without error bars over multiple random splits as decoration.

**What wins what.** For accuracy on molecules, graph transformers and equivariant GNNs lead. For node tasks at industrial scale, GraphSAGE-style sampling is what actually ships. For expressive power on graph-level tasks, GIN is the message-passing maximum and you need structural encodings to go past it. And for effort-adjusted performance on a graph you have not studied yet, the honest ranking starts with an MLP and hand-built neighbourhood features.

------------------------------------------------------------------------

## 7. Setup

Package roles:

- `torch` - every model here, written from scratch (no `torch_geometric` / DGL needed at this scale)
- `numpy` - adjacency construction and the splits
- `networkx` - graph statistics only (degree distribution, components, clustering); not used for learning
- `datasets` - PROTEINS from the Hugging Face Hub, for graph classification
- `scikit-learn` - ROC-AUC for link prediction
- `pyecharts` - all charts (repo rule), including the force-directed graph view

Cora (164 KB) and the PROTEINS cache land in `DL_tasks/datasets/`, which is gitignored. Memory: the dense normalised adjacency is 2708 x 2708 float32 = 29 MB, and the largest model here is about 92k parameters, so the whole notebook is a rounding error against the 12 GB card. The `free_memory()` discipline is still applied between sections, because that is the house rule and because it makes the pattern obvious when you scale the same notebook up.

------------------------------------------------------------------------

``` python
# Everything is plain PyTorch - torch_geometric and DGL are not needed at this scale.
# %pip install -q torch numpy networkx datasets scikit-learn pyecharts
```

``` python
import ctypes
import ctypes.util
import gc
import tarfile
import time
import urllib.request
from pathlib import Path

import numpy as np
import psutil
import torch
import torch.nn as nn
import torch.nn.functional as F
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


DATA_DIR = Path("../../datasets")        # gitignored
DATA_DIR.mkdir(exist_ok=True)
HF_CACHE = str(DATA_DIR / "hf_cache")

CORA_TGZ = DATA_DIR / "cora.tgz"
if not CORA_TGZ.exists():
    urllib.request.urlretrieve("https://linqs-data.soe.ucsc.edu/public/lbc/cora.tgz", CORA_TGZ)

torch.manual_seed(0)
memory_report("baseline")
```

    NVIDIA GeForce RTX 3060
    device: cuda:0
    RAM  baseline           15.26 / 20.97 GB
    VRAM baseline            0.00 GB allocated /  0.00 GB reserved

``` python
# cora.content: <paper_id>\t<1433 binary word features>\t<class name>
# cora.cites:   <cited_id>\t<citing_id>
with tarfile.open(CORA_TGZ) as tf:
    content = tf.extractfile("cora/cora.content").read().decode().strip().split("\n")
    cites = tf.extractfile("cora/cora.cites").read().decode().strip().split("\n")

paper_ids = [row.split("\t")[0] for row in content]
X = np.array([[int(v) for v in row.split("\t")[1:-1]] for row in content], dtype="float32")
class_names = sorted({row.split("\t")[-1] for row in content})
class_id = {name: i for i, name in enumerate(class_names)}
y = np.array([class_id[row.split("\t")[-1]] for row in content])

row_of = {pid: i for i, pid in enumerate(paper_ids)}
edges = np.array([[row_of[a], row_of[b]] for a, b in (line.split("\t") for line in cites)])

N, F_IN, N_CLASS = len(paper_ids), X.shape[1], len(class_names)
print(f"{N:,} papers, {len(edges):,} citations, {F_IN} bag-of-words features, "
      f"{N_CLASS} topics")
print("topics:", ", ".join(f"{n} ({(y == i).sum()})" for i, n in enumerate(class_names)))


def edge_homophily(e, labels):
    "Fraction of edges connecting two nodes with the same label."
    return float(np.mean(labels[e[:, 0]] == labels[e[:, 1]]))


print(f"\nedge homophily h = {edge_homophily(edges, y):.3f}  "
      f"(random baseline would be ~{sum((np.bincount(y) / N) ** 2):.3f})")
print("High homophily is why a GCN will work here - and why these numbers do not "
      "transfer to a fraud graph.")
```

    2,708 papers, 5,429 citations, 1433 bag-of-words features, 7 topics
    topics: Case_Based (298), Genetic_Algorithms (418), Neural_Networks (818), Probabilistic_Methods (426), Reinforcement_Learning (217), Rule_Learning (180), Theory (351)

    edge homophily h = 0.814  (random baseline would be ~0.180)
    High homophily is why a GCN will work here - and why these numbers do not transfer to a fraud graph.

``` python
import networkx as nx

# networkx for structure statistics only - no learning happens here.
G = nx.Graph()
G.add_nodes_from(range(N))
G.add_edges_from(edges.tolist())
G.remove_edges_from(nx.selfloop_edges(G))

components = sorted(nx.connected_components(G), key=len, reverse=True)
degrees = np.array([d for _, d in G.degree()])

print(f"nodes {G.number_of_nodes():,}   undirected edges {G.number_of_edges():,}")
print(f"connected components {len(components)}   largest {len(components[0]):,} nodes "
      f"({len(components[0]) / N:.1%})")
print(f"degree: mean {degrees.mean():.2f}  median {np.median(degrees):.0f}  "
      f"max {degrees.max()}  isolated {(degrees == 0).sum()}")
print(f"average clustering coefficient {nx.average_clustering(G):.4f}")
print(f"density {nx.density(G):.5f}  (a citation graph is extremely sparse)")

# A power-law-ish degree distribution is the norm for real graphs, and it is why
# degree normalisation (the D^-1/2 in GCN) matters: without it, hub nodes dominate.
counts = np.bincount(degrees)
print(f"\n{(degrees <= 3).mean():.1%} of papers have 3 or fewer citations; "
      f"the top node has {degrees.max()}")
```

    nodes 2,708   undirected edges 5,278
    connected components 78   largest 2,485 nodes (91.8%)
    degree: mean 3.90  median 3  max 168  isolated 0
    average clustering coefficient 0.2407
    density 0.00144  (a citation graph is extremely sparse)

    59.9% of papers have 3 or fewer citations; the top node has 168

``` python
from pyecharts.globals import CurrentConfig

# assets.pyecharts.org serves an expired TLS certificate from most nodes (2026-11-11),
# so charts failed to load. jsDelivr mirrors the same assets, pinned to one commit.
CurrentConfig.ONLINE_HOST = "https://cdn.jsdelivr.net/gh/pyecharts/pyecharts-assets@f2328d558550046fc5449672a6850e848e141590/assets/v6/"

from pyecharts import options as opts
from pyecharts.charts import Bar

deg_bar = (
    Bar()
    .add_xaxis([str(d) for d in range(1, 26)])
    .add_yaxis("papers", [int(counts[d]) if d < len(counts) else 0 for d in range(1, 26)],
               label_opts=opts.LabelOpts(is_show=False), category_gap="20%")
    .set_global_opts(
        title_opts=opts.TitleOpts(
            title="Cora degree distribution (1-25 citations)",
            subtitle=f"heavy tail: mean {degrees.mean():.1f}, max {degrees.max()} - "
                     "this is why GCN normalises by degree"),
        xaxis_opts=opts.AxisOpts(name="degree"),
        yaxis_opts=opts.AxisOpts(name="number of papers"),
        tooltip_opts=opts.TooltipOpts(trigger="axis"),
    )
)
deg_bar.render_notebook()
```

``` python
from pyecharts.charts import Graph

# A force-directed view of the largest 300-node subgraph, coloured by topic. Not a
# diagnostic - a reminder that "the structure is the data", and that the clusters a
# GCN exploits are visible to the eye.
sub_nodes = sorted(components[0], key=lambda i: -G.degree(i))[:300]
sub = G.subgraph(sub_nodes)
index = {n: k for k, n in enumerate(sub.nodes())}

gnodes = [{"name": str(n), "symbolSize": 4 + min(G.degree(n), 30) ** 0.6 * 2,
           "category": int(y[n]), "value": int(G.degree(n))} for n in sub.nodes()]
glinks = [{"source": str(a), "target": str(b)} for a, b in sub.edges()]

graph_chart = (
    Graph()
    .add("", gnodes, glinks,
         categories=[{"name": c} for c in class_names],
         layout="force", repulsion=60, is_draggable=True,
         linestyle_opts=opts.LineStyleOpts(width=0.4, opacity=0.35, curve=0.1),
         label_opts=opts.LabelOpts(is_show=False))
    .set_global_opts(
        title_opts=opts.TitleOpts(title="Cora: 300 highest-degree papers in the giant component",
                                  subtitle="colour = topic; same-colour clumps are the homophily a GCN uses"),
        legend_opts=opts.LegendOpts(pos_top="6%", type_="scroll"),
        tooltip_opts=opts.TooltipOpts(formatter="paper {b}: degree {c}"),
    )
)
graph_chart.render_notebook()
```

## 8. Building the Graph Tensors

Two decisions here carry the whole notebook.

**The normalised adjacency.** GCN’s propagation matrix is $\hat{A} = \tilde{D}^{-1/2}(A + I)\tilde{D}^{-1/2}$. The `+ I` (self-loops) keeps a node’s own features in its update - without it, a node is replaced by its neighbours rather than mixed with them. The symmetric $D^{-1/2}$ normalisation stops high-degree hubs from dominating every neighbour they touch, which matters a lot given the degree distribution above. Building it once and reusing it means each layer is a single matrix multiply.

**The split.** The standard Cora protocol is deliberately, brutally small: **20 labelled nodes per class** (140 of 2,708), 500 for validation, 1,000 for test. That is what makes it a semi-supervised benchmark - the model has to lean on structure because it barely has any labels. Note this is a **transductive** split: every node’s features and every edge are visible during training, only the labels are hidden.

The dense representation is 2708 x 2708 float32 = 29 MB. Past ~50k nodes this stops being viable and you move to sparse tensors or a sampling library; section 13 covers that.

------------------------------------------------------------------------

``` python
# Symmetric adjacency with self-loops, degree-normalised.
A = np.zeros((N, N), dtype="float32")
A[edges[:, 0], edges[:, 1]] = 1.0
A[edges[:, 1], edges[:, 0]] = 1.0      # citations are directed; the benchmark uses them undirected
np.fill_diagonal(A, 1.0)               # self-loops: keep a node's own features in its update

deg = A.sum(1)
A_hat = A / np.sqrt(np.outer(deg, deg))   # D^-1/2 (A + I) D^-1/2

print(f"dense adjacency {A.shape} = {A.nbytes / 1e6:.0f} MB "
      f"({(A > 0).mean():.4%} non-zero - sparse in reality, dense here for clarity)")

# Row-normalise the bag-of-words features (standard for Cora).
X_norm = X / np.maximum(X.sum(1, keepdims=True), 1)

Xt = torch.tensor(X_norm, device=device)
At = torch.tensor(A_hat, device=device)
yt = torch.tensor(y, device=device)

# The standard split: 20 labelled nodes per class, 500 val, 1000 test.
rng = np.random.default_rng(0)
train_idx = np.concatenate([rng.permutation(np.where(y == c)[0])[:20] for c in range(N_CLASS)])
remaining = rng.permutation(np.setdiff1d(np.arange(N), train_idx))
val_idx, test_idx = remaining[:500], remaining[500:1500]

TRAIN = torch.tensor(train_idx, device=device)
VAL = torch.tensor(val_idx, device=device)
TEST = torch.tensor(test_idx, device=device)
print(f"train {len(train_idx)} nodes ({len(train_idx) / N:.1%})  "
      f"val {len(val_idx)}  test {len(test_idx)}")
vram("graph tensors")
```

    dense adjacency (2708, 2708) = 29 MB (0.1809% non-zero - sparse in reality, dense here for clarity)
    train 140 nodes (5.2%)  val 500  test 1000
    VRAM graph tensors       0.04 GB allocated /  0.05 GB reserved

## 9. Head-to-head Benchmark: Four GNNs From Scratch

Each model below is the same message-passing skeleton with a different AGGREGATE, and each is short enough to read in one sitting. `A_hat @ (X W)` *is* the message passing: row $v$ of the product is the normalised sum of $W$-transformed features over $v$’s neighbours plus itself.

- **MLP** - no adjacency at all. The control. If a GNN cannot beat this, the edges were not carrying information.
- **GCN** (Kipf and Welling, 2017) - `A_hat @ (X W)`. Self and neighbours are mixed with fixed degree-derived weights.
- **GraphSAGE** (Hamilton et al., 2017) - mean over neighbours through one weight matrix, the node’s own features through a *separate* one, concatenated. Keeping the self term distinct is what makes it robust when neighbours are less informative, and the sampled version is what scales to billions of edges.
- **GAT** (Velickovic et al., 2018) - learn a scalar attention weight per edge and take a weighted sum. Written here as dense masked attention, which is exactly the formulation and exactly the wrong implementation past a few thousand nodes (it materialises an $N \times N$ matrix per head).
- **GIN** (Xu et al., 2019) - `MLP((1 + eps) * x_v + sum of neighbours)`. Sum rather than mean, because mean and max lose multiset information; this is the aggregation that reaches 1-WL expressiveness.

Two layers each, because a third makes every node’s receptive field cover most of the graph and accuracy falls - the oversmoothing result, measured directly in section 11.

------------------------------------------------------------------------

``` python
class MLP(nn.Module):
    "No graph at all - the control that says whether the edges matter."

    def __init__(self, f_in, hidden, n_class, p=0.5):
        super().__init__()
        self.l1, self.l2, self.p = nn.Linear(f_in, hidden), nn.Linear(hidden, n_class), p

    def forward(self, x, a_hat):
        x = F.relu(self.l1(F.dropout(x, self.p, self.training)))
        return self.l2(F.dropout(x, self.p, self.training))


class GCN(nn.Module):
    "Kipf and Welling 2017: H' = sigma(A_hat H W). Message passing in one matmul."

    def __init__(self, f_in, hidden, n_class, p=0.5):
        super().__init__()
        self.l1, self.l2, self.p = nn.Linear(f_in, hidden), nn.Linear(hidden, n_class), p

    def forward(self, x, a_hat):
        x = F.relu(a_hat @ self.l1(F.dropout(x, self.p, self.training)))
        return a_hat @ self.l2(F.dropout(x, self.p, self.training))


class SAGELayer(nn.Module):
    "GraphSAGE: mean over neighbours, plus a SEPARATE weight for the node itself."

    def __init__(self, f_in, f_out):
        super().__init__()
        self.self_w = nn.Linear(f_in, f_out)
        self.neigh_w = nn.Linear(f_in, f_out)

    def forward(self, x, a_mean):
        return self.self_w(x) + self.neigh_w(a_mean @ x)


class SAGE(nn.Module):
    def __init__(self, f_in, hidden, n_class, p=0.5):
        super().__init__()
        self.l1, self.l2, self.p = SAGELayer(f_in, hidden), SAGELayer(hidden, n_class), p

    def forward(self, x, a_mean):
        x = F.relu(self.l1(F.dropout(x, self.p, self.training), a_mean))
        x = F.normalize(x, p=2, dim=1)                 # the paper's L2 row normalisation
        return self.l2(F.dropout(x, self.p, self.training), a_mean)


class GATLayer(nn.Module):
    """GAT: a learned scalar weight per edge, softmaxed over each node's neighbours.

    Dense masked attention is the honest formulation and the wrong implementation:
    it builds an N x N score matrix per head. Fine at 2,708 nodes, hopeless at 1M -
    real implementations scatter over the edge list instead.
    """

    def __init__(self, f_in, f_out, heads=8, concat=True, p=0.6):
        super().__init__()
        self.heads, self.f_out, self.concat, self.p = heads, f_out, concat, p
        self.w = nn.Linear(f_in, heads * f_out, bias=False)
        self.a_src = nn.Parameter(torch.empty(heads, f_out))
        self.a_dst = nn.Parameter(torch.empty(heads, f_out))
        nn.init.xavier_uniform_(self.w.weight)
        nn.init.xavier_uniform_(self.a_src)
        nn.init.xavier_uniform_(self.a_dst)

    def forward(self, x, mask):
        n = x.shape[0]
        h = self.w(x).view(n, self.heads, self.f_out).transpose(0, 1)      # (heads, N, f_out)
        # e_ij = LeakyReLU(a_src . h_i + a_dst . h_j), broadcast into an (N, N) score matrix
        src = (h * self.a_src[:, None, :]).sum(-1, keepdim=True)           # (heads, N, 1)
        dst = (h * self.a_dst[:, None, :]).sum(-1, keepdim=True)
        e = F.leaky_relu(src + dst.transpose(1, 2), negative_slope=0.2)
        e = e.masked_fill(~mask, float("-inf"))                            # non-edges get no weight
        alpha = F.dropout(torch.softmax(e, dim=-1), self.p, self.training)
        out = alpha @ h                                                    # (heads, N, f_out)
        return out.transpose(0, 1).reshape(n, -1) if self.concat else out.mean(0)


class GAT(nn.Module):
    def __init__(self, f_in, hidden, n_class, heads=8, p=0.6):
        super().__init__()
        self.l1 = GATLayer(f_in, hidden, heads=heads, concat=True, p=p)
        self.l2 = GATLayer(hidden * heads, n_class, heads=1, concat=False, p=p)
        self.p = p

    def forward(self, x, mask):
        x = F.elu(self.l1(F.dropout(x, self.p, self.training), mask))
        return self.l2(F.dropout(x, self.p, self.training), mask)


class GIN(nn.Module):
    "Xu et al. 2019: MLP((1 + eps) x_v + SUM of neighbours). Sum, not mean - that is the point."

    def __init__(self, f_in, hidden, n_class, p=0.5):
        super().__init__()
        self.eps1, self.eps2 = nn.Parameter(torch.zeros(1)), nn.Parameter(torch.zeros(1))
        self.mlp1 = nn.Sequential(nn.Linear(f_in, hidden), nn.ReLU(), nn.Linear(hidden, hidden))
        self.mlp2 = nn.Sequential(nn.Linear(hidden, hidden), nn.ReLU(),
                                  nn.Linear(hidden, n_class))
        self.p = p

    def forward(self, x, a_sum):
        x = F.dropout(x, self.p, self.training)
        x = F.relu(self.mlp1((1 + self.eps1) * x + a_sum @ x))
        x = F.dropout(x, self.p, self.training)
        return self.mlp2((1 + self.eps2) * x + a_sum @ x)


for cls in (MLP, GCN, SAGE, GAT, GIN):
    m = cls(F_IN, 64, N_CLASS)
    print(f"{cls.__name__:6s} {sum(p.numel() for p in m.parameters()):>8,} parameters")
    del m
free_memory()
```

    MLP      92,231 parameters
    GCN      92,231 parameters
    SAGE    184,462 parameters
    GAT     738,318 parameters
    GIN     100,553 parameters

``` python
# Each architecture wants a different propagation operator, built once here.
A_raw = A.copy()
np.fill_diagonal(A_raw, 0.0)                                  # GIN/SAGE handle self separately
deg_raw = np.maximum(A_raw.sum(1, keepdims=True), 1)

PROP = {
    "MLP": At,                                                # ignored by the model
    "GCN": At,                                                # D^-1/2 (A+I) D^-1/2
    "GraphSAGE": torch.tensor(A_raw / deg_raw, device=device),          # neighbour MEAN
    "GAT": torch.tensor(A > 0, device=device)[None],          # boolean mask, (1, N, N)
    "GIN": torch.tensor(A_raw, device=device),                # neighbour SUM
}
ARCH = {"MLP": MLP, "GCN": GCN, "GraphSAGE": SAGE, "GAT": GAT, "GIN": GIN}


def train_node_model(name, hidden=64, epochs=200, lr=0.01, weight_decay=5e-4, seed=0):
    """Full-batch training with early model selection on the validation split.

    Full-batch is only possible because the graph is tiny; the loss is computed on
    140 labelled nodes but the forward pass touches every node, which is exactly what
    makes this transductive.
    """
    torch.manual_seed(seed)
    model = ARCH[name](F_IN, hidden, N_CLASS).to(device)
    prop = PROP[name]
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    best_val, best_test, curve = 0.0, 0.0, []
    t0 = time.perf_counter()
    for epoch in range(epochs):
        model.train()
        loss = F.cross_entropy(model(Xt, prop)[TRAIN], yt[TRAIN])
        opt.zero_grad()
        loss.backward()
        opt.step()

        model.eval()
        with torch.inference_mode():
            logits = model(Xt, prop)
            val_acc = (logits[VAL].argmax(1) == yt[VAL]).float().mean().item()
            test_acc = (logits[TEST].argmax(1) == yt[TEST]).float().mean().item()
        curve.append((loss.item(), val_acc))
        if val_acc > best_val:                    # select on val, report the matching test
            best_val, best_test = val_acc, test_acc

    seconds = time.perf_counter() - t0
    n_params = sum(p.numel() for p in model.parameters())
    del model, opt
    free_memory()
    return dict(model=name, val_acc=best_val, test_acc=best_test,
                seconds=seconds, params=n_params, curve=curve)
```

``` python
# Three seeds each: a single Cora run swings by 1-2 points, and a lot of the
# literature's reported gaps are inside that noise.
SEEDS = [0, 1, 2]
node_results, curves = [], {}
for name in ARCH:
    runs = [train_node_model(name, seed=s) for s in SEEDS]
    accs = np.array([r["test_acc"] for r in runs])
    curves[name] = runs[0]["curve"]
    node_results.append(dict(model=name,
                             test_acc=float(accs.mean()), test_std=float(accs.std()),
                             val_acc=float(np.mean([r["val_acc"] for r in runs])),
                             seconds=float(np.mean([r["seconds"] for r in runs])),
                             params=runs[0]["params"]))
    print(f"{name:10s} test {accs.mean():.3f} +/- {accs.std():.3f}   "
          f"{node_results[-1]['seconds']:5.1f}s   {runs[0]['params']:>7,} params")

memory_report("after node models")
```

    MLP        test 0.616 +/- 0.006     0.4s    92,231 params
    GCN        test 0.814 +/- 0.000     0.4s    92,231 params
    GraphSAGE  test 0.793 +/- 0.004     1.6s   184,462 params
    GAT        test 0.817 +/- 0.002     8.7s   738,318 params
    GIN        test 0.765 +/- 0.007     1.6s   100,553 params
    RAM  after node models  16.16 / 20.97 GB
    VRAM after node models   0.13 GB allocated /  0.15 GB reserved

``` python
import pandas as pd

node_bench = pd.DataFrame(node_results).sort_values("test_acc", ascending=False).reset_index(drop=True)
mlp_acc = node_bench.loc[node_bench["model"] == "MLP", "test_acc"].iloc[0]
node_bench["gain_over_mlp"] = node_bench["test_acc"] - mlp_acc
print(f"the MLP ignores every edge and scores {mlp_acc:.3f}; "
      f"the gap above it is what the graph structure is worth on Cora")
node_bench.round(4)
```

    the MLP ignores every edge and scores 0.616; the gap above it is what the graph structure is worth on Cora

|     | model     | test_acc | test_std | val_acc | seconds | params | gain_over_mlp |
|-----|-----------|----------|----------|---------|---------|--------|---------------|
| 0   | GAT       | 0.8170   | 0.0022   | 0.8080  | 8.6689  | 738318 | 0.2013        |
| 1   | GCN       | 0.8143   | 0.0005   | 0.8093  | 0.3800  | 92231  | 0.1987        |
| 2   | GraphSAGE | 0.7930   | 0.0037   | 0.7920  | 1.5914  | 184462 | 0.1773        |
| 3   | GIN       | 0.7653   | 0.0066   | 0.7767  | 1.5994  | 100553 | 0.1497        |
| 4   | MLP       | 0.6157   | 0.0058   | 0.5960  | 0.3827  | 92231  | 0.0000        |

``` python
acc_bar = (
    Bar()
    .add_xaxis(node_bench["model"].tolist())
    .add_yaxis("test accuracy", [round(float(v), 4) for v in node_bench["test_acc"]],
               label_opts=opts.LabelOpts(is_show=False))
    .set_global_opts(
        title_opts=opts.TitleOpts(
            title="Cora node classification: mean of 3 seeds",
            subtitle="140 labelled nodes (20 per class); the MLP bar is the no-graph control"),
        xaxis_opts=opts.AxisOpts(name="model"),
        yaxis_opts=opts.AxisOpts(name="test accuracy", min_=0.5, max_=0.9),
        tooltip_opts=opts.TooltipOpts(trigger="axis"),
    )
)
acc_bar.render_notebook()
```

``` python
from pyecharts.charts import Line

conv = Line().add_xaxis([str(e) for e in range(1, len(curves["GCN"]) + 1)])
for name, curve in curves.items():
    conv.add_yaxis(name, [round(float(v), 4) for _, v in curve], is_smooth=True,
                   symbol="none", label_opts=opts.LabelOpts(is_show=False))
conv.set_global_opts(
    title_opts=opts.TitleOpts(title="Validation accuracy during training (seed 0)",
                              subtitle="the graph models separate from the MLP within ~20 epochs"),
    xaxis_opts=opts.AxisOpts(name="epoch", axislabel_opts=opts.LabelOpts(interval=19)),
    yaxis_opts=opts.AxisOpts(name="validation accuracy", min_=0.1, max_=0.9),
    tooltip_opts=opts.TooltipOpts(trigger="axis"),
    legend_opts=opts.LegendOpts(pos_top="8%"),
)
conv.render_notebook()
```

## 10. Link Prediction: predicting the edges themselves

Same graph, different question: given two papers, would one cite the other? The recipe is an **encoder-decoder**: a GCN encodes every node into a vector, and the decoder scores a candidate edge as the dot product of its endpoints. Train it with binary cross-entropy against sampled negatives.

The one detail that decides whether the number means anything: **the held-out positive edges must be removed from the adjacency the encoder propagates over**, not merely excluded from the loss. Leave them in and the encoder can read the answer straight off its own input, and you get a suspiciously excellent AUC that vanishes in production. The cell builds a separate `A_train` for exactly this reason.

The second detail is **negative sampling**. Uniformly random node pairs are almost all far apart and trivially separable, so the AUC below is an optimistic ceiling. The cell also scores against **hard negatives** - non-edges drawn from 2-hop neighbourhoods, which is much closer to the real task of “which of these plausible candidates is real”.

------------------------------------------------------------------------

``` python
from sklearn.metrics import average_precision_score, roc_auc_score

# Undirected edge list, deduplicated (i < j).
und = np.unique(np.sort(edges, axis=1), axis=0)
und = und[und[:, 0] != und[:, 1]]
perm = np.random.default_rng(0).permutation(len(und))
n_test = int(0.10 * len(und))
test_edges, train_edges = und[perm[:n_test]], und[perm[n_test:]]

# The encoder must NOT see the held-out edges. Rebuild the propagation matrix from
# the training edges only - this is the step link-prediction code most often skips.
A_tr = np.zeros((N, N), dtype="float32")
A_tr[train_edges[:, 0], train_edges[:, 1]] = 1.0
A_tr[train_edges[:, 1], train_edges[:, 0]] = 1.0
np.fill_diagonal(A_tr, 1.0)
deg_tr = A_tr.sum(1)
A_tr_hat = torch.tensor(A_tr / np.sqrt(np.outer(deg_tr, deg_tr)), device=device)

print(f"{len(und):,} undirected edges -> {len(train_edges):,} for message passing + training, "
      f"{len(test_edges):,} held out")

rng_neg = np.random.default_rng(1)
existing = set(map(tuple, und.tolist()))


def sample_easy_negatives(k):
    "Uniformly random non-edges - the standard, and the easy, negative set."
    out = []
    while len(out) < k:
        a, b = rng_neg.integers(0, N, 2)
        if a != b and (min(a, b), max(a, b)) not in existing:
            out.append((a, b))
    return np.array(out)


def sample_hard_negatives(k):
    "Non-edges between nodes that share a neighbour - plausible, and much harder."
    out = []
    tries = 0
    while len(out) < k and tries < k * 200:
        tries += 1
        a = rng_neg.integers(0, N)
        nb = np.where(A_tr[a] > 0)[0]
        if len(nb) < 2:
            continue
        two_hop = np.where(A_tr[rng_neg.choice(nb)] > 0)[0]
        b = int(rng_neg.choice(two_hop))
        if a != b and (min(a, b), max(a, b)) not in existing:
            out.append((a, b))
    return np.array(out)


neg_train = sample_easy_negatives(len(train_edges))
neg_test_easy = sample_easy_negatives(len(test_edges))
neg_test_hard = sample_hard_negatives(len(test_edges))
print(f"negatives: {len(neg_train):,} for training, {len(neg_test_easy):,} easy "
      f"and {len(neg_test_hard):,} hard for evaluation")
```

    5,278 undirected edges -> 4,751 for message passing + training, 527 held out
    negatives: 4,751 for training, 527 easy and 527 hard for evaluation

``` python
class GCNEncoder(nn.Module):
    "Two GCN layers producing a node embedding; the decoder is a plain dot product."

    def __init__(self, f_in, hidden, out_dim):
        super().__init__()
        self.l1, self.l2 = nn.Linear(f_in, hidden), nn.Linear(hidden, out_dim)

    def forward(self, x, a_hat):
        return a_hat @ self.l2(F.relu(a_hat @ self.l1(x)))


def edge_scores(z, pairs):
    "Dot-product decoder: score(u, v) = z_u . z_v."
    return (z[pairs[:, 0]] * z[pairs[:, 1]]).sum(-1)


torch.manual_seed(0)
encoder = GCNEncoder(F_IN, 128, 64).to(device)
opt = torch.optim.Adam(encoder.parameters(), lr=0.01, weight_decay=5e-4)

pos_t = torch.tensor(train_edges, device=device)
neg_t = torch.tensor(neg_train, device=device)
target = torch.cat([torch.ones(len(pos_t)), torch.zeros(len(neg_t))]).to(device)

t0 = time.perf_counter()
for epoch in range(200):
    encoder.train()
    z = encoder(Xt, A_tr_hat)
    logits = torch.cat([edge_scores(z, pos_t), edge_scores(z, neg_t)])
    loss = F.binary_cross_entropy_with_logits(logits, target)
    opt.zero_grad()
    loss.backward()
    opt.step()
print(f"trained in {time.perf_counter() - t0:.1f}s, final loss {loss.item():.4f}")

encoder.eval()
with torch.inference_mode():
    z = encoder(Xt, A_tr_hat)
    pos_score = edge_scores(z, torch.tensor(test_edges, device=device)).cpu().numpy()
    for label, negs in (("easy (uniform)", neg_test_easy), ("hard (2-hop)", neg_test_hard)):
        neg_score = edge_scores(z, torch.tensor(negs, device=device)).cpu().numpy()
        s = np.concatenate([pos_score, neg_score])
        t = np.concatenate([np.ones(len(pos_score)), np.zeros(len(neg_score))])
        print(f"held-out edges vs {label:16s} negatives: "
              f"ROC-AUC {roc_auc_score(t, s):.4f}   AP {average_precision_score(t, s):.4f}")

print("\nThe gap between those two rows is how much of a link-prediction score is "
      "the negative sampler rather than the model.")
del encoder, opt, z
free_memory()
vram("after link prediction")
```

    trained in 0.4s, final loss 0.5282
    held-out edges vs easy (uniform)   negatives: ROC-AUC 0.7617   AP 0.7669
    held-out edges vs hard (2-hop)     negatives: ROC-AUC 0.5355   AP 0.5821

    The gap between those two rows is how much of a link-prediction score is the negative sampler rather than the model.
    VRAM after link prediction  0.16 GB allocated /  0.18 GB reserved

## 11. Oversmoothing: why every deployed GNN is shallow

Each message-passing layer mixes a node with its neighbours. After $k$ layers a node’s representation is a weighted average over its entire $k$-hop neighbourhood - and on a small-world graph like Cora, 4 hops already reaches most of the giant component. Every node ends up with nearly the same vector, the classes become inseparable, and accuracy collapses. This is **oversmoothing** (Li et al., 2018), and it is the reason a GNN is not made better by making it deeper.

The cell trains GCNs of depth 1 to 6 and measures two things: test accuracy, and a direct measure of the collapse - the **mean pairwise cosine distance** between final-layer node representations, where a smaller value means the nodes look more alike. Both peak at depth 2 and then fall together: the accuracy loss and the representational collapse are the same event, seen from two sides. Depth 1 is worse for the opposite reason - one hop is not enough neighbourhood to be useful.

The known mitigations - residual connections, jumping knowledge, PairNorm, and simply precomputing the propagation (SGC) - all attack the same failure, and none of them makes deep message passing genuinely useful the way depth is useful in a CNN.

------------------------------------------------------------------------

``` python
class DeepGCN(nn.Module):
    "A GCN of arbitrary depth, and a hook to look at the representation before the classifier."

    def __init__(self, f_in, hidden, n_class, depth, p=0.5):
        super().__init__()
        dims = [f_in] + [hidden] * (depth - 1) + [n_class]
        self.layers = nn.ModuleList(nn.Linear(dims[i], dims[i + 1]) for i in range(depth))
        self.p = p

    def forward(self, x, a_hat, return_hidden=False):
        h = x
        for i, layer in enumerate(self.layers):
            h = a_hat @ layer(F.dropout(h, self.p, self.training))
            if i < len(self.layers) - 1:
                h = F.relu(h)
            elif return_hidden:
                return h
        return h


def mean_pairwise_distance(h, sample=800):
    "Mean cosine distance between node representations - 0 means total collapse."
    idx = torch.randperm(h.shape[0], device=h.device)[:sample]
    v = F.normalize(h[idx], dim=1)
    sim = v @ v.T
    off = ~torch.eye(len(idx), dtype=torch.bool, device=h.device)
    return float((1 - sim[off]).mean())


depth_rows = []
for depth in range(1, 7):
    torch.manual_seed(0)
    model = DeepGCN(F_IN, 64, N_CLASS, depth).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
    best_val, best_test = 0.0, 0.0
    for epoch in range(200):
        model.train()
        loss = F.cross_entropy(model(Xt, At)[TRAIN], yt[TRAIN])
        opt.zero_grad()
        loss.backward()
        opt.step()
        model.eval()
        with torch.inference_mode():
            logits = model(Xt, At)
            va = (logits[VAL].argmax(1) == yt[VAL]).float().mean().item()
            if va > best_val:
                best_val = va
                best_test = (logits[TEST].argmax(1) == yt[TEST]).float().mean().item()
    with torch.inference_mode():
        dist = mean_pairwise_distance(model(Xt, At, return_hidden=True))
    depth_rows.append(dict(depth=depth, test_acc=best_test, mean_cosine_distance=dist))
    print(f"depth {depth}  test {best_test:.3f}   mean pairwise cosine distance {dist:.4f}")
    del model, opt
    free_memory()

depth_df = pd.DataFrame(depth_rows)
memory_report("after depth sweep")
```

    depth 1  test 0.719   mean pairwise cosine distance 0.7895
    depth 2  test 0.814   mean pairwise cosine distance 0.8997
    depth 3  test 0.787   mean pairwise cosine distance 0.8575
    depth 4  test 0.774   mean pairwise cosine distance 0.7628
    depth 5  test 0.731   mean pairwise cosine distance 0.6648
    depth 6  test 0.640   mean pairwise cosine distance 0.7013
    RAM  after depth sweep  16.30 / 20.97 GB
    VRAM after depth sweep   0.16 GB allocated /  0.18 GB reserved

``` python
smooth = (
    Line()
    .add_xaxis([str(d) for d in depth_df["depth"]])
    .add_yaxis("test accuracy", [round(float(v), 4) for v in depth_df["test_acc"]],
               is_smooth=True, symbol="circle", symbol_size=8,
               label_opts=opts.LabelOpts(is_show=False))
    .add_yaxis("mean pairwise cosine distance",
               [round(float(v), 4) for v in depth_df["mean_cosine_distance"]],
               is_smooth=True, symbol="circle", symbol_size=8, yaxis_index=1,
               label_opts=opts.LabelOpts(is_show=False))
    .extend_axis(yaxis=opts.AxisOpts(name="representation spread", min_=0,
                                     position="right"))
    .set_global_opts(
        title_opts=opts.TitleOpts(
            title="Oversmoothing: accuracy falls as representations collapse",
            subtitle="every extra layer averages over a wider neighbourhood until all nodes look alike"),
        xaxis_opts=opts.AxisOpts(name="GCN layers"),
        yaxis_opts=opts.AxisOpts(name="test accuracy", min_=0),
        tooltip_opts=opts.TooltipOpts(trigger="axis"),
        legend_opts=opts.LegendOpts(pos_top="8%"),
    )
)
smooth.render_notebook()
```

## 12. Graph Classification: one label per graph

The third task level. Instead of predicting per node, encode a whole graph into one vector and classify that. The extra ingredient is a **readout** (or pooling) function - sum, mean or max over all node representations - and it must be permutation invariant, which is why it is a simple symmetric aggregation rather than anything learned over an ordering.

**Sum readout is not a detail.** Mean pooling throws away graph size, and max pooling throws away multiplicity; sum keeps both, which is why GIN specifies it. On PROTEINS - 1,113 proteins as graphs of secondary-structure elements, labelled enzyme or not - graph size is genuinely informative, so this choice is measurable.

The realistic caveat: with ~1,000 graphs, a single train/test split has an error bar of several points. The standard protocol is 10-fold cross-validation, and Errico et al. (2020) showed that under it, much of the reported progress on TU datasets disappears. The cell below runs 5 folds - enough to show the spread, not enough to settle anything.

**Watch the last baseline in the output.** A logistic regression on a single feature - the number of nodes in the graph - lands within a point or two of the GIN. That is the Errico result in miniature: PROTEINS labels correlate strongly with graph size, so most of what a GNN scores here can be obtained without looking at a single edge. It does not mean GIN is useless; it means a PROTEINS number reported without that baseline tells you almost nothing.

------------------------------------------------------------------------

``` python
from datasets import load_dataset

# PROTEINS: each row is one graph - edge_index (2, E), node_feat (n, 3), y (1,).
proteins = load_dataset("graphs-datasets/PROTEINS", split="train", cache_dir=HF_CACHE)
sizes = np.array(proteins["num_nodes"])
labels_g = np.array([int(np.ravel(v)[0]) for v in proteins["y"]])

print(f"{len(proteins):,} graphs   nodes: median {np.median(sizes):.0f}, "
      f"mean {sizes.mean():.1f}, max {sizes.max()}")
print(f"labels: {np.bincount(labels_g)} (enzyme / not)")
print(f"node features: {np.array(proteins[0]['node_feat']).shape[1]} per node")
print(f"\nmean size by class: {[round(float(sizes[labels_g == c].mean()), 1) for c in (0, 1)]}"
      "  <- graph size alone carries signal, which is why SUM readout beats MEAN here")
```

    1,113 graphs   nodes: median 26, mean 39.1, max 620
    labels: [663 450] (enzyme / not)
    node features: 3 per node

    mean size by class: [50.0, 22.9]  <- graph size alone carries signal, which is why SUM readout beats MEAN here

``` python
# Dense per-graph tensors, padded to the largest graph in the batch. At ~39 nodes
# median this is cheap; a real pipeline uses a block-diagonal sparse batch instead.
def to_dense(record):
    "One PROTEINS record -> (node features, neighbour-sum adjacency, n_nodes)."
    n = int(record["num_nodes"])
    feat = np.asarray(record["node_feat"], dtype="float32").reshape(n, -1)
    ei = np.asarray(record["edge_index"], dtype="int64").reshape(2, -1)
    adj = np.zeros((n, n), dtype="float32")
    adj[ei[0], ei[1]] = 1.0
    adj[ei[1], ei[0]] = 1.0
    np.fill_diagonal(adj, 0.0)              # GIN adds the self term with its own eps
    return feat, adj, n


GRAPHS = [to_dense(r) for r in proteins]
F_G = GRAPHS[0][0].shape[1]
MAXN = max(g[2] for g in GRAPHS)
print(f"{len(GRAPHS)} graphs, {F_G} node features, largest graph {MAXN} nodes")


def make_batch(items, dev):
    "Pad a list of graphs into (feats, adjs, mask) tensors."
    m = max(g[2] for g in items)
    fb = np.zeros((len(items), m, F_G), dtype="float32")
    ab = np.zeros((len(items), m, m), dtype="float32")
    mb = np.zeros((len(items), m, 1), dtype="float32")
    for i, (feat, adj, n) in enumerate(items):
        fb[i, :n], ab[i, :n, :n], mb[i, :n] = feat, adj, 1.0
    return (torch.tensor(fb, device=dev), torch.tensor(ab, device=dev),
            torch.tensor(mb, device=dev))


class GraphGIN(nn.Module):
    "GIN layers + a permutation-invariant readout over nodes."

    def __init__(self, f_in, hidden, n_class=2, layers=3, readout="sum", p=0.5):
        super().__init__()
        self.readout, self.p = readout, p
        self.eps = nn.ParameterList(nn.Parameter(torch.zeros(1)) for _ in range(layers))
        dims = [f_in] + [hidden] * layers
        self.mlps = nn.ModuleList(
            nn.Sequential(nn.Linear(dims[i], hidden), nn.ReLU(),
                          nn.Linear(hidden, hidden), nn.BatchNorm1d(hidden), nn.ReLU())
            for i in range(layers))
        self.head = nn.Linear(hidden, n_class)

    def forward(self, x, adj, mask):
        b, n, _ = x.shape
        for eps, mlp in zip(self.eps, self.mlps):
            x = (1 + eps) * x + adj @ x                 # sum aggregation
            x = mlp(x.reshape(b * n, -1)).reshape(b, n, -1) * mask   # re-mask the padding
        pooled = x.sum(1) if self.readout == "sum" else x.sum(1) / mask.sum(1).clamp(min=1)
        return self.head(F.dropout(pooled, self.p, self.training))
```

    1113 graphs, 3 node features, largest graph 620 nodes

``` python
from sklearn.model_selection import StratifiedKFold

BATCH = 64


def run_fold(train_ids, test_ids, readout, epochs=40, seed=0):
    torch.manual_seed(seed)
    model = GraphGIN(F_G, 64, readout=readout).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=0.01)
    sched = torch.optim.lr_scheduler.StepLR(opt, step_size=25, gamma=0.5)
    y_g = torch.tensor(labels_g, device=device)

    for _ in range(epochs):
        model.train()
        order = np.random.default_rng(seed).permutation(train_ids)
        for s in range(0, len(order), BATCH):
            ids = order[s: s + BATCH]
            if len(ids) < 2:                       # BatchNorm needs more than one row
                continue
            fb, ab, mb = make_batch([GRAPHS[i] for i in ids], device)
            loss = F.cross_entropy(model(fb, ab, mb), y_g[torch.tensor(ids, device=device)])
            opt.zero_grad()
            loss.backward()
            opt.step()
        sched.step()

    model.eval()
    correct = 0
    with torch.inference_mode():
        for s in range(0, len(test_ids), BATCH):
            ids = test_ids[s: s + BATCH]
            fb, ab, mb = make_batch([GRAPHS[i] for i in ids], device)
            pred = model(fb, ab, mb).argmax(1)
            correct += (pred == y_g[torch.tensor(ids, device=device)]).sum().item()
    acc = correct / len(test_ids)
    del model, opt
    free_memory()
    return acc


skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
folds = list(skf.split(np.zeros(len(GRAPHS)), labels_g))

graph_rows = []
for readout in ("sum", "mean"):
    t0 = time.perf_counter()
    accs = [run_fold(tr, te, readout) for tr, te in folds]
    graph_rows.append(dict(model=f"GIN ({readout} readout)", mean_acc=float(np.mean(accs)),
                           std=float(np.std(accs)), seconds=time.perf_counter() - t0))
    print(f"GIN {readout:4s} readout: {np.mean(accs):.3f} +/- {np.std(accs):.3f}   "
          f"folds {[round(a, 3) for a in accs]}")

# The baseline that keeps graph learning honest: predict from graph size alone.
from sklearn.linear_model import LogisticRegression

size_accs = []
for tr, te in folds:
    clf = LogisticRegression(max_iter=1000).fit(sizes[tr].reshape(-1, 1), labels_g[tr])
    size_accs.append(clf.score(sizes[te].reshape(-1, 1), labels_g[te]))
graph_rows.append(dict(model="logistic regression on graph size",
                       mean_acc=float(np.mean(size_accs)), std=float(np.std(size_accs)),
                       seconds=0.0))
print(f"\nnode-count-only baseline: {np.mean(size_accs):.3f} +/- {np.std(size_accs):.3f}"
      "  <- the number every PROTEINS result should be reported against")

memory_report("after graph classification")
pd.DataFrame(graph_rows).round(4)
```

    GIN sum  readout: 0.721 +/- 0.020   folds [0.735, 0.682, 0.726, 0.725, 0.734]
    GIN mean readout: 0.692 +/- 0.030   folds [0.704, 0.641, 0.682, 0.703, 0.73]

    node-count-only baseline: 0.712 +/- 0.009  <- the number every PROTEINS result should be reported against
    RAM  after graph classification 16.38 / 20.97 GB
    VRAM after graph classification  0.16 GB allocated /  0.18 GB reserved

|     | model                             | mean_acc | std    | seconds |
|-----|-----------------------------------|----------|--------|---------|
| 0   | GIN (sum readout)                 | 0.7206   | 0.0199 | 19.2481 |
| 1   | GIN (mean readout)                | 0.6919   | 0.0295 | 19.1234 |
| 2   | logistic regression on graph size | 0.7116   | 0.0092 | 0.0000  |

``` python
graph_bar = (
    Bar()
    .add_xaxis([r["model"] for r in graph_rows])
    .add_yaxis("5-fold accuracy", [round(r["mean_acc"], 4) for r in graph_rows],
               label_opts=opts.LabelOpts(is_show=False))
    .set_series_opts(markline_opts=opts.MarkLineOpts(
        data=[opts.MarkLineItem(y=round(float(np.bincount(labels_g).max() / len(labels_g)), 4),
                                name="majority class")]))
    .set_global_opts(
        title_opts=opts.TitleOpts(
            title="PROTEINS graph classification, 5-fold cross-validation",
            subtitle="the marked line is the majority-class rate; error bars across folds are +/- 2-3 points"),
        xaxis_opts=opts.AxisOpts(axislabel_opts=opts.LabelOpts(rotate=15, font_size=10)),
        yaxis_opts=opts.AxisOpts(name="accuracy", min_=0.4, max_=0.85),
        tooltip_opts=opts.TooltipOpts(trigger="axis"),
    )
)
graph_bar.render_notebook()
```

## 13. Common Frameworks

Graph learning’s ecosystem is shaped by one constraint: the dense adjacency matrix used above for clarity stops being viable somewhere around 50k nodes, because it is quadratic in memory. Past that, everything is sparse message passing and neighbour sampling - and the sampler you pick determines which architectures are even available to you. The other half of the ecosystem is the graph itself, which usually lives in a database rather than a tensor.

| Framework | Layer | What it gives you | License | Reach for it when |
|----|----|----|----|----|
| [PyTorch Geometric](https://github.com/pyg-team/pytorch_geometric) | modelling | Sparse message passing, ~60 prebuilt convolutions, `NeighborLoader` sampling, and heterogeneous-graph support | MIT | Default. Every model in section 9 is a two-line import here, and it scales where the dense version cannot |
| [DGL](https://github.com/dmlc/dgl) | modelling | The same job with a different API, strong multi-GPU and distributed training, and a mature heterogeneous-graph story | Apache 2.0 | Distributed training, or a codebase already built on it |
| [transformers](https://github.com/huggingface/transformers) / graph transformers ([GraphGPS](https://github.com/rampasek/GraphGPS)) | modelling | Attention over small graphs, which sidesteps both oversmoothing and the 1-WL expressiveness ceiling | Apache 2.0 / MIT | Molecules and other small graphs, where quadratic attention is affordable. Graphormer left the main `transformers` library - use the reference implementation or PyG’s `GPSConv` |
| Equivariant models ([MACE](https://github.com/ACEsuit/mace), [NequIP](https://github.com/mir-group/nequip)) | modelling | Rotation and translation symmetry by construction, for graphs with 3-D coordinates | MIT | Molecular dynamics and materials. Worth more than any amount of extra data or depth |
| [NetworkX](https://github.com/networkx/networkx) / [igraph](https://github.com/igraph/igraph) / [cuGraph](https://github.com/rapidsai/cugraph) | data | Graph construction, classical algorithms, connected components and centrality - and cuGraph does them on a GPU | BSD-3 / GPL-2.0 / Apache 2.0 | Building and inspecting the graph. Homophily, degree distribution and component structure decide your architecture |
| [Neo4j](https://github.com/neo4j/neo4j) + GDS / [Memgraph](https://github.com/memgraph/memgraph) | data | The graph as a queryable database with persistence and incremental updates, plus in-database algorithms | GPL-3.0 (community) / BSL | Production. Real graphs change continuously, and rebuilding a tensor per batch is not a design |
| Samplers: `NeighborLoader`, Cluster-GCN, [GraphSAINT](https://github.com/GraphSAINT/GraphSAINT) | data | k-hop neighbour sampling, partitioning and subgraph sampling - how billion-edge graphs actually get trained | MIT | Anything past a laptop-sized graph. Pick the sampler first; it constrains the architecture |
| [ONNX Runtime](https://onnxruntime.ai/) / TorchScript | inference runtime | The trained GNN exported for inference, usually as a scoring step inside a larger service | MIT | Deployment. Most production GNNs precompute embeddings in batch and serve those, rather than running message passing online |
| [OGB](https://github.com/snap-stanford/ogb) + [torchmetrics](https://github.com/Lightning-AI/torchmetrics) | evaluation | Fixed splits at realistic scale, with a leaderboard that has not been tuned into meaninglessness | MIT / Apache 2.0 | Always. Anything concluded from the 2,708 nodes in this notebook is intuition, not evidence |

**The 2026 default stack** is NetworkX to build and inspect, PyTorch Geometric to model with `NeighborLoader` sampling, OGB to evaluate, and precomputed embeddings served from a normal database. Neo4j when the graph is a live object rather than a dataset.

The common wrong turn is choosing an architecture before measuring homophily. On a low-homophily graph the standard GNNs underperform a plain MLP on the node features alone, and the fixes are specific - separating self and neighbour representations, higher-order neighbourhoods, signed aggregation - rather than “a deeper model”. The second is depth: section 11 shows why every deployed GNN is two or three layers, and adding a fourth is the most reliable way to make results worse.

------------------------------------------------------------------------

## 14. Going Further

**Sample, do not shrink.** At production scale the model is rarely the bottleneck. `NeighborLoader` (GraphSAGE-style k-hop sampling), Cluster-GCN partitioning, and GraphSAINT subgraph sampling are how billion-edge graphs get trained. Pick the sampler first; it determines what architectures are even available to you.

**Measure homophily before choosing an architecture.** If it is low, the standard GNNs will underperform a plain MLP. The fixes are specific - separating self and neighbour representations (which GraphSAGE already does), higher-order neighbourhoods, signed aggregation - and the [heterophilous graph benchmark](https://github.com/yandex-research/heterophilous-graphs) is the place to validate them.

**Benchmark on OGB, not Cora.** Fixed splits, realistic scale, and a leaderboard that has not been tuned into meaninglessness. Anything you conclude from the 2,708 nodes in this notebook should be treated as intuition, not evidence.

**Try a graph transformer on small graphs.** For molecules (tens of atoms), Graphormer and GraphGPS sidestep both oversmoothing and the 1-WL expressiveness ceiling, at quadratic cost that is affordable at that size. `transformers` shipped a Graphormer implementation in the 4.x line; it was moved out of the main library, so use the reference implementation or PyG’s `GPSConv` today.

**For anything with 3D coordinates, use an equivariant model.** MACE, NequIP and Allegro respect rotation and translation symmetry by construction, which is worth more than any amount of extra data on molecular and materials tasks. This is where graph learning has had its clearest scientific wins - GNoME’s 2.2 million predicted stable crystals, and the interatomic potentials now running molecular dynamics at near-DFT accuracy.

**Keep the boring baseline.** Neighbour-aggregated features (mean of neighbour features, degree, PageRank, triangle count, component size) in a table, fed to LightGBM, is fast, inductive, explainable, and beats a badly-tuned GNN more often than the literature admits. Make the GNN earn its place against it.

**Related notebooks in this repo:** `Tabular/00_Tabular_Classification` (the same labels when rows are independent), `Natural_Language_Processing/07_Feature_Extraction` (embeddings as the input to a graph model), and `Reinforcement_Learning/00_Reinforcement_Learning` (graphs as the state space in combinatorial problems).

------------------------------------------------------------------------

Back to top
