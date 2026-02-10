# 🚦 Traffic Risk Prediction System

A **Spatio-Temporal Graph Neural Network (ST-GNN)** for predicting intersection crash risk in Dallas, TX. This project is part of an Intelligent Transportation Systems (ITS) Master's thesis.

---

## 📋 Project Overview

This system uses deep learning to predict traffic accident risk by analyzing:

- **Spatial patterns**: Geographic relationships between intersections
- **Temporal patterns**: Time-based accident trends
- **Environmental factors**: Weather, visibility, and lighting conditions
- **Behavioral indicators**: Driver distraction annotations via NLP

---

## 🏗️ Project Structure

```
traffic-risk-system/
├── data/
│   ├── raw/                        # Original accident data (not tracked)
│   ├── processed/                  # Pipeline outputs (not tracked)
│   │   ├── dallas_crashes_annotated.csv
│   │   ├── crashes_mapped.csv
│   │   ├── dallas_drive_net.graphml
│   │   ├── node_mapping.pkl
│   │   ├── processed_graph_data.pt
│   │   ├── temporal_signal.pt
│   │   ├── train_dataset.pt
│   │   └── test_dataset.pt
│   └── external/                   # External data sources
├── src/
│   ├── __init__.py
│   ├── data_loader.py              # Phase 1: Data engineering & NLP annotation
│   ├── graph_builder.py            # Phase 2: Road network graph construction
│   ├── crash_mapper.py             # Phase 2: Crash-to-node spatial snapping
│   ├── drisi_calculator.py         # Phase 2: DR-ISI target variable
│   ├── pyg_converter.py            # Phase 2: PyTorch Geometric export
│   ├── build_graph_pipeline.py     # Phase 2: End-to-end orchestrator
│   ├── temporal_processor.py       # Phase 3: Temporal sequence generation
│   ├── model_architecture.py       # Phase 4: HybridSTGNN model definition
│   ├── train_model.py              # Phase 4: Training loop & evaluation
│   └── evaluate_model.py           # Phase 4: Standalone model evaluation
├── models/
│   ├── best_stgnn_model.pth        # Best model weights (not tracked)
│   └── checkpoint.pt               # Full training checkpoint for resume (not tracked)
├── notebooks/                      # Jupyter notebooks for exploration
├── environment.yml                 # Conda environment specification
├── requirements.txt                # Pip dependencies
└── README.md
```

---

## 🚀 Getting Started

### Prerequisites

- Python 3.11+
- CUDA-capable GPU (recommended: RTX 3080 or better)
- ~4GB disk space for dependencies
- ~500MB for processed Dallas data

### Installation

1. **Clone the repository**

   ```bash
   git clone https://github.com/yourusername/traffic-risk-system.git
   cd traffic-risk-system
   ```

2. **Create conda environment**

   ```bash
   conda env create -f environment.yml
   conda activate traffic-risk
   ```

3. **Verify GPU support**

   ```bash
   python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
   ```

4. **Download the dataset**

   Download [US Accidents (March 2023)](https://www.kaggle.com/datasets/sobhanmoosavi/us-accidents) from Kaggle and place it in:

   ```
   data/raw/US_Accidents_March23.csv
   ```

---

## 📊 Phase 1: Data Engineering & NLP Annotation

### Running the Pipeline

```bash
python src/data_loader.py
```

### What it does:

1. **Chunked Processing**: Reads the ~3GB dataset in 100k-row chunks to manage memory
2. **Geographic Filtering**: Extracts only Dallas, TX records
3. **Data Cleaning**: Drops missing coordinates/timestamps, rounds coordinates
4. **NLP Annotation**: Uses `all-MiniLM-L6-v2` sentence transformer to detect distraction-related crashes via semantic similarity

### Configuration Options

| Argument           | Default                                       | Description                          |
| ------------------ | --------------------------------------------- | ------------------------------------ |
| `--input, -i`      | `data/raw/US_Accidents_March23.csv`           | Input file path                      |
| `--output, -o`     | `data/processed/dallas_crashes_annotated.csv` | Output file path                     |
| `--chunk-size, -c` | `100000`                                      | Rows per processing chunk            |
| `--threshold, -t`  | `0.35`                                        | Similarity threshold for distraction |
| `--no-gpu`         | -                                             | Disable CUDA acceleration            |

### Output Statistics (Dallas, TX)

| Metric                        | Value           |
| ----------------------------- | --------------- |
| Total Records                 | 115,545         |
| Distracted Crashes            | 28,571 (24.73%) |
| Avg Severity (Distracted)     | 2.530           |
| Avg Severity (Non-Distracted) | 2.218           |

---

## 🗺️ Phase 2: Spatial Graph Construction & DR-ISI

### Running the Pipeline

```bash
python src/build_graph_pipeline.py
```

### What it does:

1. **Graph Construction** (`graph_builder.py`): Downloads the Dallas driving network from OpenStreetMap via OSMnx, projects to UTM Zone 14N (EPSG:32614), consolidates complex intersections (15m tolerance), and extracts the Largest Strongly Connected Component (LSCC)
2. **Feature Engineering**: Computes 5 node features (degree, street_count, betweenness centrality, bearing entropy, avg incident speed) and 4 edge features (length, bearing, speed, highway type)
3. **Crash Snapping** (`crash_mapper.py`): Vectorized projection of 115k crash coordinates to UTM and snapping to nearest road edges via `ox.nearest_edges`
4. **DR-ISI Target** (`drisi_calculator.py`): Computes the Distraction-Related Intersection Severity Index using EPDO weights (Fatal=12, Injury=3, PDO=1) with log-normalisation: `DR_ISI = log(WSS + 1)`
5. **PyG Export** (`pyg_converter.py`): Sanitises non-numeric graph attributes and constructs a `torch_geometric.data.Data` object

### Pipeline Results

| Metric                     | Value                           |
| -------------------------- | ------------------------------- |
| Raw graph                  | 36,474 nodes → 92,964 edges     |
| After consolidation + LSCC | **24,697 nodes → 71,392 edges** |
| LSCC retention             | 99.0%                           |
| Median snap distance       | **1.7 m**                       |
| Nodes with crashes         | 11,557 / 24,697 (46.8%)         |
| Nodes with DR-ISI > 0      | 3,184 (12.9%)                   |
| DR-ISI range               | [0.693, 6.943]                  |
| Pipeline runtime           | ~7.5 minutes                    |

### Output Tensors (`processed_graph_data.pt`)

| Tensor       | Shape      | Content                                                       |
| ------------ | ---------- | ------------------------------------------------------------- |
| `x`          | (24697, 5) | degree, street_count, betweenness, bearing_entropy, avg_speed |
| `edge_index` | (2, 71392) | COO sparse adjacency                                          |
| `edge_attr`  | (71392, 4) | length, bearing, speed_mph, highway_enc                       |
| `y`          | (24697,)   | DR-ISI target scores                                          |
| `pos`        | (24697, 2) | UTM coordinates                                               |

### Configuration Options

| Argument           | Default                                       | Description                          |
| ------------------ | --------------------------------------------- | ------------------------------------ |
| `--crash-csv`      | `data/processed/dallas_crashes_annotated.csv` | Annotated crash data from Phase 1    |
| `--output-dir`     | `data/processed`                              | Directory for all output files       |
| `--force-download` | -                                             | Re-download OSM graph even if cached |

---

## ⏱️ Phase 3: Spatio-Temporal Sequence Generation

### Running the Pipeline

```bash
python src/temporal_processor.py
```

### What it does:

1. **Crash Snapping** (`crash_mapper`): Re-snaps crashes to graph nodes and saves `crashes_mapped.csv` (cached for subsequent runs)
2. **Temporal Aggregation**: Bins 115k crashes into hourly intervals across 24,697 nodes using scipy sparse matrices (0.0071% density)
3. **Dynamic Features**: Generates 5 per-node features per timestep — crash_count + cyclical encodings (hour_sin, hour_cos, dow_sin, dow_cos)
4. **Sliding Windows**: Creates 24-hour look-back windows with 1-hour prediction horizon
5. **Lazy Loading**: `TemporalCrashDataset` materialises each `(24, N, 5)` window on-the-fly from sparse storage (~2 MB vs ~24 GB dense)

### Pipeline Results

| Metric                | Value                   |
| --------------------- | ----------------------- |
| Date range            | 2016-01-01 → 2022-12-31 |
| Total hours           | 61,368                  |
| Sparse matrix density | **0.0071%**             |
| Train samples         | **49,074** (80%)        |
| Test samples          | **12,269** (20%)        |

### Output Tensor Shapes (per sample)

| Tensor | Shape          | Description                           |
| ------ | -------------- | ------------------------------------- |
| `X`    | (24, 24697, 5) | 24h window × nodes × dynamic features |
| `Y`    | (24697,)       | log(1 + WSS) target per node          |

### Usage in Model Training

```python
from src.temporal_processor import load_dataset

train_ds, train_meta = load_dataset(split="train")
test_ds, test_meta = load_dataset(split="test")

X, Y = train_ds[0]  # Lazy: materialised from sparse on-the-fly
edge_index = train_meta["edge_index"]  # (2, 71392) — static graph
static_x = train_meta["static_x"]      # (24697, 5) — static node features
```

### Configuration Options

| Argument        | Default                                       | Description                          |
| --------------- | --------------------------------------------- | ------------------------------------ |
| `--crash-csv`   | `data/processed/dallas_crashes_annotated.csv` | Phase 1 output                       |
| `--data-dir`    | `data/processed`                              | Directory for all outputs            |
| `--train-ratio` | `0.8`                                         | Chronological train/test split ratio |

---

## 🧠 Phase 4: Model Architecture & Training

### Architecture: HybridSTGNN

A 3-stage spatio-temporal model with **157,697 parameters**:

| Stage           | Component  | Details                               |
| --------------- | ---------- | ------------------------------------- |
| Spatial Encoder | 2× GCNConv | (5 → 128 → 128) + ReLU + Dropout(0.1) |
| Temporal Core   | LSTM       | (128 → 128), 1 layer, batch_first     |
| Risk Decoder    | MLP        | (128 → 64 → 1) + ReLU                 |

**Data flow**: `(B, 24, N, 5)` → GCN per timestep → `(B, N, 24, 128)` → LSTM per node → `(B, N, 128)` → MLP → `(B, N)` risk scores

### Training the Model

```bash
# Full training (50 epochs, ~15-25 hours with early stopping)
python src/train_model.py

# Train for only 5 epochs
python src/train_model.py --epochs 5

# Resume training from where you left off (for another 10 epochs total)
python src/train_model.py --resume --epochs 10

# Quick smoke test (2 epochs, large stride)
python src/train_model.py --epochs 2 --stride 2000
```

> **💡 Resume workflow**: Train for 5 epochs → stop → resume later for 10 more.
> The checkpoint (`models/checkpoint.pt`) saves model weights, optimizer, scheduler,
> epoch counter, and best validation loss — so training continues seamlessly.

### Training Pipeline

- **Loss**: Weighted MSE — 10× weight on non-zero targets (handles 87% zero-target sparsity)
- **Optimizer**: Adam (lr=0.001)
- **Scheduler**: ReduceLROnPlateau (patience=3, factor=0.5)
- **Early stopping**: patience=5 epochs on validation loss
- **Gradient clipping**: max_norm=1.0 (prevents LSTM exploding gradients)
- **Gradient accumulation**: batch_size=2 × 4 accum steps = **effective batch of 8**
- **Data stride**: Every 6th sample → ~6,900 train steps/epoch

### VRAM Optimisation

The 24,697-node graph requires careful memory management:

| Optimisation | Approach                                                   |
| ------------ | ---------------------------------------------------------- |
| GCN          | Processes one sample at a time (avoids B×N node blow-up)   |
| LSTM         | Chunks nodes in groups of 4,096 (avoids 197k-sequence OOM) |
| Batch size   | B=2 micro-batch × 4 accumulation = effective B=8           |
| Peak VRAM    | **~10 GB** (fits 16 GB GPU)                                |

### Data Splits

| Split | Samples                        | Source        |
| ----- | ------------------------------ | ------------- |
| Train | ~41,700 (85% of Phase 3 train) | Chronological |
| Val   | ~7,360 (15% of Phase 3 train)  | Chronological |
| Test  | ~12,270 (Phase 3 test)         | Chronological |

### Configuration Options

| Argument        | Default                       | Description                                             |
| --------------- | ----------------------------- | ------------------------------------------------------- |
| `--batch-size`  | `2`                           | Micro-batch size per GPU step                           |
| `--accum-steps` | `4`                           | Gradient accumulation (effective batch = batch × accum) |
| `--stride`      | `6`                           | Sample every N-th window                                |
| `--epochs`      | `50`                          | Maximum training epochs                                 |
| `--lr`          | `0.001`                       | Learning rate                                           |
| `--hidden-dim`  | `128`                         | GCN and LSTM hidden dimension                           |
| `--pos-weight`  | `10.0`                        | Weight multiplier for non-zero targets                  |
| `--save-path`   | `models/best_stgnn_model.pth` | Model checkpoint path                                   |
| `--resume`      | `false`                       | Resume from `models/checkpoint.pt`                      |

### Evaluating a Trained Model

```bash
# Evaluate on the test set (after training is complete)
python src/evaluate_model.py

# Full evaluation without stride (slower but exact)
python src/evaluate_model.py --stride 1

# Evaluate on the training set
python src/evaluate_model.py --split train

# Use a specific model checkpoint
python src/evaluate_model.py --model-path models/best_stgnn_model.pth
```

Reports: Overall RMSE/MAE, Non-zero target RMSE/MAE, Zero-target accuracy, Prediction statistics.

---

## 🔧 Technical Details

### NLP Distraction Detection (Phase 1)

Instead of regex-based keyword matching, we use **semantic similarity** with anchor phrases:

```python
ANCHOR_PHRASES = [
    "driver texting while driving",
    "using cell phone",
    "driver inattentive",
    "distracted by mobile device",
    ...
]
```

The model encodes accident descriptions and computes cosine similarity against these anchors. If `max(similarity) >= 0.35`, the crash is labeled as distraction-related.

### Graph Construction (Phase 2)

- **Projection**: EPSG:32614 (UTM Zone 14N) for metric distance accuracy
- **Consolidation**: 15m tolerance merges dual-carriageway junctions
- **Betweenness centrality**: Sampled with k=1000 (full computation if <10k nodes)
- **Speed imputation**: Missing `maxspeed` tags imputed by highway type (e.g., motorway=65mph, residential=25mph)
- **Bearing entropy**: Shannon entropy of incident-edge bearings in 8 compass bins — measures intersection complexity
- **DR-ISI**: EPDO-weighted severity index with log normalisation to compress skewed distribution

### Temporal Processing (Phase 3)

- **Sparse-first strategy**: Full dense tensor (61k × 25k × 5) = ~24 GB; sparse representation = ~2 MB
- **Lazy materialisation**: Windows built on-the-fly in `__getitem__` — peak RAM ~2-3 GB
- **Cyclical encoding**: `sin/cos` for hour-of-day and day-of-week (avoids one-hot explosion)
- **Dynamic target**: `Y = log(1 + Σ(EPDO_weight × crashes))` per node per hour
- **Chronological split**: No shuffle — preserves temporal ordering for valid evaluation

### Model Architecture (Phase 4)

- **Per-sample GCN**: Processes one sample at a time through 2-layer GCNConv to avoid B×N VRAM blow-up
- **Chunked LSTM**: Processes nodes in groups of 4,096 to prevent 197k-sequence OOM
- **Gradient accumulation**: B=2 × 4 steps = effective batch of 8 within 10 GB VRAM
- **Weighted MSE**: 10× weight on non-zero targets handles extreme sparsity (87% zero risk)
- **Shape-annotated code**: Every tensor reshape has a `# → (shape)` comment for debugging

### Memory Management

- Chunked CSV reading (100k rows/chunk)
- Sparse COO/CSR matrices for temporal aggregation
- Lazy window materialisation from scipy sparse
- Per-sample GCN + chunked LSTM for GPU memory efficiency
- Gradient accumulation for larger effective batch sizes
- Explicit garbage collection after each chunk

---

## 📁 Data Dictionary

Output columns in `dallas_crashes_annotated.csv`:

| Column              | Type      | Description                     |
| ------------------- | --------- | ------------------------------- |
| `ID`                | string    | Unique accident identifier      |
| `Severity`          | int (1-4) | Accident severity level         |
| `Start_Time`        | datetime  | Accident timestamp              |
| `Start_Lat`         | float     | Latitude (5 decimal precision)  |
| `Start_Lng`         | float     | Longitude (5 decimal precision) |
| `Description`       | string    | Accident description text       |
| `Weather_Condition` | string    | Weather at time of accident     |
| `Visibility(mi)`    | float     | Visibility in miles             |
| `Traffic_Signal`    | bool      | Near traffic signal             |
| `Junction`          | bool      | At junction/intersection        |
| `Sunrise_Sunset`    | string    | Day/Night indicator             |
| `Is_Distracted`     | int (0/1) | NLP-derived distraction label   |

---

## 📈 Roadmap

- [x] **Phase 1**: Data Engineering & NLP Annotation
- [x] **Phase 2**: Spatial Graph Construction & DR-ISI Target
- [x] **Phase 3**: Spatio-Temporal Sequence Generation
- [x] **Phase 4**: HybridSTGNN Model Architecture & Training
- [ ] **Phase 5**: Full Training & Evaluation
- [ ] **Phase 6**: Deployment & Visualization

---

## 📚 References

- **Dataset**: [US Accidents (2016-2023)](https://www.kaggle.com/datasets/sobhanmoosavi/us-accidents)
- **NLP Model**: [all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2)
- **Road Network**: [OpenStreetMap](https://www.openstreetmap.org/) via [OSMnx](https://osmnx.readthedocs.io/)
- **Graph ML**: [PyTorch Geometric](https://pyg.org/)

---

## 📄 License

This project is for academic research purposes as part of a Master's thesis in Intelligent Transportation Systems.

---

## 👥 Authors

- Ahmed - Lead Developer & Researcher
