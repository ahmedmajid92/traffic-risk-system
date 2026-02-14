# Traffic Risk Prediction System — Details

## Completed Phases

### Phase 1: Data Engineering & NLP Annotation

**Files:** `src/data_loader.py`

- Reads the ~3GB US Accidents dataset in chunks, filters to Dallas, TX only.
- Cleans missing data, rounds coordinates.
- Uses a sentence-transformer (`all-MiniLM-L6-v2`) to detect distraction-related crashes via semantic similarity.
- **Output:** `data/processed/dallas_crashes_annotated.csv` (~54k crashes).

### Phase 2: Spatial Graph Construction & DR-ISI

**Files:** `src/graph_builder.py`, `src/crash_mapper.py`, `src/drisi_calculator.py`, `src/pyg_converter.py`, `src/build_graph_pipeline.py`

- Downloads Dallas road network from OpenStreetMap and builds a graph of **24,697 intersections** and **71,392 edges**.
- Computes node features: degree, betweenness centrality, avg speed, bearing entropy, and a binary signal flag.
- Snaps each crash to its nearest graph node using a KD-Tree.
- Computes the DR-ISI target: an EPDO-weighted severity index per node per hour.
- Exports to PyTorch Geometric format.
- **Output:** `data/processed/processed_graph_data.pt`, `data/processed/dallas_drive_net.graphml`.

### Phase 3: Spatio-Temporal Sequence Generation

**Files:** `src/temporal_processor.py`

- Creates hourly sparse matrices (crash counts + weighted severity) across the full time range.
- Adds cyclical time features (sin/cos for hour-of-day and day-of-week).
- Builds sliding windows of 24 hours (W=24) with lazy materialisation to keep RAM at ~2-3 GB.
- Splits data chronologically into train (80%) and test (20%).
- **Output:** `data/processed/temporal_signal.pt`, `data/processed/train_dataset.pt`, `data/processed/test_dataset.pt`.

### Phase 4: Model Architecture & Training

**Files:** `src/model_architecture.py`, `src/train_model.py`, `src/evaluate_model.py`

- **HybridSTGNN** architecture: 2-layer GCN encoder → LSTM temporal core → MLP risk decoder (157,697 parameters).
- Training uses weighted MSE loss (10× for non-zero targets), Adam optimizer, LR scheduling, early stopping, and gradient accumulation (effective batch = 8).
- Supports **resume training** via `--resume` flag (saves full checkpoint after each epoch).
- Standalone evaluation script reports RMSE, MAE, and non-zero target metrics.
- **Output:** `models/best_stgnn_model.pth`, `models/checkpoint.pt`.

---

## Why Training Takes ~5-6 Hours Per Epoch

The main bottleneck is the **graph size**: 24,697 nodes × 71,392 edges.

| Factor                        | Detail                                                                                                                                                                      |
| ----------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **GCN per-sample processing** | To fit in 16 GB VRAM, GCN processes one sample at a time (not batched), so each step runs a 2-layer GCN on 24,697 nodes × 24 timesteps = **592,728 GCN passes per sample**. |
| **Training steps per epoch**  | ~3,476 steps/epoch (even after stride=6 sub-sampling from ~41k windows).                                                                                                    |
| **LSTM on all nodes**         | Each step also runs LSTM on 24,697 nodes × 24 timesteps, chunked in groups of 4,096 to avoid OOM.                                                                           |
| **Gradient accumulation**     | 4 micro-batches are accumulated before each optimizer step, adding overhead.                                                                                                |
| **Backward pass**             | Backpropagation through GCN + LSTM doubles the computation time vs forward-only.                                                                                            |

**In short:** The combination of a very large graph (24k nodes), long temporal windows (24 hours), and VRAM constraints (forcing sequential instead of parallel processing) makes each step slow (~5-8 seconds), and there are ~3,476 steps per epoch.
