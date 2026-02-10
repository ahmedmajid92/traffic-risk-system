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
│   │   ├── dallas_drive_net.graphml
│   │   ├── node_mapping.pkl
│   │   └── processed_graph_data.pt
│   └── external/                   # External data sources
├── src/
│   ├── __init__.py
│   ├── data_loader.py              # Phase 1: Data engineering & NLP annotation
│   ├── graph_builder.py            # Phase 2: Road network graph construction
│   ├── crash_mapper.py             # Phase 2: Crash-to-node spatial snapping
│   ├── drisi_calculator.py         # Phase 2: DR-ISI target variable
│   ├── pyg_converter.py            # Phase 2: PyTorch Geometric export
│   └── build_graph_pipeline.py     # Phase 2: End-to-end orchestrator
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

### Memory Management

- Chunked CSV reading (100k rows/chunk)
- Early filtering before processing
- Explicit garbage collection after each chunk
- GPU memory monitoring via `torch.cuda.memory_allocated()`

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
- [ ] **Phase 3**: Temporal Feature Engineering
- [ ] **Phase 4**: ST-GNN Model Development
- [ ] **Phase 5**: Training & Evaluation
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
