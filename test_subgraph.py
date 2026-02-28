"""Quick test of subgraph extraction."""
import sys
sys.path.insert(0, "src")
import torch
from torch_geometric.utils import k_hop_subgraph
from temporal_processor import load_dataset

# Load graph
dataset, meta = load_dataset(split="test")
edge_index = meta["edge_index"]
num_nodes = dataset.num_nodes

target = 24062  # one of the top risk nodes

print(f"Full graph: {num_nodes} nodes, {edge_index.shape[1]} edges")
print(f"Target node: {target}")

sub_nodes, sub_edge_index, mapping, edge_mask = k_hop_subgraph(
    node_idx=target,
    num_hops=2,
    edge_index=edge_index,
    relabel_nodes=True,
    num_nodes=num_nodes,
)

print(f"sub_nodes shape: {sub_nodes.shape}, dtype: {sub_nodes.dtype}")
print(f"sub_nodes[:10]: {sub_nodes[:10]}")
print(f"sub_edge_index shape: {sub_edge_index.shape}")
print(f"mapping: {mapping}")
print(f"Subgraph: {len(sub_nodes)} nodes, {sub_edge_index.shape[1]} edges")

# Find target in sub_nodes
match = (sub_nodes == target).nonzero(as_tuple=True)[0]
print(f"Target found at local idx: {match}")

# The mapping output IS the local index of the target node(s)
print(f"mapping value (this IS the local index): {mapping.item()}")
