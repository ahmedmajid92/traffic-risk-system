"""
Phase 1: Data Engineering & NLP Annotation Pipeline
====================================================

This module provides a memory-efficient data processing pipeline for the
US Accidents dataset, filtering for Dallas-TX accidents and annotating
distraction-related crashes using semantic similarity via sentence-transformers.

Author: Traffic Risk System - ST-GNN Project
Date: 2026-02-09
"""

from __future__ import annotations

import gc
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ============================================================================
# Configuration
# ============================================================================

@dataclass
class PipelineConfig:
    """Configuration for the data processing pipeline."""
    
    # File paths
    input_path: Path = field(default_factory=lambda: Path("data/raw/US_Accidents_March23.csv"))
    output_path: Path = field(default_factory=lambda: Path("data/processed/dallas_crashes_annotated.csv"))
    
    # Processing parameters
    chunk_size: int = 100_000
    similarity_threshold: float = 0.35
    coordinate_precision: int = 5
    
    # Filter criteria
    target_city: str = "Dallas"
    target_state: str = "TX"
    
    # Columns to retain
    columns_to_keep: list[str] = field(default_factory=lambda: [
        "ID", "Severity", "Start_Time", "Start_Lat", "Start_Lng",
        "Description", "Weather_Condition", "Visibility(mi)",
        "Traffic_Signal", "Junction", "Sunrise_Sunset"
    ])
    
    # Model configuration
    model_name: str = "all-MiniLM-L6-v2"
    use_gpu: bool = True
    
    # Anchor phrases for distraction detection
    anchor_phrases: list[str] = field(default_factory=lambda: [
        "driver texting while driving",
        "using cell phone",
        "driver inattentive",
        "looking down at phone",
        "distracted by mobile device",
        "driver not paying attention",
        "driver on phone",
        "texting and driving",
        "cell phone distraction",
        "driver distracted",
        "not focused on road",
        "eyes off the road"
    ])


# ============================================================================
# Semantic Distraction Annotator
# ============================================================================

class SemanticDistractor:
    """
    Annotates accident descriptions with distraction labels using semantic similarity.
    
    Uses sentence-transformers to encode text and compute cosine similarity
    between accident descriptions and predefined distraction anchor phrases.
    
    Attributes:
        model: SentenceTransformer model for text encoding
        anchor_embeddings: Pre-computed embeddings of anchor phrases
        threshold: Similarity threshold for distraction classification
        device: Computation device (cuda/cpu)
    """
    
    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        anchor_phrases: list[str] | None = None,
        threshold: float = 0.35,
        use_gpu: bool = True
    ) -> None:
        """
        Initialize the semantic distractor.
        
        Args:
            model_name: Name of the sentence-transformer model
            anchor_phrases: List of phrases describing distraction
            threshold: Cosine similarity threshold for classification
            use_gpu: Whether to use GPU acceleration if available
        """
        self.threshold = threshold
        
        # Setup device
        if use_gpu and torch.cuda.is_available():
            self.device = "cuda"
            logger.info(f"Using GPU: {torch.cuda.get_device_name(0)}")
        else:
            self.device = "cpu"
            logger.info("Using CPU for inference")
        
        # Load model
        logger.info(f"Loading model: {model_name}")
        self.model = SentenceTransformer(model_name, device=self.device)
        
        # Default anchor phrases
        if anchor_phrases is None:
            anchor_phrases = [
                "driver texting while driving",
                "using cell phone",
                "driver inattentive",
                "looking down at phone",
                "distracted by mobile device",
                "driver not paying attention",
                "driver on phone"
            ]
        
        # Pre-compute anchor embeddings
        logger.info("Encoding anchor phrases...")
        self.anchor_embeddings = self.model.encode(
            anchor_phrases,
            convert_to_tensor=True,
            device=self.device,
            show_progress_bar=False
        )
        logger.info(f"Anchor embeddings shape: {self.anchor_embeddings.shape}")
    
    def annotate(self, descriptions: pd.Series) -> pd.Series:
        """
        Annotate descriptions with distraction labels.
        
        Args:
            descriptions: Series of accident description texts
            
        Returns:
            Series of binary labels (1=distracted, 0=not distracted)
        """
        # Handle missing descriptions
        descriptions = descriptions.fillna("")
        descriptions_list = descriptions.tolist()
        
        if not descriptions_list:
            return pd.Series([], dtype=int)
        
        # Encode all descriptions in batch
        desc_embeddings = self.model.encode(
            descriptions_list,
            convert_to_tensor=True,
            device=self.device,
            show_progress_bar=False,
            batch_size=256
        )
        
        # Compute cosine similarities
        # Shape: (num_descriptions, num_anchors)
        similarities = torch.nn.functional.cosine_similarity(
            desc_embeddings.unsqueeze(1),
            self.anchor_embeddings.unsqueeze(0),
            dim=2
        )
        
        # Get max similarity per description
        max_similarities = similarities.max(dim=1).values
        
        # Apply threshold
        labels = (max_similarities >= self.threshold).int()
        
        return pd.Series(labels.cpu().numpy(), index=descriptions.index)
    
    def get_similarity_scores(self, descriptions: pd.Series) -> pd.Series:
        """
        Get the maximum similarity scores for each description.
        
        Args:
            descriptions: Series of accident description texts
            
        Returns:
            Series of maximum similarity scores
        """
        descriptions = descriptions.fillna("")
        descriptions_list = descriptions.tolist()
        
        if not descriptions_list:
            return pd.Series([], dtype=float)
        
        desc_embeddings = self.model.encode(
            descriptions_list,
            convert_to_tensor=True,
            device=self.device,
            show_progress_bar=False,
            batch_size=256
        )
        
        similarities = torch.nn.functional.cosine_similarity(
            desc_embeddings.unsqueeze(1),
            self.anchor_embeddings.unsqueeze(0),
            dim=2
        )
        
        max_similarities = similarities.max(dim=1).values
        
        return pd.Series(max_similarities.cpu().numpy(), index=descriptions.index)


# ============================================================================
# Data Processor
# ============================================================================

class DataProcessor:
    """
    Memory-efficient processor for large accident datasets.
    
    Implements chunked reading with early filtering to handle datasets
    that exceed available RAM. Integrates with SemanticDistractor for
    NLP-based annotation.
    
    Attributes:
        config: Pipeline configuration
        distractor: SemanticDistractor instance for annotation
    """
    
    def __init__(self, config: Optional[PipelineConfig] = None) -> None:
        """
        Initialize the data processor.
        
        Args:
            config: Pipeline configuration (uses defaults if not provided)
        """
        self.config = config or PipelineConfig()
        self.distractor: Optional[SemanticDistractor] = None
    
    def _init_distractor(self) -> None:
        """Initialize the semantic distractor (lazy loading)."""
        if self.distractor is None:
            self.distractor = SemanticDistractor(
                model_name=self.config.model_name,
                anchor_phrases=self.config.anchor_phrases,
                threshold=self.config.similarity_threshold,
                use_gpu=self.config.use_gpu
            )
    
    def _filter_dallas(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Filter dataframe for Dallas, TX records only.
        
        Args:
            df: Input dataframe
            
        Returns:
            Filtered dataframe
        """
        mask = (
            (df["City"] == self.config.target_city) &
            (df["State"] == self.config.target_state)
        )
        return df[mask].copy()
    
    def _clean_chunk(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean and preprocess a data chunk.
        
        Operations:
        - Drop rows with missing coordinates or timestamps
        - Convert timestamps to datetime
        - Round coordinates to specified precision
        - Select only required columns
        
        Args:
            df: Input dataframe chunk
            
        Returns:
            Cleaned dataframe
        """
        # Drop rows with missing critical fields
        required_fields = ["Start_Lat", "Start_Lng", "Start_Time"]
        df = df.dropna(subset=required_fields)
        
        if df.empty:
            return df
        
        # Convert timestamp
        df["Start_Time"] = pd.to_datetime(df["Start_Time"], errors="coerce")
        df = df.dropna(subset=["Start_Time"])
        
        if df.empty:
            return df
        
        # Round coordinates
        df["Start_Lat"] = df["Start_Lat"].round(self.config.coordinate_precision)
        df["Start_Lng"] = df["Start_Lng"].round(self.config.coordinate_precision)
        
        # Select columns (keep only those that exist in the data)
        available_columns = [
            col for col in self.config.columns_to_keep 
            if col in df.columns
        ]
        df = df[available_columns]
        
        return df
    
    def _annotate_distractions(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add distraction annotations to the dataframe.
        
        Args:
            df: Input dataframe with Description column
            
        Returns:
            Dataframe with Is_Distracted column
        """
        if "Description" not in df.columns:
            logger.warning("No Description column found, skipping annotation")
            df["Is_Distracted"] = 0
            return df
        
        self._init_distractor()
        df["Is_Distracted"] = self.distractor.annotate(df["Description"])
        
        return df
    
    def process_csv(
        self,
        input_path: Optional[Path] = None,
        output_path: Optional[Path] = None
    ) -> pd.DataFrame:
        """
        Process the entire CSV file using chunked reading.
        
        Args:
            input_path: Path to input CSV (uses config default if not provided)
            output_path: Path to output CSV (uses config default if not provided)
            
        Returns:
            Processed and annotated dataframe
        """
        input_path = input_path or self.config.input_path
        output_path = output_path or self.config.output_path
        
        logger.info(f"Processing: {input_path}")
        logger.info(f"Chunk size: {self.config.chunk_size:,}")
        
        # Ensure output directory exists
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Process in chunks
        processed_chunks: list[pd.DataFrame] = []
        total_rows_read = 0
        total_dallas_rows = 0
        
        # Count total rows for progress bar (read just the index)
        logger.info("Counting total rows...")
        total_rows = sum(1 for _ in open(input_path, 'r', encoding='utf-8')) - 1  # -1 for header
        num_chunks = (total_rows // self.config.chunk_size) + 1
        
        logger.info(f"Total rows in file: {total_rows:,}")
        logger.info(f"Estimated chunks: {num_chunks}")
        
        # Process chunks
        with tqdm(total=num_chunks, desc="Processing chunks") as pbar:
            for chunk in pd.read_csv(
                input_path,
                chunksize=self.config.chunk_size,
                low_memory=False
            ):
                total_rows_read += len(chunk)
                
                # Early filter for Dallas, TX
                filtered = self._filter_dallas(chunk)
                
                if not filtered.empty:
                    # Clean the data
                    cleaned = self._clean_chunk(filtered)
                    
                    if not cleaned.empty:
                        total_dallas_rows += len(cleaned)
                        processed_chunks.append(cleaned)
                
                # Memory management
                del chunk, filtered
                gc.collect()
                
                pbar.update(1)
                pbar.set_postfix({
                    "Dallas rows": total_dallas_rows,
                    "Memory (MB)": f"{torch.cuda.memory_allocated() / 1e6:.1f}" if torch.cuda.is_available() else "N/A"
                })
        
        if not processed_chunks:
            logger.error("No Dallas, TX records found!")
            raise ValueError("No data to process")
        
        # Concatenate all chunks
        logger.info("Concatenating chunks...")
        df = pd.concat(processed_chunks, ignore_index=True)
        logger.info(f"Total Dallas records: {len(df):,}")
        
        # Free memory
        del processed_chunks
        gc.collect()
        
        # Apply NLP annotation
        logger.info("Applying distraction annotation...")
        df = self._annotate_distractions(df)
        
        # Save to CSV
        logger.info(f"Saving to: {output_path}")
        df.to_csv(output_path, index=False)
        logger.info(f"Saved {len(df):,} records")
        
        return df
    
    @staticmethod
    def generate_report(df: pd.DataFrame) -> dict:
        """
        Generate statistical report for the processed data.
        
        Args:
            df: Processed dataframe with Is_Distracted column
            
        Returns:
            Dictionary containing statistics
        """
        total_rows = len(df)
        
        if "Is_Distracted" not in df.columns:
            return {"total_rows": total_rows, "error": "No Is_Distracted column"}
        
        distracted = df[df["Is_Distracted"] == 1]
        non_distracted = df[df["Is_Distracted"] == 0]
        
        distracted_count = len(distracted)
        distracted_pct = (distracted_count / total_rows) * 100 if total_rows > 0 else 0
        
        avg_severity_distracted = distracted["Severity"].mean() if len(distracted) > 0 else 0
        avg_severity_non_distracted = non_distracted["Severity"].mean() if len(non_distracted) > 0 else 0
        
        report = {
            "total_rows": total_rows,
            "distracted_count": distracted_count,
            "non_distracted_count": len(non_distracted),
            "distracted_percentage": round(distracted_pct, 2),
            "avg_severity_distracted": round(avg_severity_distracted, 3),
            "avg_severity_non_distracted": round(avg_severity_non_distracted, 3),
        }
        
        return report
    
    @staticmethod
    def print_report(report: dict) -> None:
        """Print a formatted statistical report."""
        print("\n" + "=" * 60)
        print("STATISTICAL REPORT: Dallas, TX Crash Data")
        print("=" * 60)
        print(f"{'Total Records:':<35} {report['total_rows']:,}")
        print(f"{'Distracted Crashes:':<35} {report['distracted_count']:,}")
        print(f"{'Non-Distracted Crashes:':<35} {report['non_distracted_count']:,}")
        print(f"{'Distraction Percentage:':<35} {report['distracted_percentage']:.2f}%")
        print("-" * 60)
        print(f"{'Avg Severity (Distracted):':<35} {report['avg_severity_distracted']:.3f}")
        print(f"{'Avg Severity (Non-Distracted):':<35} {report['avg_severity_non_distracted']:.3f}")
        print("=" * 60 + "\n")


# ============================================================================
# Main Entry Point
# ============================================================================

def main() -> None:
    """Main entry point for the data processing pipeline."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Process US Accidents data for Dallas, TX with distraction annotation"
    )
    parser.add_argument(
        "--input", "-i",
        type=Path,
        default=Path("data/raw/US_Accidents_March23.csv"),
        help="Input CSV file path"
    )
    parser.add_argument(
        "--output", "-o",
        type=Path,
        default=Path("data/processed/dallas_crashes_annotated.csv"),
        help="Output CSV file path"
    )
    parser.add_argument(
        "--chunk-size", "-c",
        type=int,
        default=100_000,
        help="Chunk size for processing"
    )
    parser.add_argument(
        "--threshold", "-t",
        type=float,
        default=0.35,
        help="Similarity threshold for distraction classification"
    )
    parser.add_argument(
        "--no-gpu",
        action="store_true",
        help="Disable GPU acceleration"
    )
    
    args = parser.parse_args()
    
    # Create configuration
    config = PipelineConfig(
        input_path=args.input,
        output_path=args.output,
        chunk_size=args.chunk_size,
        similarity_threshold=args.threshold,
        use_gpu=not args.no_gpu
    )
    
    # Process data
    processor = DataProcessor(config)
    df = processor.process_csv()
    
    # Generate and print report
    report = processor.generate_report(df)
    processor.print_report(report)
    
    # Sample output
    print("\nSample of processed data:")
    print(df.head(10).to_string())
    
    print("\nSample of distracted crashes:")
    distracted_sample = df[df["Is_Distracted"] == 1].head(5)
    if len(distracted_sample) > 0:
        print(distracted_sample[["ID", "Severity", "Description", "Is_Distracted"]].to_string())
    else:
        print("No distracted crashes found in sample.")


if __name__ == "__main__":
    main()
