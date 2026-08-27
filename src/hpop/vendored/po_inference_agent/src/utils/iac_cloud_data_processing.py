"""
IAC Cloud Data Processing Module

This module processes IAC cloud event data to extract linear extensions
based on session observations. Each session represents a sequence of 
service events that can be used for preference learning.

Author: Generated for HPO Inference Agent
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Tuple, Optional
from collections import defaultdict


def load_iac_cloud_data(file_path: str) -> pd.DataFrame:
    """
    Load the IAC cloud data from Excel file.
    
    Args:
        file_path: Path to the iac_cloud.csv (actually xlsx format) file
        
    Returns:
        DataFrame with the loaded data
    """
    df = pd.read_excel(file_path, engine='openpyxl')
    return df


def create_combined_node(df: pd.DataFrame, 
                          separator: str = "_") -> pd.DataFrame:
    """
    Combine event_name and service_name to create a new node identifier.
    
    Args:
        df: Input DataFrame with event_name and service_name columns
        separator: String to use between event_name and service_name
        
    Returns:
        DataFrame with new 'node' column
    """
    df = df.copy()
    df['node'] = df['event_name'] + separator + df['service_name']
    return df


def extract_linear_extensions_by_session(
    df: pd.DataFrame,
    session_col: str = 'session_id',
    node_col: str = 'node',
    time_col: str = 'gmt_created'
) -> Dict[str, List[str]]:
    """
    Extract linear extensions (ordered sequences) for each session.
    
    Each session represents one observation of a linear extension,
    where the order is determined by the timestamp of events.
    
    Args:
        df: DataFrame with node and session information
        session_col: Column name for session identifier
        node_col: Column name for the node (combined event+service)
        time_col: Column name for timestamp ordering
        
    Returns:
        Dictionary mapping session_id to list of nodes in order
    """
    linear_extensions = {}
    
    for session_id, group in df.groupby(session_col):
        # Sort by timestamp to get the order
        sorted_group = group.sort_values(by=time_col)
        # Extract the sequence of nodes
        sequence = sorted_group[node_col].tolist()
        linear_extensions[session_id] = sequence
    
    return linear_extensions


def get_unique_nodes(linear_extensions: Dict[str, List[str]]) -> List[str]:
    """
    Get all unique nodes across all linear extensions.
    
    Args:
        linear_extensions: Dictionary of session_id to node sequences
        
    Returns:
        Sorted list of unique nodes
    """
    all_nodes = set()
    for sequence in linear_extensions.values():
        all_nodes.update(sequence)
    return sorted(list(all_nodes))


def convert_to_ranking_indices(
    linear_extensions: Dict[str, List[str]],
    node_to_idx: Optional[Dict[str, int]] = None
) -> Tuple[List[List[int]], Dict[str, int], Dict[int, str]]:
    """
    Convert linear extensions from node names to integer indices.
    
    Args:
        linear_extensions: Dictionary of session_id to node sequences
        node_to_idx: Optional pre-defined mapping from nodes to indices
        
    Returns:
        Tuple of:
        - List of rankings as integer sequences
        - Dictionary mapping node names to indices
        - Dictionary mapping indices to node names
    """
    if node_to_idx is None:
        unique_nodes = get_unique_nodes(linear_extensions)
        node_to_idx = {node: idx for idx, node in enumerate(unique_nodes)}
    
    idx_to_node = {idx: node for node, idx in node_to_idx.items()}
    
    rankings = []
    for session_id, sequence in linear_extensions.items():
        ranking = [node_to_idx[node] for node in sequence if node in node_to_idx]
        rankings.append(ranking)
    
    return rankings, node_to_idx, idx_to_node


def remove_consecutive_duplicates(sequence: List[str]) -> List[str]:
    """
    Remove consecutive duplicate nodes from a sequence.
    
    Some events may be logged multiple times consecutively,
    this function keeps only the first occurrence in a run.
    
    Args:
        sequence: List of node names
        
    Returns:
        List with consecutive duplicates removed
    """
    if not sequence:
        return []
    
    result = [sequence[0]]
    for node in sequence[1:]:
        if node != result[-1]:
            result.append(node)
    return result


def extract_linear_extensions_deduplicated(
    df: pd.DataFrame,
    session_col: str = 'session_id',
    node_col: str = 'node',
    time_col: str = 'gmt_created',
    remove_all_duplicates: bool = False
) -> Dict[str, List[str]]:
    """
    Extract linear extensions with deduplication options.
    
    Args:
        df: DataFrame with node and session information
        session_col: Column name for session identifier
        node_col: Column name for the node
        time_col: Column name for timestamp ordering
        remove_all_duplicates: If True, keep only unique nodes per session
                               If False, only remove consecutive duplicates
        
    Returns:
        Dictionary mapping session_id to deduplicated node sequences
    """
    linear_extensions = extract_linear_extensions_by_session(
        df, session_col, node_col, time_col
    )
    
    deduplicated = {}
    for session_id, sequence in linear_extensions.items():
        if remove_all_duplicates:
            # Keep only first occurrence of each node
            seen = set()
            unique_seq = []
            for node in sequence:
                if node not in seen:
                    seen.add(node)
                    unique_seq.append(node)
            deduplicated[session_id] = unique_seq
        else:
            # Only remove consecutive duplicates
            deduplicated[session_id] = remove_consecutive_duplicates(sequence)
    
    return deduplicated


def get_session_statistics(linear_extensions: Dict[str, List[str]]) -> pd.DataFrame:
    """
    Compute statistics about the linear extensions.
    
    Args:
        linear_extensions: Dictionary of session_id to node sequences
        
    Returns:
        DataFrame with session statistics
    """
    stats = []
    for session_id, sequence in linear_extensions.items():
        stats.append({
            'session_id': session_id,
            'sequence_length': len(sequence),
            'unique_nodes': len(set(sequence)),
            'has_duplicates': len(sequence) != len(set(sequence))
        })
    
    return pd.DataFrame(stats)


def process_iac_cloud_data(
    file_path: str,
    separator: str = "_",
    deduplicate: bool = True,
    remove_all_duplicates: bool = True,
    use_event_name_only: bool = False
) -> Tuple[List[List[int]], Dict[str, int], Dict[int, str], pd.DataFrame]:
    """
    Main function to process IAC cloud data end-to-end.
    
    Args:
        file_path: Path to the data file
        separator: Separator for combining event and service names
        deduplicate: Whether to apply deduplication
        remove_all_duplicates: If True, remove all duplicates; 
                               if False, only consecutive ones
        use_event_name_only: If True, use only event_name as node identifier;
                             if False, combine event_name and service_name
        
    Returns:
        Tuple of:
        - List of rankings as integer sequences
        - Dictionary mapping node names to indices
        - Dictionary mapping indices to node names
        - Original DataFrame with 'node' column added
    """
    # Load data
    df = load_iac_cloud_data(file_path)
    
    # Create node column
    if use_event_name_only:
        df = df.copy()
        df['node'] = df['event_name']
    else:
        df = create_combined_node(df, separator)
    
    # Extract linear extensions
    if deduplicate:
        linear_extensions = extract_linear_extensions_deduplicated(
            df, 
            remove_all_duplicates=remove_all_duplicates
        )
    else:
        linear_extensions = extract_linear_extensions_by_session(df)
    
    # Convert to indices
    rankings, node_to_idx, idx_to_node = convert_to_ranking_indices(linear_extensions)
    
    return rankings, node_to_idx, idx_to_node, df


def print_summary(
    rankings: List[List[int]], 
    node_to_idx: Dict[str, int],
    idx_to_node: Dict[int, str]
):
    """
    Print a summary of the processed data.
    """
    print("=" * 60)
    print("IAC Cloud Data Processing Summary")
    print("=" * 60)
    print(f"\nTotal number of observations (sessions): {len(rankings)}")
    print(f"Total number of unique nodes: {len(node_to_idx)}")
    
    lengths = [len(r) for r in rankings]
    print(f"\nSequence length statistics:")
    print(f"  Min length: {min(lengths)}")
    print(f"  Max length: {max(lengths)}")
    print(f"  Mean length: {np.mean(lengths):.2f}")
    print(f"  Median length: {np.median(lengths):.2f}")
    
    print(f"\nNode mapping (first 20):")
    for node, idx in list(node_to_idx.items())[:20]:
        print(f"  {idx}: {node}")
    
    if len(node_to_idx) > 20:
        print(f"  ... and {len(node_to_idx) - 20} more nodes")
    
    print(f"\nSample linear extensions (first 5):")
    for i, ranking in enumerate(rankings[:5]):
        nodes = [idx_to_node[idx] for idx in ranking]
        print(f"\n  Session {i+1} ({len(ranking)} nodes):")
        for j, node in enumerate(nodes[:10]):
            print(f"    {j+1}. {node}")
        if len(nodes) > 10:
            print(f"    ... and {len(nodes) - 10} more nodes")


if __name__ == "__main__":
    import os
    
    # Get the path to the data file
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(os.path.dirname(script_dir))
    data_path = os.path.join(project_root, "data", "iac_cloud.csv")
    
    print(f"Processing data from: {data_path}")
    print()
    
    # Process the data
    rankings, node_to_idx, idx_to_node, df = process_iac_cloud_data(
        data_path,
        separator="_",
        deduplicate=True,
        remove_all_duplicates=True  # Keep only unique nodes per session
    )
    
    # Print summary
    print_summary(rankings, node_to_idx, idx_to_node)
    
    # Also show some raw linear extensions
    print("\n" + "=" * 60)
    print("Raw Linear Extensions (as node names)")
    print("=" * 60)
    
    linear_extensions = extract_linear_extensions_deduplicated(
        df,
        remove_all_duplicates=True
    )
    
    for i, (session_id, sequence) in enumerate(list(linear_extensions.items())[:3]):
        print(f"\nSession: {session_id[:30]}...")
        print(f"Sequence ({len(sequence)} nodes): {sequence}")

