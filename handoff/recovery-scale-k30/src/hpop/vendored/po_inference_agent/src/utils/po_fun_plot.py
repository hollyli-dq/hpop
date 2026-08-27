from collections import Counter
import numpy as np
from scipy.stats import multivariate_normal
import networkx as nx
import seaborn as sns
import pandas as pd
from scipy.stats import beta as beta_dist
import itertools
import matplotlib.pyplot as plt
from typing import List, Optional, Dict, Any
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
import math
from scipy.stats import beta, kstest

# Optional import for pygraphviz
try:
    import pygraphviz
    PYGRAPHVIZ_AVAILABLE = True
except ImportError:
    PYGRAPHVIZ_AVAILABLE = False
    print("Warning: pygraphviz not available. Graph visualization features will be limited.")
from scipy.integrate import quad
import matplotlib.pyplot as plt
# import pygraphviz as pgv
from scipy.stats import expon, kstest, probplot
from scipy.special import gammaln, digamma
import os 
from typing import Dict, Any, List, Optional, Union, Tuple
from tabulate import tabulate
import sys
import csv 
# Make sure these paths and imports match your local project structure
sys.path.append('../src/utils')  # Example path
from src.utils.po_fun import BasicUtils, StatisticalUtils, GenerationUtils, ConversionUtils


class PO_plot:
    @staticmethod
    def save_rankings_to_csv(y_a_i_dict, output_file='data/observed_rankings.csv'):
        """
        Save y_a_i_dict to a CSV file
        
        Args:
            y_a_i_dict: Dictionary with assessor rankings
            output_file: Path to output CSV file
        """
        # Create data directory if it doesn't exist
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        
        # Open CSV file for writing
        with open(output_file, 'w', newline='') as csvfile:
            # Create CSV writer
            writer = csv.writer(csvfile)
            
            # Write header
            writer.writerow(['assessor_id', 'task_id', 'ranking'])
            
            # Write data rows
            for assessor, tasks in y_a_i_dict.items():
                # Handle the case where tasks is a list (output from generate_total_orders_for_assessor)
                if isinstance(tasks, list):
                    for task_id, order in enumerate(tasks):
                        # Convert order to string if it's a list or tuple
                        if isinstance(order, (list, tuple)):
                            order_str = ','.join(map(str, order))
                        else:
                            order_str = str(order)
                        
                        writer.writerow([assessor, task_id, order_str])
                # Handle the case where tasks is a dictionary (alternative format)
                elif isinstance(tasks, dict):
                    for task_id, orders in tasks.items():
                        for order in orders:
                            # Convert order to string if it's a list or tuple
                            if isinstance(order, (list, tuple)):
                                order_str = ','.join(map(str, order))
                            else:
                                order_str = str(order)
                        
                            writer.writerow([assessor, task_id, order_str])
        
        print(f"Rankings saved to {output_file}")

    @staticmethod
    def plot_Z_trace(Z_trace, index_to_item):
        """
        Plots the trace of multidimensional latent variables Z over iterations.
        
        Parameters:
        - Z_trace: List of Z matrices over iterations. Each Z is an (n x K) array.
        - index_to_item: Dictionary mapping item indices to item labels.
        """
        Z_array = np.array(Z_trace)  # Shape should be (iterations, n, K)
        iterations = Z_array.shape[0]  # Number of iterations

        # Check dimensions
        if Z_array.ndim != 3:
            raise ValueError("Z_trace should be a list of Z matrices with shape (n, K).")

        _, n_items, K = Z_array.shape  # Extract number of items and latent dimensions

        # Create subplots for each dimension
        fig, axes = plt.subplots(K, 1, figsize=(12, 4 * K), sharex=True)
        if K == 1:
            axes = [axes]  # Ensure axes is iterable when K=1

        for k in range(K):
            ax = axes[k]
            for idx in range(n_items):
                ax.plot(range(iterations), Z_array[:, idx, k], label=f"{index_to_item[idx]}")
            ax.set_ylabel(f'Latent Variable Z (Dimension {k + 1})')
            ax.legend(loc='best', fontsize='small')
            ax.grid(True)
        
        axes[-1].set_xlabel('Iteration')
        plt.suptitle('Trace Plot of Multidimensional Latent Variables Z', fontsize=16)
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.show()

    @staticmethod
    def plot_acceptance_rates(accepted_iterations, acceptance_rates):
        """
        Plots the acceptance rates over iterations.
        
        Parameters:
        - accepted_iterations: List of iteration numbers where acceptance rates were recorded.
        - acceptance_rates: List of acceptance rates corresponding to the iterations.
        """
        plt.figure(figsize=(10, 6))
        plt.plot(accepted_iterations, acceptance_rates, marker='o', linestyle='-', color='blue')
        plt.xlabel('Iteration', fontsize=12)
        plt.ylabel('Acceptance Rate', fontsize=12)
        plt.title('Acceptance Rate Over Time', fontsize=14)
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    @staticmethod
    def plot_top_partial_orders(top_percentages, top_n=4, item_labels=None, max_labels=15):
        """
        Plot the top N partial orders as heatmaps with their corresponding frequencies and percentages.
        
        Parameters:
        -----------
        top_percentages : List[Tuple[np.ndarray, int, float]]
            List of tuples containing (partial_order_matrix, count, percentage).
        top_n : int, optional
            Number of top partial orders to plot (default: 4).
        item_labels : List[str], optional
            List of labels for the items. If None, numerical indices are used.
        max_labels : int, optional
            Maximum number of items to show labels for. If matrix is larger, labels are hidden (default: 15).
        """
        if not top_percentages:
            print("Warning: top_percentages is empty. Nothing to plot.")
            return
        
        # Get matrix size from first partial order
        first_matrix = top_percentages[0][0]
        n_items = first_matrix.shape[0]
        
        # Determine if we should show labels
        show_labels = item_labels is not None and n_items <= max_labels
        display_labels = item_labels if show_labels else [str(i) for i in range(n_items)] if n_items <= max_labels else False
        
        # Calculate font sizes based on matrix size
        annot_fontsize = max(6, min(10, int(120 / max(1, n_items))))
        label_fontsize = max(7, min(10, int(140 / max(1, n_items))))
        
        # Determine the layout of subplots
        n_cols = 2  # Number of columns in the subplot grid
        n_rows = (top_n + n_cols - 1) // n_cols  # Ceiling division for rows
        
        # Adjust figure size based on number of items
        fig_width = max(8, min(12, 5 * n_cols + n_items * 0.3))
        fig_height = max(6, min(10, 4 * n_rows + n_items * 0.2))
        plt.figure(figsize=(fig_width, fig_height))
        
        for idx, (order, count, percentage) in enumerate(top_percentages[:top_n], 1):
            ax = plt.subplot(n_rows, n_cols, idx)
            
            # Get matrix size for this specific matrix (in case sizes differ)
            current_n_items = order.shape[0]
            
            # Determine if we should annotate (show numbers in cells)
            # Only annotate if matrix is small enough
            annot = current_n_items <= 10
            
            # Get labels for this matrix size
            if show_labels and current_n_items == n_items:
                # Use the precomputed labels if sizes match
                current_labels = display_labels
            elif item_labels is not None and current_n_items <= max_labels:
                # Use provided labels if available and matrix is small enough
                current_labels = item_labels[:current_n_items] if len(item_labels) >= current_n_items else [str(i) for i in range(current_n_items)]
            elif current_n_items <= max_labels:
                # Use numeric indices if small enough
                current_labels = [str(i) for i in range(current_n_items)]
            else:
                # Hide labels for large matrices
                current_labels = False
            
            # Calculate font size for this matrix
            current_annot_fontsize = max(6, min(10, int(120 / max(1, current_n_items))))
            
            # Black & white heatmap
            sns.heatmap(
                order, 
                annot=annot,
                fmt="d", 
                cmap="Greys",  # Black & white
                cbar=False, 
                linewidths=0.3, 
                linecolor='gray',
                xticklabels=current_labels,
                yticklabels=current_labels,
                annot_kws={'size': current_annot_fontsize} if annot else {},
                ax=ax
            )
            
            ax.set_title(f"Top {idx}: {percentage:.1f}% (n={count})", fontsize=10, fontweight='bold')
            
            # Rotate x-axis labels for readability if showing labels
            if current_labels:
                current_label_fontsize = max(7, min(10, int(140 / max(1, current_n_items))))
                ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right', fontsize=current_label_fontsize)
                ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=current_label_fontsize)
            
            ax.set_xlabel("")
            ax.set_ylabel("")
        
        # Remove any empty subplots
        total_plots = n_rows * n_cols
        if top_n < total_plots:
            for empty_idx in range(top_n + 1, total_plots + 1):
                plt.subplot(n_rows, n_cols, empty_idx)
                plt.axis('off')
        
        plt.tight_layout()
        plt.show()



    @staticmethod
    def plot_log_likelihood(log_likelihood_data: Union[Dict[str, Any], List[float]], 
                          burn_in: int = 100,
                            title: str = 'Log Likelihood Over MCMC Iterations') -> None:
        """Plot the total log likelihood over MCMC iterations."""
        if isinstance(log_likelihood_data, dict):
            log_likelihood_currents = log_likelihood_data.get('log_likelihood_currents', [])
        else:
            log_likelihood_currents = log_likelihood_data
        
        if len(log_likelihood_currents) > burn_in:
            burned_ll = log_likelihood_currents[burn_in:]
            iterations = np.arange(burn_in + 1, len(log_likelihood_currents) + 1)
            print(f"Excluding {burn_in} burn-in iterations")
        else:
            burned_ll = log_likelihood_currents
            iterations = np.arange(1, len(log_likelihood_currents) + 1)
            burn_in = 0
            print("No burn-in period applied")
        
        sns.set(style="whitegrid")
        plt.figure(figsize=(14, 7))
        
        sns.lineplot(x=iterations, y=burned_ll, label='Current State', color='blue')
        
        plt.title(f'{title} (Post Burn-in)', fontsize=16)
        plt.xlabel('Iteration', fontsize=14)
        plt.ylabel('Total Log Likelihood', fontsize=14)
        
        plt.legend(title='State')
        plt.tight_layout()
        plt.show()
    @staticmethod
    def plot_acceptance_rate(acceptance_rates: List[float], num_iterations: int) -> None:
        """
        Plot the cumulative acceptance rate over MCMC iterations.
        
        Parameters:
        - acceptance_rates (List[float]): Cumulative acceptance rates up to each iteration.
        - num_iterations (int): Total number of iterations.
        
        Returns:
        - None. Displays a matplotlib plot.
        """
        sns.set(style="whitegrid")
        plt.figure(figsize=(14, 7))
        iterations = np.arange(1, num_iterations + 1)
        sns.lineplot(x=iterations, y=acceptance_rates, color='green')
        plt.title('Cumulative Acceptance Rate Over MCMC Iterations', fontsize=16)
        plt.xlabel('Iteration', fontsize=14)
        plt.ylabel('Cumulative Acceptance Rate', fontsize=14)
        plt.tight_layout()
        plt.show()

    @staticmethod
    def plot_list_cluster_diagnostics(
        mcmc_result: Dict[str, Any],
        config: Dict[str, Any],
        burn_in: int = 10_000,
        max_clusters: int = 20,
        figsize: Tuple[int, int] = (12, 4),
        prior_sims: int = 5_000,
        random_seed: int = 42,
    ):
        """
        Diagnostics for list clusters from `c_vec_trace`.

        Parameters
        ----------
        mcmc_result : Dict[str, Any]
            Dictionary returned by the HPO MCMC sampler (must contain 'c_vec_trace').
        config : Dict[str, Any]
            Configuration dictionary containing Pitman–Yor hyperparameters under
            `config['prior']`.
        burn_in : int, optional
            Number of iterations to discard as burn-in (default: 10_000).
        max_clusters : int, optional
            Maximum number of clusters to display in the histogram (default: 20).
        figsize : Tuple[int, int], optional
            Figure size for the diagnostics plots (default: (12, 4)).
        prior_sims : int, optional
            Number of Monte Carlo simulations for the Pitman–Yor prior overlay (default: 5_000).
        random_seed : int, optional
            Random seed for prior simulations (default: 42).

        Returns
        -------
        matplotlib.figure.Figure or None
            The diagnostics figure if plotting succeeds, otherwise ``None``.
        """
        import numpy as np
        import matplotlib.pyplot as plt
        from collections import Counter

        c_vec_trace = mcmc_result.get("c_vec_trace", [])
        if not c_vec_trace:
            print("❗ No 'c_vec_trace' in mcmc_result.")
            return None

        trace = c_vec_trace[burn_in:]
        if not trace:
            print("❗ No draws remain after burn-in.")
            return None

        first = np.asarray(trace[0])
        if first.ndim != 1:
            print("❗ Each element of 'c_vec_trace' must be 1D.")
            return None
        n_lists = int(first.shape[0])

        k_trace = [len(set(np.asarray(z).tolist())) for z in trace]
        k_min, k_max_obs = int(min(k_trace)), int(max(k_trace))
        sample_mean = float(np.mean(k_trace))
        sample_std = float(np.std(k_trace))
        print(f"Draws used (post burn-in): {len(k_trace)}")
        print(f"Observed #clusters range : {k_min} .. {k_max_obs}")

        prior_cfg = (config or {}).get("prior", {})
        d = prior_cfg.get("Dri_alpha", prior_cfg.get("discount_d", 0.0))
        theta = prior_cfg.get("Dri_theta", prior_cfg.get("concentration_theta", None))
        if theta is None:
            theta = 1.0
            print("   Using default theta=1.0")

        fixes = []
        if d < 0:
            fixes.append(f"d={d}→0")
            d = 0.0
        if d >= 1:
            fixes.append(f"d={d}→0.99")
            d = 0.99
        if theta <= -d:
            new_theta = -d + 1e-6
            fixes.append(f"theta={theta}→{new_theta} (must be > -d)")
            theta = new_theta
        if fixes:
            print("⚠️ Adjusted invalid (theta, d): " + ", ".join(fixes))
        print(f"Hyperparameters used: n_lists={n_lists}, theta={theta}, d={d}")

        if d == 0.0:
            prior_mean_analytic = float(theta * (digamma(theta + n_lists) - digamma(theta)))
        else:
            log_ratio = (
                gammaln(theta + d + n_lists)
                - gammaln(theta + d)
                - gammaln(theta + n_lists)
                + gammaln(theta)
            )
            poch_ratio = float(np.exp(np.clip(log_ratio, -700, 700)))
            prior_mean_analytic = float((theta / d) * (poch_ratio - 1.0))

        rng = np.random.default_rng(random_seed)

        def simulate_py_K(n: int, theta_: float, d_: float, sims: int) -> Counter:
            cnt = Counter()
            for _ in range(sims):
                K = 0
                sizes: List[int] = []
                for _t in range(n):
                    if K == 0:
                        K, sizes = 1, [1]
                    else:
                        w_exist = [max(s - d_, 0.0) for s in sizes]
                        w_new = max(theta_ + d_ * K, 0.0)
                        total = w_new + sum(w_exist)
                        if total <= 0:
                            sizes.append(1)
                            K += 1
                        else:
                            u = rng.uniform(0.0, total)
                            acc = 0.0
                            chosen = None
                            for k_idx, w in enumerate(w_exist):
                                acc += w
                                if u <= acc:
                                    chosen = k_idx
                                    break
                            if chosen is None:
                                sizes.append(1)
                                K += 1
                            else:
                                sizes[chosen] += 1
                cnt[K] += 1
            return cnt

        prior_counts = simulate_py_K(n_lists, float(theta), float(d), int(prior_sims))
        xmax = max(max_clusters, k_max_obs)
        ks = np.arange(1, xmax + 1, dtype=int)
        pmf = np.array([prior_counts.get(int(k), 0) for k in ks], dtype=float)
        pmf = pmf / pmf.sum() if pmf.sum() > 0 else np.ones_like(pmf) / len(pmf)
        prior_mean_mc = float(np.sum(ks * pmf))

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
        fig.suptitle("List-Cluster Diagnostics (c_vec_trace)", fontsize=15, fontweight="bold")

        ax1.plot(np.arange(len(k_trace)), k_trace, color="teal", alpha=0.85, lw=0.9)
        ax1.axhline(sample_mean, color="red", ls="--", lw=2, label=f"Mean: {sample_mean:.1f}")
        ax1.set_title("Trace: #Clusters (lists)")
        ax1.set_xlabel("Draw (post burn-in)")
        ax1.set_ylabel("Number of clusters")
        ax1.grid(True, alpha=0.3)
        ax1.legend()

        bins = np.arange(min(k_trace), max(k_trace) + 2)
        ax2.hist(k_trace, bins=bins, color="teal", alpha=0.7, ec="white", lw=0.5)
        scale = len(k_trace)
        ax2.plot(ks, pmf * scale, "k--", lw=2, label=("PY Prior" if d > 0 else "CRP Prior"))
        ax2.axvline(sample_mean, color="red", ls="--", lw=2, label=f"Sample Mean: {sample_mean:.1f}")
        ax2.axvline(
            prior_mean_analytic,
            color="gray",
            ls=":",
            lw=2,
            label=f"Prior Mean (exact): {prior_mean_analytic:.1f}",
        )
        ax2.set_title("Distribution: #Clusters with Prior Overlay")
        ax2.set_xlabel("Number of clusters")
        ax2.set_ylabel("Count")
        ax2.grid(True, alpha=0.3)
        ax2.legend()

        plt.tight_layout()
        plt.show()

        print("\n📈 Summary (list clusters):")
        print(f"   Sample mean     : {sample_mean:.2f}  (std {sample_std:.2f})")
        print(f"   Prior mean (MC) : {prior_mean_mc:.2f}")
        print(f"   Prior mean (exact): {prior_mean_analytic:.2f}  (θ={theta}, d={d})")

        return fig
    @staticmethod
    def visualize_partial_order(
        final_h: np.ndarray,
        Ma_list: list,
        title: str = None,
        ax: Optional[plt.Axes] = None,
        figsize: tuple = None,
        node_scale: float = 1.5,
        edge_weight_matrix: Optional[np.ndarray] = None,
        edge_weight_fmt: str = "{:.2f}",
    ) -> None:
        """
        Visualizes the partial order adjacency matrix as a top-to-bottom hierarchical DAG.
        Uses clean black & white styling with PyGraphviz 'dot' layout.

        final_h : np.ndarray
            N x N adjacency matrix for the partial order
        Ma_list : list
            Labels for the nodes
        title : str, optional
            Title for the plot
        ax : matplotlib.axes.Axes, optional
            If provided, we draw onto this axes. Otherwise, we create a new figure.
        figsize : tuple, optional
            Figure size (width, height). Auto-calculated if None.
        node_scale : float, optional
            Multiplier for node and label sizes.
        """
        if title is None:
            title = "Partial Order Graph"

        n_nodes = len(Ma_list)
        node_scale = max(0.5, float(node_scale))
        
        # Auto-calculate figure size: wider for more nodes, taller for hierarchy
        if figsize is None:
            width = max(18, min(30, n_nodes * 0.8))
            height = max(14, min(24, n_nodes * 0.6))
            figsize = (width, height)

        # Build a directed graph from final_h
        G = nx.DiGraph(final_h)
        edge_labels = {}
        if edge_weight_matrix is not None:
            for u, v in G.edges():
                try:
                    weight_val = float(edge_weight_matrix[u, v])
                except (TypeError, ValueError, IndexError):
                    continue
                edge_labels[(u, v)] = edge_weight_fmt.format(weight_val)

        # Create a label mapping
        labels = {i: str(Ma_list[i]) for i in range(len(Ma_list))}

        # Try PyGraphviz for proper hierarchical 'dot' layout (top-to-bottom)
        try:
            A = nx.nx_agraph.to_agraph(G)
            
            # Set graph attributes for cleaner, WIDER layout
            A.graph_attr['rankdir'] = 'TB'  # Top to Bottom
            A.graph_attr['ranksep'] = '1.2'  # Vertical spacing between ranks
            A.graph_attr['nodesep'] = '1.0'  # Horizontal spacing between nodes
            A.graph_attr['splines'] = 'true'
            A.graph_attr['dpi'] = '300'  # Higher resolution
            A.graph_attr['size'] = f'{figsize[0]},{figsize[1]}!'  # Force size
            A.graph_attr['ratio'] = 'fill'  # Fill the available space
            
            # Calculate font size based on number of nodes
            font_size = max(11, min(22, int((240 / max(1, n_nodes)) * node_scale)))
            node_width = max(1.2, min(3.2, (34.0 / max(1, n_nodes)) * node_scale))
            node_height = max(0.7, min(1.9, (18.0 / max(1, n_nodes)) * node_scale))
            
            # Set node attributes - black & white style
            for node in A.nodes():
                try:
                    node_int = int(node)
                except ValueError:
                    node_int = node
                node_label = labels.get(node_int, str(node))
                node.attr['label'] = node_label
                node.attr['shape'] = 'ellipse'
                node.attr['style'] = 'filled'
                node.attr['fillcolor'] = 'white'
                node.attr['color'] = 'black'
                node.attr['fontsize'] = str(font_size)
                node.attr['fontname'] = 'Helvetica'
                node.attr['width'] = str(node_width)
                node.attr['height'] = str(node_height)
            
            # Set edge attributes - black arrows
            for u, v in G.edges():
                edge = A.get_edge(u, v)
                edge.attr['color'] = 'black'
                edge.attr['arrowsize'] = '0.8'
                if edge_labels:
                    label = edge_labels.get((u, v))
                    if label is not None:
                        edge.attr['label'] = label
                        edge.attr['fontsize'] = str(max(8, font_size - 2))
            
            A.layout('dot')
            
            # Save to temp file and display
            import tempfile
            import os
            temp_png = tempfile.NamedTemporaryFile(suffix='.png', delete=False)
            temp_png.close()
            A.draw(temp_png.name, format='png', prog='dot')
            
            # Create figure and display with larger size
            fig, ax_new = plt.subplots(figsize=figsize)
            img = plt.imread(temp_png.name)
            ax_new.imshow(img)
            ax_new.axis('off')
            ax_new.set_title(title, fontsize=16, fontweight='bold')
            plt.tight_layout()
            if plt.get_backend().lower() != "agg":
                plt.show()
            
            # Clean up temp file
            os.unlink(temp_png.name)
            return
            
        except (ImportError, Exception) as e:
            # Fallback: use NetworkX with hierarchical layout
            pass

        # Fallback: NetworkX multipartite layout (top-to-bottom)
        own_figure = False
        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)
            own_figure = True

        try:
            # Compute topological generations for layered layout
            for layer, nodes in enumerate(nx.topological_generations(G)):
                for node in nodes:
                    G.nodes[node]["layer"] = layer
            
            # Use multipartite layout (vertical: top-to-bottom)
            pos = nx.multipartite_layout(G, subset_key="layer", align="vertical", scale=2.0)
            # Flip y-axis so roots are at top, and spread horizontally
            pos = {node: (x * 1.5, -y) for node, (x, y) in pos.items()}
            
        except (nx.NetworkXError, nx.NetworkXUnfeasible):
            pos = nx.kamada_kawai_layout(G, scale=2.0)

        # Black & white styling - larger nodes and fonts
        font_size = max(9, min(13, int((180 / max(1, n_nodes)) * node_scale)))
        node_size = max(1600, min(6000, int((25000 / max(1, n_nodes)) * (node_scale ** 2))))
        
        # Draw edges (black)
        nx.draw_networkx_edges(
            G, pos, ax=ax,
            edge_color='black',
            arrows=True,
            arrowsize=15,
            arrowstyle='-|>',
            width=1.0,
            alpha=0.8
        )

        # Edge labels for posterior probabilities
        if edge_labels:
            nx.draw_networkx_edge_labels(
                G,
                pos,
                edge_labels=edge_labels,
                ax=ax,
                font_size=max(7, min(10, int((160 / max(1, n_nodes)) * node_scale))),
                font_color='black'
            )
        
        # Draw nodes (white with black border)
        nx.draw_networkx_nodes(
            G, pos, ax=ax,
            node_color='white',
            node_size=node_size,
            edgecolors='black',
            linewidths=1.5
        )
        
        # Draw labels
        nx.draw_networkx_labels(
            G, pos, labels, ax=ax,
            font_size=font_size,
            font_color='black'
        )
        
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.axis('off')
        ax.margins(0.1)
        
        if own_figure:
            plt.tight_layout()
            if plt.get_backend().lower() != "agg":
                plt.show()
            
    @staticmethod
    def visualize_total_orders(total_orders: List[List[int]], top_print: int = 15, top_plot: int = 10) -> None:        
        """
        Visualize the frequency of total orders.

        Parameters:
            total_orders (List[List[int]]): List of total orders, each order is a list of integers.
            top_print (int): Number of top total orders to print.
            top_plot (int): Number of top total orders to plot.

        Returns:
            None. Prints the top_print total orders and displays a bar plot of the top_plot orders.
        """
        
        # 1. Convert total orders to tuples for counting
        total_orders_tuples = [tuple(order) for order in total_orders]
        
        # 2. Count the frequency of each unique total order
        order_counts = Counter(total_orders_tuples)
        
        # 3. Convert tuples to readable strings for better visualization
        total_orders_strings = [' > '.join(map(str, order)) for order in order_counts.keys()]
        frequencies = list(order_counts.values())
        
        # 4. Create a DataFrame from the counter with readable total orders
        df_order_counts = pd.DataFrame({
            'Total Order': total_orders_strings,
            'Frequency': frequencies
        })
        
        # 5. Sort the DataFrame by frequency in descending order
        df_order_counts.sort_values(by='Frequency', ascending=False, inplace=True)
        
        # 6. Reset index for better readability
        df_order_counts.reset_index(drop=True, inplace=True)
        
        # 7. Print the top_print most frequent total orders
        print(f"\nTop {top_print} Most Frequent Total Orders:")
        print(df_order_counts.head(top_print))
        
        # 8. Visualize the frequency counts using Seaborn's barplot
        sns.set(style="whitegrid")  # Set the aesthetic style of the plots
        plt.figure(figsize=(14, 8))  # Set the figure size for better readability
        
        # Create the barplot for top_plot total orders
        sns.barplot(
            x='Total Order',
            y='Frequency',
            data=df_order_counts.head(top_plot),
            palette='viridis'  # Choose a color palette
        )
        
        # Add titles and labels with increased font sizes for clarity
        plt.title(f'Top {top_plot} Most Frequent Total Orders', fontsize=16)
        plt.xlabel('Total Order', fontsize=14)
        plt.ylabel('Frequency', fontsize=14)
        
        # Rotate x-axis labels for better readability
        plt.xticks(rotation=45, ha='right')
        
        # Adjust layout to prevent clipping of tick-labels
        plt.tight_layout()
        
        # Display the plot
        plt.show()

        @staticmethod   
        def plot_heatmap_and_graph(h_matrix: np.ndarray, title: str, item_labels: Optional[List[str]] = None) -> plt.Figure:
            """
            Create a figure with two subplots:
            - Left: A heatmap of the partial order (h_matrix)
            - Right: A network graph visualization using a spring layout
            
            Parameters:
            h_matrix (np.ndarray): The partial order adjacency matrix.
            title (str): A title for the plots.
            item_labels (list, optional): Labels for items used as tick labels.
            
            Returns:
            fig (plt.Figure): The figure containing the two subplots.
            """
            fig, axes = plt.subplots(1, 2, figsize=(12, 6))
            
            # Left subplot: Heatmap of h_matrix
            sns.heatmap(h_matrix, annot=True, cmap="viridis", 
                        xticklabels=item_labels, yticklabels=item_labels, ax=axes[0])
            axes[0].set_title("Heatmap: " + title)
            axes[0].set_xlabel("Items")
            axes[0].set_ylabel("Items")
            
            # Right subplot: Network graph using spring layout
            G = nx.DiGraph()
            n = h_matrix.shape[0]
            # Add nodes, using item_labels if available
            for idx in range(n):
                label = item_labels[idx] if item_labels and idx < len(item_labels) else str(idx)
                G.add_node(idx, label=label)
            # Add edges from the adjacency matrix
            for i in range(n):
                for j in range(n):
                    if h_matrix[i, j] == 1:
                        G.add_edge(i, j)
                        
            pos = nx.spring_layout(G)
            nx.draw(G, pos, with_labels=True, labels=nx.get_node_attributes(G, 'label'),
                    node_size=2000, node_color='lightblue', arrowsize=20, ax=axes[1])
            axes[1].set_title("Graph: " + title)
            axes[1].axis('off')
            
            plt.tight_layout()
            return fig
        @staticmethod   
        def plot_mcmc_results(result_dict: Dict[str, Any], pdf_filename: str, item_labels: Optional[List[str]] = None) -> None:
            pp = PdfPages(pdf_filename)
            
            # Use the length of rho_trace for the iteration axis.
            iterations = range(len(result_dict["rho_trace"]))
            
            # --- Page 1: rho trace ---
            fig, ax = plt.subplots(figsize=(8,6))
            ax.plot(iterations, result_dict["rho_trace"], label="rho", marker='o')
            ax.set_xlabel("Iteration")
            ax.set_ylabel("rho")
            ax.set_title("Trace of rho")
            ax.legend()
            ax.grid(True)
            pp.savefig(fig)
            plt.close(fig)
            
            # --- Page 2: tau trace ---
            fig, ax = plt.subplots(figsize=(8,6))
            ax.plot(iterations, result_dict["tau_trace"], label="tau", color="orange", marker='o')
            ax.set_xlabel("Iteration")
            ax.set_ylabel("tau")
            ax.set_title("Trace of tau")
            ax.legend()
            ax.grid(True)
            pp.savefig(fig)
            plt.close(fig)
            
            # --- Page 3: Noise parameters ---
            fig, ax = plt.subplots(figsize=(8,6))
            ax.plot(iterations, result_dict["prob_noise_trace"], label="prob_noise", color="green", marker='o')
            ax.plot(iterations, result_dict["mallow_theta_trace"], label="mallow_theta", color="red", marker='o')
            ax.set_xlabel("Iteration")
            ax.set_ylabel("Noise Parameters")
            ax.set_title("Trace of Noise Parameters")
            ax.legend()
            ax.grid(True)
            pp.savefig(fig)
            plt.close(fig)
            
            # --- Page 4: Log Likelihoods ---
            fig, ax = plt.subplots(figsize=(8,6))
            ax.plot(iterations, result_dict["log_likelihood_currents"], label="Current Log Likelihood", color="blue", marker='o')
            ax.plot(iterations, result_dict["log_likelihood_primes"], label="Proposed Log Likelihood", 
                    color="purple", marker='o', linestyle="--")
            ax.set_xlabel("Iteration")
            ax.set_ylabel("Log Likelihood")
            ax.set_title("Log Likelihood Trace")
            ax.legend()
            ax.grid(True)
            pp.savefig(fig)
            plt.close(fig)
            
            # --- Page 5: Acceptance Rates ---
            fig, ax = plt.subplots(figsize=(8,6))
            ax.plot(iterations, result_dict["acceptance_rates"], label="Cumulative Acceptance Rate", color="magenta", marker='o')
            ax.set_xlabel("Iteration")
            ax.set_ylabel("Acceptance Rate")
            ax.set_title("Cumulative Acceptance Rate")
            ax.legend()
            ax.grid(True)
            pp.savefig(fig)
            plt.close(fig)
            
            # --- Page 6: Final Partial Orders with Graphs ---
            if "H_final" in result_dict:
                H_final = result_dict["H_final"]
                # Plot global partial order if available (assumed under key 0)
                if 0 in H_final:
                    fig = PO_plot.plot_heatmap_and_graph(H_final[0], "Final Global Partial Order (H0)", item_labels=item_labels)
                    pp.savefig(fig)
                    plt.close(fig)
                # Plot assessor-level partial orders
                for a in H_final:
                    if a == 0:
                        continue
                    value = H_final[a]
                    if isinstance(value, dict):
                        for task, hm in value.items():
                            fig = PO_plot.plot_heatmap_and_graph(hm, f"Assessor {a} - Task {task} Partial Order", item_labels=item_labels)
                            pp.savefig(fig)
                            plt.close(fig)
                    elif isinstance(value, np.ndarray):
                        fig = PO_plot.plot_heatmap_and_graph(value, f"Assessor {a} Partial Order", item_labels=item_labels)
                        pp.savefig(fig)
                        plt.close(fig)
            
            pp.close()
            print(f"Plots saved to {pdf_filename}")



    @staticmethod
    def plot_inferred_variables(mcmc_results: Dict[str, Any],
                                config: Dict[str, Any],
                                burn_in: int = 100,
                                output_filename: str = "inferred_parameters.pdf",
                                output_filepath: str = ".",
                                assessors: Optional[List[int]] = None,
                                M_a_dict: Optional[Dict[int, Any]] = None,
                                paper_format: bool = False) -> None:
        """
        Plot MCMC traces and posterior densities for inferred parameters (rho, tau, K, softmax_beta, and noise parameters)
        excluding beta. Shows only posterior distributions without true value comparisons.

        Args:
            paper_format: If True, use smaller fonts suitable for academic papers
        """
        sns.set_style("whitegrid")
        if paper_format:
            sns.set_context("paper", font_scale=0.8)
        else:
            sns.set_context("talk", font_scale=1.1)

        # Define configurations for each variable we want to plot (excluding beta).
        var_configs = {
            'rho': {
                'color': '#1f77b4',
                'prior': 'beta',
                'prior_params': {
                    'a': 1.0,
                    'b': config["prior"].get("rho_prior", 1.0)
                },
                'truncated': True  # plot rho only up to 1 - tol
            },
            'tau': {
                'color': 'brown',
                'prior': 'uniform'
            },
            'K': {
                'color': 'darkcyan',
                # Use a truncated Poisson prior for K.
                'prior': 'truncated_poisson',
                'prior_params': {
                    'lambda': config["prior"].get("K_prior", 1.0)
                }
            }
        }

        # Add softmax parameters if present (common in queue-based models)
        if 'softmax_beta_trace' in mcmc_results:
            var_configs['softmax_beta'] = {
                'color': 'red',
                'prior': None  # Often no explicit prior in the code
            }

        # Add epsilon parameter if present
        if 'epsilon_trace' in mcmc_results:
            var_configs['epsilon'] = {
                'color': 'green',
                'prior': None
            }

        # Add noise parameters if present.
        noise_model = config.get("noise", {}).get("noise_option", "").lower()
        if noise_model == "queue_jump":
            var_configs['prob_noise'] = {
                'color': 'orange',
                'prior': 'beta',
                'prior_params': {
                    'a': 1.0,
                    'b': config["prior"].get("noise_beta_prior", 1.0)
                }
            }
        elif noise_model == "mallows_noise":
            var_configs['mallow_theta'] = {
                'color': 'purple',
                'prior': None
            }
            
        # Extract traces (post burn-in)
        traces = {}
        for var_name, var_config in var_configs.items():
            trace_key = f"{var_name}_trace"
            if trace_key in mcmc_results and mcmc_results[trace_key] is not None:
                trace_after_burnin = np.array(mcmc_results[trace_key])[burn_in:]
                if len(trace_after_burnin) > 0:  # Only include non-empty traces
                    traces[var_name] = trace_after_burnin
                else:
                    print(f"Warning: Empty trace for {var_name} after burn-in, skipping")
        
        # Create subplots: one row per variable, 2 columns (trace and density)
        n_vars = len(traces)
        if n_vars == 0:
            print("Error: No valid traces found after burn-in. Cannot create plots.")
            return
        
        # Increase width a bit if necessary (especially to widen the K axis)
        fig, axes = plt.subplots(n_vars, 2, figsize=(14, 4 * n_vars), squeeze=False)
        
        for idx, (var_name, trace) in enumerate(traces.items()):
            var_config = var_configs[var_name]
            
            # --- Trace Plot ---
            ax_trace = axes[idx, 0]
            iterations = np.arange(burn_in + 1, burn_in + 1 + len(trace))
            # Assume trace is 1D for these parameters.
            ax_trace.plot(iterations, trace, color=var_config['color'], lw=1.2, alpha=0.8)
            ax_trace.set_ylabel(var_name, fontsize=12)
            ax_trace.set_xlabel("Iteration", fontsize=12)
            ax_trace.set_title(f"Trace: {var_name}", fontsize=14)
            ax_trace.grid(True, alpha=0.3)
            
            # --- Density / Histogram Plot ---
            ax_hist = axes[idx, 1]
            if var_name == 'rho' and var_config.get('truncated', False):
                tol = 1e-4
                trunc_point = 1 - tol
                bin_edges = np.linspace(0, trunc_point, 101)
                bin_edges[-1] += 1e-6
                sns.histplot(trace, kde=False, ax=ax_hist, color=var_config['color'],
                            bins=bin_edges, edgecolor='black', linewidth=0)
                ax_hist.set_xlim(0.5, trunc_point)
                x_vals = np.linspace(0.5, trunc_point, 1000)
                norm_const = beta_dist.cdf(trunc_point, **var_config['prior_params'])
                norm_const = max(norm_const, 1e-15)
                prior_pdf = beta_dist.pdf(x_vals, **var_config['prior_params']) / norm_const
                ax_hist.plot(x_vals, prior_pdf, 'k-', lw=2, label='Theoretical PDF')
            elif var_name == 'K' and var_config['prior'] == 'truncated_poisson':
                lam = var_config['prior_params']['lambda']
                truncated_poisson = StatisticalUtils.TruncatedPoisson(lam)
                x_vals = np.arange(1, int(np.max(trace)) + 3)
                pdf_vals = np.array([truncated_poisson.pdf(x) for x in x_vals])
                counts, _ = np.histogram(trace, bins=np.append(x_vals, x_vals[-1]+1))
                ax_hist.bar(x_vals, counts, color=var_config['color'], alpha=0.5, width=0.8, label='Sampled')
                scale_factor = len(trace)
                ax_hist.plot(x_vals, pdf_vals * scale_factor, 'k--', lw=2, label='Trunc. Poisson Prior')
                ax_hist.set_xticks(x_vals)
                ax_hist.set_xlabel(var_name, fontsize=12)
            else:
                sns.histplot(trace, kde=True, ax=ax_hist, color=var_config['color'], alpha=0.5)
                if var_config.get('prior') == 'beta' and var_name != 'rho':
                    x_vals = np.linspace(0, 1, 300)
                    pdf_vals = beta_dist.pdf(x_vals, **var_config['prior_params'])
                    scale_factor = len(trace) * (ax_hist.get_xlim()[1] / 30.0)
                    ax_hist.plot(x_vals, pdf_vals * scale_factor, 'k--', label='Beta Prior')
                elif var_config.get('prior') == 'uniform':
                    if len(trace) > 0:
                        x_vals = np.linspace(0, max(trace)*1.1, 300)
                        pdf_vals = np.ones_like(x_vals)
                        scale_factor = len(trace) * (ax_hist.get_xlim()[1] / 30.0)
                        ax_hist.plot(x_vals, pdf_vals * scale_factor, 'k--', label='Uniform Prior')
                    else:
                        print(f"Warning: Empty trace for {var_name}, skipping prior plot")
                    
            ax_hist.set_title(f"Density: {var_name}", fontsize=14)
            ax_hist.set_xlabel(var_name, fontsize=12)
            ax_hist.set_ylabel("Count", fontsize=12)
            sample_mean = np.mean(trace)
            ax_hist.axvline(sample_mean, color='green', linestyle='--', linewidth=1.5, label='Posterior Mean')
            ax_hist.legend(loc="best", fontsize=10)
        
        plt.tight_layout()
        full_output = os.path.join(output_filepath, output_filename)
        plt.savefig(full_output, dpi=300, bbox_inches='tight')
        print(f"[INFO] Saved inferred parameters plot to '{full_output}'")
        plt.show()
        
    
# -------------------------------
# Function 2: Plot Beta Parameters Separately
# -------------------------------
    @staticmethod
    def plot_beta_parameters(mcmc_results: Dict[str, Any],
                            true_param: Dict[str, Any],
                            config: Dict[str, Any],
                            burn_in: int = 100,
                            output_filepath: str = ".") -> None:
        """
        Plot each component of beta in a separate figure. Each figure has two subplots:
        one for the trace and one for the density. Font sizes for beta labels are set very small.
        
        Assumes that mcmc_results["beta_trace"] is a 2D array with shape (n_samples, p),
        and that true_param["beta_true"] is a NumPy array of length p.
        """
        sns.set_style("whitegrid")
        # Use a small font for beta plots.
        beta_font = {
            'title': 8,
            'label': 7,
            'legend': 6,
            'ticks': 6
        }
        
        # Extract beta trace and true beta
        beta_trace = np.array(mcmc_results.get("beta_trace", []))
        if beta_trace.size == 0:
            print("No beta trace available.")
            return
        beta_trace = beta_trace[burn_in:]
        true_beta = true_param.get("beta_true", None)
        
        # Determine dimensions
        n_samples, p_dim = beta_trace.shape
        
        # Create one separate figure per beta coefficient.
        for d in range(p_dim):
            fig, (ax_trace, ax_hist) = plt.subplots(1, 2, figsize=(8, 3))
            iterations = np.arange(burn_in + 1, burn_in + 1 + n_samples)
            
            # TRACE subplot for beta_d
            ax_trace.plot(iterations, beta_trace[:, d], color=plt.cm.tab10(d), lw=1.2, alpha=0.8)
            ax_trace.set_title(f"β{d} Trace", fontsize=beta_font['title'])
            ax_trace.set_xlabel("Iteration", fontsize=beta_font['label'])
            ax_trace.set_ylabel("β value", fontsize=beta_font['label'])
            ax_trace.tick_params(axis='both', labelsize=beta_font['ticks'])
            ax_trace.grid(True, alpha=0.3)
            
            # DENSITY subplot for beta_d
            sns.histplot(beta_trace[:, d], kde=True, ax=ax_hist, color=plt.cm.tab10(d), alpha=0.5)
            ax_hist.set_title(f"β{d} Density", fontsize=beta_font['title'])
            ax_hist.set_xlabel("β value", fontsize=beta_font['label'])
            ax_hist.set_ylabel("Count", fontsize=beta_font['label'])
            ax_hist.tick_params(axis='both', labelsize=beta_font['ticks'])
            if true_beta is not None and d < len(true_beta):
                ax_hist.axvline(true_beta[d], color=plt.cm.tab10(d), linestyle='--', lw=1,
                                label="True β")
            sample_mean = np.mean(beta_trace[:, d])
            ax_hist.axvline(sample_mean, color='green', linestyle='--', lw=1, label="Sample Mean")
            ax_hist.legend(fontsize=beta_font['legend'])
            
            plt.tight_layout()
            outname = os.path.join(output_filepath, f"beta_{d}_plot.pdf")
            plt.savefig(outname, dpi=300, bbox_inches='tight')
            print(f"[INFO] Saved beta coefficient {d} plot to '{outname}'")
            plt.show()



    @staticmethod
    #  This function is used to compare two MCMC states
    def compare_two_mcmc_states(
        results_dict: Dict[str, Any],
        h_U_dict: Dict[int, np.ndarray],   # True partial order dictionary: h_U_dict[0] = global, h_U_dict[a] = assessor a
        iteration_idx_1: int,
        iteration_idx_2: int,
        threshold: float = 0.5,
        assessors: Optional[List[int]] = None,
        items: Optional[List[str]] = None,
        plot_partial_orders: bool = True
    ) -> Dict[str, Any]:
        """
        Compare two sampled states from the MCMC trace and the true partial order (h_U_dict).
        We produce a 3-column plot showing:
        - True partial order
        - Iteration 1 partial order
        - Iteration 2 partial order

        The first (top) row is for the global partial order (key=0),
        and subsequent rows are for each assessor in 'assessors'.
        """

        # 1. Convert iteration to trace indices for partial orders
        idx1 = iteration_idx_1 // 100   # thi isbecause when we store the trace, we store it every 100 iterations 
        idx2 = iteration_idx_2 // 100

        # 2. Retrieve partial orders from the MCMC traces
        H_trace = results_dict.get("H_trace", [])
        if not H_trace:
            raise ValueError("No partial-order trace found in results_dict['H_trace'].")
        max_idx = len(H_trace) - 1
        if idx1 > max_idx or idx2 > max_idx:
            raise ValueError(f"H_trace has {len(H_trace)} samples, so max index is {max_idx}, "
                            f"but requested indices are {idx1} and {idx2}.")

        state1 = H_trace[idx1]  # dict: state1[0] = global adjacency, state1[a] = adjacency for assessor a
        state2 = H_trace[idx2]

        if assessors is None:
            assessors = []

        # We'll define a helper to threshold a matrix
        def threshold_matrix(mat: np.ndarray, thr: float) -> np.ndarray:
            return (mat >= thr).astype(int)

        # 3. Plot partial orders
        if plot_partial_orders:
            # Figure with (n_assessors + 1) rows, 3 columns
            n_rows = len(assessors) + 1
            n_cols = 3
            fig, axes = plt.subplots(nrows=n_rows, ncols=n_cols, figsize=(5*n_cols, 4*n_rows))

            # If there's only 1 row (0 assessors), axes might be 1D. 
            # Convert to 2D array for consistent indexing
            if n_rows == 1:
                axes = np.array([axes])  # shape (1,3)

            # A small function to handle each "cell" in the grid
            def plot_PO_in_cell(ax, matrix: Optional[np.ndarray], item_labels: List[str], 
                                title: str):
                """
                Plots the partial order (matrix) onto a given Axes (ax).
                If matrix is None, we just leave it blank.
                """
                if matrix is None:
                    ax.set_title(f"{title}\n(no data)")
                    ax.axis("off")
                    return

                # If requested, do thresholding or transitive reduction.
                # But first do thresholding if needed
                # (in your code you did thresholding outside this function)
                matrix = BasicUtils.transitive_reduction(matrix)

                # Actually draw onto the Axes
                PO_plot.visualize_partial_order(
                    final_h=matrix,
                    Ma_list=item_labels,
                    title=title,
                    ax=ax
                )

            # (A) The top row is the global partial order (key=0)
            row_idx = 0
            # col 0: True partial order from h_U_dict[0]
            ax_true_global = axes[row_idx, 0]
            true_global = h_U_dict.get(0, None)
            plot_PO_in_cell(
                ax=ax_true_global,
                matrix=true_global,
                item_labels=items if items else [],
                title="True Global PO"
            )

            # col 1: iteration_idx_1 partial order
            ax_iter1_global = axes[row_idx, 1]
            mat1_global = state1.get(0, None)
            if mat1_global is not None:
                mat1_global = threshold_matrix(mat1_global, threshold)
            plot_PO_in_cell(
                ax=ax_iter1_global,
                matrix=mat1_global,
                item_labels=items if items else [],
                title=f"Global PO (Iter {iteration_idx_1})"
            )

            # col 2: iteration_idx_2 partial order
            ax_iter2_global = axes[row_idx, 2]
            mat2_global = state2.get(0, None)
            if mat2_global is not None:
                mat2_global = threshold_matrix(mat2_global, threshold)
            plot_PO_in_cell(
                ax=ax_iter2_global,
                matrix=mat2_global,
                item_labels=items if items else [],
                title=f"Global PO (Iter {iteration_idx_2})"
            )

            # (B) Rows for each assessor
            for i, assessor in enumerate(assessors, start=1):
                # col 0 => True partial order for this assessor
                ax_true_local = axes[i, 0]
                true_local = h_U_dict.get(assessor, None)
                plot_PO_in_cell(
                    ax=ax_true_local,
                    matrix=true_local,
                    item_labels=items if items else [],
                    title=f"True PO (Assessor {assessor})"
                )

                # col 1 => iteration_idx_1 partial order for assessor
                ax_iter1_local = axes[i, 1]
                mat1_local = state1.get(assessor, None)
                if mat1_local is not None:
                    mat1_local = threshold_matrix(mat1_local, threshold)
                plot_PO_in_cell(
                    ax=ax_iter1_local,
                    matrix=mat1_local,
                    item_labels=items if items else [],
                    title=f"PO (Assr {assessor}, Iter {iteration_idx_1})"
                )

                # col 2 => iteration_idx_2 partial order for assessor
                ax_iter2_local = axes[i, 2]
                mat2_local = state2.get(assessor, None)
                if mat2_local is not None:
                    mat2_local = threshold_matrix(mat2_local, threshold)
                plot_PO_in_cell(
                    ax=ax_iter2_local,
                    matrix=mat2_local,
                    item_labels=items if items else [],
                    title=f"PO (Assr {assessor}, Iter {iteration_idx_2})"
                )

            plt.tight_layout()
            plt.show()

        # 4. Retrieve parameter values from the relevant traces
        param_rho   = results_dict.get("rho_trace", [])
        param_tau   = results_dict.get("tau_trace", [])
        param_noise = results_dict.get("prob_noise_trace", [])
        param_mth   = results_dict.get("mallow_theta_trace", [])
        param_beta  = results_dict.get("beta_trace", [])
        param_K     = results_dict.get("K_trace", [])

        def get_or_none(param_list, ix):
            if (ix < 0) or (ix >= len(param_list)):
                return None
            return param_list[ix]

        rho1 = get_or_none(param_rho,   idx1)
        tau1 = get_or_none(param_tau,   idx1)
        pn1  = get_or_none(param_noise, idx1)
        mth1 = get_or_none(param_mth,   idx1)
        beta1 = get_or_none(param_beta, idx1)
        K1 = get_or_none(param_K, idx1)
        rho2 = get_or_none(param_rho,   idx2)
        tau2 = get_or_none(param_tau,   idx2)
        pn2  = get_or_none(param_noise, idx2)
        mth2 = get_or_none(param_mth,   idx2)
        beta2 = get_or_none(param_beta, idx2)
        K2 = get_or_none(param_K, idx2)
        # 5. Log-likelihood retrieval
        ll_list = results_dict.get("log_likelihood_currents", [])
        llk1 = ll_list[iteration_idx_1] if iteration_idx_1 < len(ll_list) else None
        llk2 = ll_list[iteration_idx_2] if iteration_idx_2 < len(ll_list) else None

        # 6. Build final adjacency data: threshold + transitive_reduction for iteration1 & iteration2
        iteration1_final = {}
        iteration2_final = {}

        # We'll collect both global (0) and assessors
        all_keys = [0] + assessors

        for a in all_keys:
            mat_1 = state1.get(a, None)
            mat_2 = state2.get(a, None)

            if mat_1 is not None:
                bin_1 = (mat_1 >= threshold).astype(int)
                red_1 = BasicUtils.transitive_reduction(bin_1) 
                iteration1_final[a] = red_1

            if mat_2 is not None:
                bin_2 = (mat_2 >= threshold).astype(int)
                red_2 = BasicUtils.transitive_reduction(bin_2) 
                iteration2_final[a] = red_2

        # 7. Return a summary dictionary
        comparison_out = {
            "iteration1_index": iteration_idx_1,
            "iteration2_index": iteration_idx_2,
            "rho1": rho1, "tau1": tau1, "prob_noise1": pn1, "mallow_theta1": mth1,
            "beta1": beta1, "K1": K1,
            "rho2": rho2, "tau2": tau2, "prob_noise2": pn2, "mallow_theta2": mth2,
            "beta2": beta2, "K2": K2,
            "loglik1": llk1, "loglik2": llk2,
            "iteration1_final": iteration1_final,
            "iteration2_final": iteration2_final
        }

        return comparison_out

    @staticmethod
    def print_mcmc_comparison_table(comparison_out, rho_true, tau_true, prob_noise_true):
        # Extract iteration indices
        iterationA = comparison_out["iteration1_index"]
        iterationB = comparison_out["iteration2_index"]

        # Collect rows of data in a list of lists
        table_data = [
            [
                "Iteration 1",
                iterationA,
                comparison_out["rho1"],
                comparison_out["tau1"],
                comparison_out["prob_noise1"],
                comparison_out["mallow_theta1"],
                comparison_out["loglik1"]
            ],
            [
                "Iteration 2",
                iterationB,
                comparison_out["rho2"],
                comparison_out["tau2"],
                comparison_out["prob_noise2"],
                comparison_out["mallow_theta2"],
                comparison_out["loglik2"]
            ],
            [
                "True Params",
                "–",  # No specific iteration index for true params
                rho_true,
                tau_true,
                prob_noise_true,
                "–",  # If you have a true Mtheta, replace "–" with that
                "–"  # True log-likelihood typically not available; replace if you have it
            ]
        ]

        # Define the table headers
        headers = [
            "Label",
            "Iteration",
            "rho",
            "tau",
            "prob_noise",
            "mtheta",
            "log-likelihood"
        ]

        # Print the table in a nice grid format
        print("\n--- Comparison Output Summary (Tabular) ---")
        print(tabulate(table_data, headers=headers, tablefmt="fancy_grid"))



    @staticmethod
    # Display the first few rows for verification.
    def plot_time_components_by_category(df: pd.DataFrame) -> None:
        """
        Plots per-iteration time trends for each timing component (PriorTime, LikelihoodTime,
        UpdateTime) broken down by update category.
        
        The function creates three vertically arranged subplots (one per component). 
        Each subplot plots a line (with markers) for each update category.

        Parameters:
        -----------
        df : pd.DataFrame
            A DataFrame containing the following columns:
            - "Iteration": iteration number
            - "UpdateCategory": update category (e.g. "rho", "tau", "noise", "U0", "Ua", "rho_tau")
            - "PriorTime": time spent on prior computations in that iteration
            - "LikelihoodTime": time spent on likelihood calculations in that iteration
            - "UpdateTime": time spent on the update branch in that iteration
        """
        # Set a clean seaborn style.
        sns.set(style="whitegrid", context="talk")
        
        # Filter out rows where UpdateCategory is missing.
        df = df[df["UpdateCategory"].notna()].copy()
        
        # Check if any columns contain dictionary values and convert them to strings
        for col in df.columns:
            if df[col].apply(lambda x: isinstance(x, dict)).any():
                print(f"Warning: Column '{col}' contains dictionary values. Converting to strings.")
                df[col] = df[col].apply(lambda x: str(x) if isinstance(x, dict) else x)
        
        # Unique update categories.
        categories = sorted(df["UpdateCategory"].unique())
        
        # Define the three timing components to plot.
        components = ["PriorTime", "LikelihoodTime", "UpdateTime"]
        # Define distinct colors for each update category (modify as needed).
        color_map = {"rho": "blue", "tau": "green", "noise": "red", "U0": "purple", "Ua": "orange", "rho_tau": "brown", "K_dim": "cyan", "beta": "magenta"}
        
        # Create one subplot for each timing component.
        fig, axes = plt.subplots(len(components), 1, figsize=(14, 12), sharex=True)
        
        for comp, ax in zip(components, axes):
            for cat in categories:
                # Filter rows for this update category and sort by iteration.
                subset = df[df["UpdateCategory"] == cat].sort_values("Iteration")
                # Get a color for the category, or default to None.
                cat_color = color_map.get(cat, None)
                ax.plot(subset["Iteration"], subset[comp],
                        marker='o', markersize=1, linestyle='-', label=cat,
                        color=cat_color)
            ax.set_ylabel(f"{comp} (s)", fontsize=14)
            ax.legend(title="Update Category", fontsize=12)
            ax.tick_params(axis='both', which='major', labelsize=12)
        
        axes[-1].set_xlabel("Iteration", fontsize=14)
        fig.suptitle("Per-Iteration Time Trends by Update Category", fontsize=16)
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        plt.show()



        # ------------------ Plot 2: Stacked Bar Plot by Update Category ------------------
        # Group by update category and compute the mean (or total) timing for each component.
        grouped = df.groupby("UpdateCategory")[["PriorTime", "LikelihoodTime"]].mean()

        plt.figure(figsize=(10, 6))
        grouped.plot(kind="bar", stacked=True, color=["lightcoral", "lightskyblue"],
                    edgecolor='black')
        plt.xlabel("Update Category", fontsize=12)
        plt.ylabel("Average Time per Iteration (seconds)", fontsize=12)
        plt.title("Average Timing Breakdown per Update Category", fontsize=14)
        plt.legend(title="Component", fontsize=10)
        plt.grid(axis="y", linestyle="--", alpha=0.7)
        plt.tight_layout()
        plt.show()


    @staticmethod
    def print_comparison_summary(comparison_out):
        """
        Print a formatted summary of the output from a comparison of two MCMC iterations.
        Includes parameter values, log-likelihoods, and partial order adjacency matrices.
        """
        print("\n========== Comparison Output Summary ==========")
        print(f"Iterations: {comparison_out['iteration1_index']} vs {comparison_out['iteration2_index']}\n")

        print("---- Iteration 1 Parameters ----")
        print(f"rho         = {comparison_out['rho1']}")
        print(f"tau         = {comparison_out['tau1']}")
        print(f"prob_noise  = {comparison_out['prob_noise1']}")
        print(f"mallow_theta= {comparison_out['mallow_theta1']}")
        print(f"log-likelihood = {comparison_out['loglik1']:.4f}\n")

        print("---- Iteration 2 Parameters ----")
        print(f"rho         = {comparison_out['rho2']}")
        print(f"tau         = {comparison_out['tau2']}")
        print(f"prob_noise  = {comparison_out['prob_noise2']}")
        print(f"mallow_theta= {comparison_out['mallow_theta2']}")
        print(f"log-likelihood = {comparison_out['loglik2']:.4f}\n")

        print("---- Iteration 1 Partial Orders (transitive-reduced) ----")
        for a, adj in comparison_out["iteration1_final"].items():
            print(f"Assessor {a}, shape = {adj.shape}\n{adj}\n")

        print("---- Iteration 2 Partial Orders (transitive-reduced) ----")
        for a, adj in comparison_out["iteration2_final"].items():
            print(f"Assessor {a}, shape = {adj.shape}\n{adj}\n")

        print("=========================================================")



    @staticmethod
    def plot_update_acceptance_by_category(mcmc_results, desired_order=None, jitter_strength=0.08):
        """
        Summarize and plot MCMC update acceptance by category.

        Parameters:
        - mcmc_results: dict containing key "update_df" with a DataFrame of updates.
        - desired_order: list of category names in the order you'd like them plotted.
                        If None, it defaults to a common order.
        - jitter_strength: float controlling the amount of vertical jitter for visualization.
        """
        if desired_order is None:
            desired_order = ["rho", "tau", "rho_tau","K_dim","beta", "noise", "U0", "Ua"]

        # Create a mapping: category name → numeric value
        category_to_num = {cat: i for i, cat in enumerate(desired_order)}

        # Filter the update DataFrame
        update_df = mcmc_results["update_df"]
        update_df_filtered = update_df[update_df["category"].isin(desired_order)]

        print("\n--- MCMC Update Acceptance Rates by Category ---")
        for category in desired_order:
            cat_data = update_df_filtered[update_df_filtered["category"] == category]
            if not cat_data.empty:
                acceptance_rate = cat_data["accepted"].mean()
                print(f"{category:10s}: {acceptance_rate * 100:.2f}%")
            else:
                print(f"{category:10s}: No updates recorded.")

        # Plotting
        plt.figure(figsize=(25, 6))
        accepted_plotted = set()
        rejected_plotted = set()

        for category in desired_order:
            cat_data = update_df_filtered[update_df_filtered["category"] == category]
            if cat_data.empty:
                continue

            numeric_cat = category_to_num[category]
            accepted = cat_data[cat_data["accepted"]]
            rejected = cat_data[~cat_data["accepted"]]

            if not accepted.empty:
                y_jitter = np.random.uniform(-jitter_strength, jitter_strength, size=len(accepted))
                label = "Accepted" if "Accepted" not in accepted_plotted else None
                accepted_plotted.add("Accepted")
                plt.scatter(accepted["iteration"], numeric_cat + y_jitter,
                            color="green", marker="o", s=20, alpha=0.8, label=label)

            if not rejected.empty:
                y_jitter = np.random.uniform(-jitter_strength, jitter_strength, size=len(rejected))
                label = "Rejected" if "Rejected" not in rejected_plotted else None
                rejected_plotted.add("Rejected")
                plt.scatter(rejected["iteration"], numeric_cat + y_jitter,
                            color="red", marker="x", s=20, alpha=0.8, label=label)

        # Format plot
        plt.yticks(ticks=list(category_to_num.values()), labels=desired_order, fontsize=12)
        plt.xlabel("Iteration", fontsize=14)
        plt.ylabel("Update Category", fontsize=14)
        plt.title("Update Category by Iteration (Green=Accepted, Red=Rejected)", fontsize=16)
        plt.grid(axis="y", linestyle="--", alpha=0.5)
        plt.legend()
        plt.tight_layout()
        plt.show()





    @staticmethod
    def compare_and_visualize_global(
        h_true_global: np.ndarray,
        h_inferred_global: np.ndarray,
        index_to_item_global: Dict[int, int],
        global_Ma_list: List[str],
        do_transitive_reduction: bool = True
    ) -> None:

        h_true_plot = BasicUtils.transitive_reduction(h_true_global)
        h_inferred_plot = BasicUtils.transitive_reduction(h_inferred_global)

        PO_plot.visualize_partial_order(
            final_h=h_true_plot,
            Ma_list=global_Ma_list,
            title='True Global Partial Order'
        )
        PO_plot.visualize_partial_order(
            final_h=h_inferred_plot,
            Ma_list=global_Ma_list,
            title='Inferred Global Partial Order'
        )

        # Compute and print missing and redundant relationships.
        missing_relationships = BasicUtils.compute_missing_relationships(h_true_plot,h_inferred_plot, index_to_item_global)
        redundant_relationships = BasicUtils.compute_redundant_relationships(h_true_plot, h_inferred_plot, index_to_item_global)

        if missing_relationships:
            print("\nMissing (true PO edges not in inferred PO):")
            for i, j in missing_relationships:
                print(f"{i} < {j}")
        else:
            print("\nNo missing relationships in global partial order.")

        if redundant_relationships:
            print("\nRedundant (inferred PO edges not in true PO):")
            for i, j in redundant_relationships:
                print(f"{i} < {j}")
        else:
            print("\nNo redundant relationships in global partial order.")

    @staticmethod
    def compare_and_visualize_assessor(
        assessor: int,
        Ma_list: List[str],
        h_true_a: np.ndarray,
        h_inferred_a: np.ndarray,
        index_to_item_local: Dict[int, int],
        do_transitive_reduction: bool = True
    ) -> None:
        """
        Compare and visualize the partial orders for a specific assessor, printing missing and redundant edges.

        Parameters
        ----------
        assessor : int
            The assessor's identifier.
        Ma_list : List[str]
            List of item labels for the assessor.
        h_true_a : np.ndarray
            True local partial order adjacency matrix for the assessor.
        h_inferred_a : np.ndarray
            Inferred local partial order adjacency matrix for the assessor.
        index_to_item_local : Dict[int, int]
            Mapping from local indices to items for the assessor.
        do_transitive_reduction : bool, optional
            Whether to apply transitive reduction (default is True).
        """


        h_true_plot = BasicUtils.transitive_reduction(h_true_a)
        h_inferred_plot = BasicUtils.transitive_reduction(h_inferred_a)



        # Visualize local partial orders for the assessor.
        PO_plot.visualize_partial_order(
            final_h=h_true_plot,
            Ma_list=Ma_list,
            title=f"True Local Partial Order (Assessor={assessor})"
        )
        PO_plot.visualize_partial_order(
            final_h=h_inferred_plot,
            Ma_list=Ma_list,
            title=f"Inferred Local Partial Order (Assessor={assessor})"
        )

        # Compute and print missing and redundant relationships.
        missing_relationships = BasicUtils.compute_missing_relationships(h_true_a, h_inferred_a, index_to_item_local)
        redundant_relationships = BasicUtils.compute_redundant_relationships(h_true_a, h_inferred_a, index_to_item_local)

        if missing_relationships:
            print(f"\nMissing edges for assessor {assessor}:")
            for i, j in missing_relationships:
                print(f"{i} < {j}")
        else:
            print(f"\nNo missing edges for assessor {assessor}.")

        if redundant_relationships:
            print(f"\nRedundant edges for assessor {assessor}:")
            for i, j in redundant_relationships:
                print(f"{i} < {j}")
        else:
            print(f"\nNo redundant edges for assessor {assessor}.")

    @staticmethod
    def plot_joint_parameters(mcmc_results):
        """
        Given an mcmc_results dictionary from the HPO simulation, this function
        creates scatter plots of:
        - p (i.e. prob_noise) versus rho
        - p (i.e. prob_noise) versus tau
        so you can inspect the joint behavior of these parameters across MCMC iterations.
        
        Parameters:
        -----------
        mcmc_results : dict
            Dictionary output from mcmc_simulation_hpo, expected to contain:
            - "rho_trace": list of rho values per iteration
            - "tau_trace": list of tau values per iteration
            - "prob_noise_trace": list of noise probability values per iteration (interpreted as p)
        """
        # Extract traces
        rho_trace = mcmc_results["rho_trace"]
        tau_trace = mcmc_results["tau_trace"]
        prob_noise_trace = mcmc_results["prob_noise_trace"]

        # Create a figure with two subplots.
        fig, axes = plt.subplots(1, 3, figsize=(14, 6))

        # Plot p vs. rho
        axes[0].scatter(prob_noise_trace, rho_trace, alpha=0.6 , color='navy')
        axes[0].set_xlabel("p (noise probability)", fontsize=12)
        axes[0].set_ylabel("rho", fontsize=12)
        axes[0].set_title("Joint Behavior: p vs. rho", fontsize=14)
        axes[0].grid(True, linestyle='--', alpha=0.5)

        # Plot p vs. tau
        axes[1].scatter(prob_noise_trace, tau_trace, alpha=0.6, color='darkgreen')
        axes[1].set_xlabel("p (noise probability)", fontsize=12)
        axes[1].set_ylabel("tau", fontsize=12)
        axes[1].set_title("Joint Behavior: p vs. tau", fontsize=14)
        axes[1].grid(True, linestyle='--', alpha=0.5)


        axes[2].scatter(rho_trace, tau_trace, alpha=0.6, color='darkred')
        axes[2].set_xlabel("rho)", fontsize=12)
        axes[2].set_ylabel("tau", fontsize=12)
        axes[2].set_title("Joint Behavior: p vs. tau", fontsize=14)
        axes[2].grid(True, linestyle='--', alpha=0.5)

        plt.tight_layout()
        plt.show()


    @staticmethod

    def plot_u0_ua_diagnostics(results, assessors, M0, K):
        """
        Present posterior diagnostics for U0 and Ua by displaying clean, informative plots.

        Parameters
        ----------
        results : dict
            Output from mcmc_simulation_hpo(...). Must include:
                - "U0_trace": list of (n_global x K) arrays
                - "Ua_trace": list of assessor->(M_a x K) dicts
        assessors : list[int]
            List of assessor IDs.
        M0 : list[int]
            Global item indices.
        K : int
            Dimension of latent space.
        """
        sns.set(style="whitegrid", font_scale=1.1)

        # ─────────────────────────────────────────────────────────────
        # U0 Posterior Diagnostics
        # ─────────────────────────────────────────────────────────────
        U0_trace = results.get("U0_trace", [])


        U0_stack = np.stack(U0_trace)  # (n_samples, n_global, K)
        U0_flat = U0_stack.reshape(-1, K)

        print(f"[U0] Total samples: {U0_flat.shape[0]} | K = {K}")

        # 1. Histogram per U0 dimension
        fig, axes = plt.subplots(1, K, figsize=(4.5 * K, 4))
        axes = axes if isinstance(axes, np.ndarray) else [axes]

        for k in range(K):
            sns.histplot(U0_flat[:, k], bins=100, stat="density", ax=axes[k],
                        color="dodgerblue", edgecolor=None, alpha=0.6)
            axes[k].set_title(f"U0: dim {k}", fontsize=13)
            axes[k].set_xlabel("Latent Value")
            axes[k].set_ylabel("Density")

        plt.suptitle("Posterior Marginals of U0 (Flattened)", fontsize=15, fontweight='bold')
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        plt.show()

        # 2. Correlation matrix
        corr_matrix = np.corrcoef(U0_flat.T)
        plt.figure(figsize=(4, 4))
        sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="coolwarm", square=True, center=0,
                    cbar_kws={"shrink": 0.6})
        plt.title("Correlation Matrix Across U0 Dimensions", fontsize=12)
        plt.tight_layout()
        plt.show()

        # ─────────────────────────────────────────────────────────────
        # Ua Posterior Diagnostics
        # ─────────────────────────────────────────────────────────────
        Ua_trace = results.get("Ua_trace", [])


        for a in assessors:
            all_rows = [Ua[a] for Ua in Ua_trace if a in Ua]
            if not all_rows:
                print(f"[Ua] Assessor {a} not found in Ua trace.")
                continue

            arr_flat = np.concatenate(all_rows, axis=0)  # shape: (n_total_items, K)
            print(f"[Ua] Assessor {a}: {arr_flat.shape[0]} total rows.")

            fig, axes = plt.subplots(1, K, figsize=(4.5 * K, 4))
            axes = axes if isinstance(axes, np.ndarray) else [axes]

            for k in range(K):
                sns.histplot(arr_flat[:, k], bins=100, stat="density", ax=axes[k],
                            color="mediumseagreen", edgecolor=None, alpha=0.6)
                axes[k].set_title(f"Ua: assessor={a}, dim {k}", fontsize=13)
                axes[k].set_xlabel("Latent Value")
                axes[k].set_ylabel("Density")

            plt.suptitle(f"Posterior Marginals for Assessor {a} (Flattened)", fontsize=15, fontweight='bold')
            plt.tight_layout(rect=[0, 0, 1, 0.95])
            plt.show()
                   
    @staticmethod 
    def plot_cluster_posterior_vs_crp_prior(mcmc_result, config=None, data=None, burn_in=80000, max_clusters=10):
        """
        Plot the posterior distribution of number of clusters from MCMC vs CRP prior.
        
        Works with sound data script - gets parameters from config or infers from data.
        
        Parameters:
        -----------
        mcmc_result : dict
            MCMC results containing assessor_cluster_trace
        config : dict, optional
            Configuration dictionary with CRP parameters
        data : dict, optional
            Data dictionary (fallback for parameters)
        burn_in : int
            Burn-in iterations to exclude
        max_clusters : int
            Maximum number of clusters to plot
        """
        print(f"📊 Plotting cluster number posterior vs CRP prior...")
        
        # Get assessor cluster trace after burn-in
        assessor_cluster_trace = mcmc_result.get('assessor_cluster_trace', [])
        
        if not assessor_cluster_trace:
            print("❗ No assessor_cluster_trace found")
            return
        
        # Apply burn-in
        trace_filtered = assessor_cluster_trace[burn_in:]
        print(f"   Using {len(trace_filtered)} iterations after burn-in")
        
        # Count number of clusters in each iteration
        cluster_counts = []
        for iteration_clusters in trace_filtered:
            if hasattr(iteration_clusters, 'values'):
                unique_clusters = len(set(iteration_clusters.values()))
                cluster_counts.append(unique_clusters)
        
        if not cluster_counts:
            print("❗ No cluster counts could be extracted")
            return
            
        print(f"   Cluster count range: {min(cluster_counts)} to {max(cluster_counts)}")
        
        # Calculate posterior distribution
        from collections import Counter
        cluster_count_dist = Counter(cluster_counts)
        total_iterations = len(cluster_counts)
        
        # Prepare data for plotting
        k_values = list(range(1, max_clusters + 1))
        posterior_probs = [cluster_count_dist.get(k, 0) / total_iterations for k in k_values]
        
        print(f"   Posterior distribution calculated")
        
        # Get CRP parameters - try multiple sources
        n_assessors = None
        concentration_theta = None
        discount_d = 0.0  # Default for pure CRP
        
        # Try to get number of assessors from various sources
        if config is not None:
            # From config
            if 'assessors' in config:
                n_assessors = len(config['assessors'])
            elif 'data' in config and 'assessors' in config['data']:
                n_assessors = len(config['data']['assessors'])
            
            # Get CRP parameters from config
            if 'prior' in config:
                concentration_theta = config['prior'].get('Dri_alpha', config['prior'].get('concentration_theta', None))
                discount_d = config['prior'].get('Dri_theta', config['prior'].get('discount_d', 0.0))
        
        # Fallback to data dictionary
        if data is not None:
            if n_assessors is None:
                n_assessors = len(data.get('assessors', []))
            if concentration_theta is None:
                true_params = data.get('true_params', {})
                concentration_theta = true_params.get('concentration_theta', true_params.get('Dri_alpha', None))
                discount_d = true_params.get('discount_d', true_params.get('Dri_theta', 0.0))
        
        # Final fallback: infer from MCMC trace
        if n_assessors is None:
            # Get unique assessors from first iteration
            if trace_filtered:
                n_assessors = len(set(trace_filtered[0].keys()))
        
        # Default concentration parameter if still None
        if concentration_theta is None:
            concentration_theta = 1.0  # Default CRP concentration
            print(f"   Using default concentration parameter: {concentration_theta}")
        
        if n_assessors is None:
            print("❗ Could not determine number of assessors")
            return
        
        print(f"   CRP parameters: n={n_assessors}, θ={concentration_theta}, d={discount_d}")

        # Calculate CRP prior probabilities using correct Pitman-Yor formula
        def calculate_crp_prior(n, theta, d, max_k):
            """Calculate CRP prior probabilities for number of clusters."""
            from scipy.special import gammaln
            import warnings
            
            priors = np.zeros(max_k)
            
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                
                for k in range(1, min(max_k + 1, n + 1)):
                    try:
                        if d == 0:  # Pure CRP (Dirichlet Process)
                            # P(K=k|n,θ) ∝ θ^k * S(n,k) where S(n,k) is unsigned Stirling number
                            log_prob = k * np.log(max(theta, 1e-10))
                            # Simplified approximation for Stirling number
                            if k == 1:
                                log_prob += 0  # S(n,1) = 1
                            else:
                                # Rough approximation: S(n,k) ≈ (n-1)! / (k-1)! for small k
                                log_prob += gammaln(n) - gammaln(k)
                            priors[k-1] = np.exp(min(log_prob, 700))  # Prevent overflow
                        else:  # Pitman-Yor process
                            # P(K=k|n,θ,d) ∝ θ * (θ+d) * ... * (θ+(k-1)d) * |s(n,k)|
                            log_prob = 0
                            for j in range(k):
                                log_prob += np.log(max(theta + j * d, 1e-10))
                            # Approximate unsigned Stirling number
                            log_prob += gammaln(n) - gammaln(k)
                            priors[k-1] = np.exp(min(log_prob, 700))  # Prevent overflow
                    except (ValueError, OverflowError):
                        priors[k-1] = 0
            
            # Normalize
            total = np.sum(priors)
            if total > 0:
                priors = priors / total
            else:
                # Uniform fallback
                priors = np.ones(max_k) / max_k
            
            return priors
        
        # Calculate prior probabilities
        crp_priors = calculate_crp_prior(n_assessors, concentration_theta, discount_d, max_clusters)
        
        print(f"   CRP prior probabilities calculated")
        
        # Create the plot in the same style as your parameter plots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        
        # Left plot: Trace of number of clusters over iterations
        n_clusters_trace = [len(set(iter_clusters.values())) for iter_clusters in trace_filtered]
        ax1.plot(range(len(n_clusters_trace)), n_clusters_trace, color='teal', alpha=0.8, linewidth=0.8)
        ax1.set_xlabel('Iteration')
        ax1.set_ylabel('Number of Clusters')
        ax1.set_title('Trace: Number of Clusters')
        ax1.grid(True, alpha=0.3)
        
        # Right plot: Density histogram with prior overlay
        ax2.hist(cluster_counts, bins=range(1, max(cluster_counts)+2), 
                density=False, alpha=0.7, color='teal', edgecolor='white', linewidth=0.5)
        
        # Overlay theoretical prior as line
        cluster_range = np.array(k_values)
        prior_scaled = crp_priors * total_iterations  # Scale to match histogram
        ax2.plot(cluster_range, prior_scaled, 'k--', linewidth=2, label='CRP Prior')
        
        # Add vertical lines for sample mean
        sample_mean = np.mean(cluster_counts)
        ax2.axvline(sample_mean, color='cyan', linestyle='--', linewidth=2, label='Sample Mean')
        
        # Try to add true clusters if available
        true_clusters = None
        if data is not None:
            true_clusters = len(data.get('cluster_ids', []))
        if config is not None and true_clusters is None:
            # Try to infer from config
            if 'K_true' in config:
                true_clusters = config['K_true']
            elif 'data' in config and 'K_true' in config['data']:
                true_clusters = config['data']['K_true']
        
        if true_clusters is not None and true_clusters > 0:
            ax2.axvline(true_clusters, color='red', linestyle='--', linewidth=2, label='True')
        
        ax2.set_xlabel('Number of Clusters')
        ax2.set_ylabel('Count')
        ax2.set_title('Density: Number of Clusters')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
        # Calculate summary statistics
        posterior_mean = np.sum([k * p for k, p in zip(k_values, posterior_probs)])
        prior_mean = np.sum([k * p for k, p in zip(k_values, crp_priors)])
        
        # Print detailed results
        print(f"\n📈 Detailed Results:")
        print(f"   Posterior distribution:")
        for k, prob in zip(k_values, posterior_probs):
            if prob > 0.001:
                count = cluster_count_dist.get(k, 0)
                print(f"     K={k}: {prob:.3f} ({count:,} iterations)")
        
        print(f"\n   CRP Prior distribution:")
        for k, prob in zip(k_values, crp_priors):
            if prob > 0.001:
                print(f"     K={k}: {prob:.3f}")
        
        print(f"\n   Summary Statistics:")
        print(f"     Posterior mean K: {posterior_mean:.2f}")
        print(f"     Prior mean K: {prior_mean:.2f}")
        print(f"     Sample mean K: {sample_mean:.2f}")
        if true_clusters is not None:
            print(f"     True K: {true_clusters}")
        
        return {
            'k_values': k_values,
            'posterior_probs': posterior_probs,
            'crp_priors': crp_priors,
            'posterior_mean': posterior_mean,
            'prior_mean': prior_mean,
            'sample_mean': sample_mean,
            'cluster_counts': cluster_counts,
            'true_clusters': true_clusters
        }

    @staticmethod
    def plot_cluster_number_distribution(mcmc_results, config=None, burn_in=80000, max_clusters=15):
        """
        Plot the posterior distribution of number of clusters with CRP prior overlay.
        
        Simplified function specifically for sound data analysis.
        
        Parameters:
        -----------
        mcmc_results : dict
            MCMC results containing assessor_cluster_trace
        config : dict, optional
            Configuration dictionary with CRP parameters
        burn_in : int
            Burn-in iterations to exclude (default: 80000)
        max_clusters : int
            Maximum number of clusters to plot (default: 15)
        
        Returns:
        --------
        dict : Analysis results including posterior and prior distributions
        """
        import matplotlib.pyplot as plt
        from collections import Counter
        from scipy.special import gammaln
        import numpy as np
        
        print(f"📊 Analyzing cluster number distribution...")
        
        # Extract cluster trace
        cluster_trace = mcmc_results.get('assessor_cluster_trace', [])
        if not cluster_trace:
            print("❗ No assessor_cluster_trace found in MCMC results")
            return None
        
        # Apply burn-in
        post_burnin = cluster_trace[burn_in:]
        if not post_burnin:
            print("❗ No iterations left after burn-in")
            return None
            
        print(f"   Using {len(post_burnin)} iterations after burn-in of {burn_in}")
        
        # Count clusters in each iteration
        cluster_counts = []
        for iteration in post_burnin:
            if isinstance(iteration, dict):
                n_clusters = len(set(iteration.values()))
                cluster_counts.append(n_clusters)
        
        if not cluster_counts:
            print("❗ Could not extract cluster counts")
            return None
        
        print(f"   Observed cluster range: {min(cluster_counts)} - {max(cluster_counts)}")
        
        # Calculate posterior distribution
        count_dist = Counter(cluster_counts)
        total_samples = len(cluster_counts)
        
        # Prepare plotting data
        k_range = list(range(1, max_clusters + 1))
        posterior = [count_dist.get(k, 0) / total_samples for k in k_range]
        
        # Get CRP parameters from config
        n_assessors = None
        alpha = 1.0  # Default concentration parameter
        
        if config is not None:
            # Try various parameter locations
            if 'prior' in config:
                alpha = config['prior'].get('Dri_alpha', config['prior'].get('concentration_theta', 1.0))
            
            # Try to get number of assessors
            if 'assessors' in config:
                n_assessors = len(config['assessors'])
            elif 'data' in config and 'assessors' in config['data']:
                n_assessors = len(config['data']['assessors'])
        
        # Fallback: infer from MCMC trace
        if n_assessors is None and post_burnin:
            n_assessors = len(set(post_burnin[0].keys()))
        
        if n_assessors is None:
            n_assessors = 20  # Default fallback
            print(f"   Warning: Using default n_assessors = {n_assessors}")
        
        print(f"   CRP parameters: n={n_assessors}, α={alpha}")
        
        # Calculate CRP prior (simplified Dirichlet Process)
        def crp_prior_probs(n, alpha, max_k):
            """Calculate CRP prior probabilities for number of clusters."""
            probs = np.zeros(max_k)
            
            for k in range(1, min(max_k + 1, n + 1)):
                # Approximate: P(K=k) ∝ α^k * Γ(k) * S(n,k) / Γ(n+α)
                # Simplified approximation for small k
                if k == 1:
                    log_prob = np.log(alpha)
                else:
                    log_prob = k * np.log(alpha) + gammaln(k) - 0.5 * k * np.log(2 * np.pi * k)
                
                # Add Stirling number approximation
                if k < n:
                    log_prob += gammaln(n) - gammaln(n - k + 1) - (k - 1) * np.log(k)
                
                probs[k-1] = np.exp(min(log_prob, 700))  # Prevent overflow
            
            # Normalize
            total = np.sum(probs)
            if total > 0:
                probs = probs / total
            
            return probs
        
        # Calculate prior
        crp_prior = crp_prior_probs(n_assessors, alpha, max_clusters)
        
        # Create visualization
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        # Left panel: Trace plot
        ax1.plot(cluster_counts, color='steelblue', alpha=0.7, linewidth=0.8)
        ax1.set_xlabel('Iteration (post burn-in)')
        ax1.set_ylabel('Number of Clusters')
        ax1.set_title('Cluster Count Trace')
        ax1.grid(True, alpha=0.3)
        
        # Right panel: Distribution comparison
        x_pos = np.arange(1, max_clusters + 1)
        width = 0.35
        
        # Posterior bars
        bars1 = ax2.bar(x_pos - width/2, posterior, width, 
                       label='Posterior', alpha=0.8, color='steelblue')
        
        # Prior bars
        bars2 = ax2.bar(x_pos + width/2, crp_prior, width,
                       label='CRP Prior', alpha=0.8, color='orange')
        
        # Add sample mean line
        sample_mean = np.mean(cluster_counts)
        ax2.axvline(sample_mean, color='red', linestyle='--', linewidth=2, 
                   label=f'Sample Mean ({sample_mean:.1f})')
        
        ax2.set_xlabel('Number of Clusters (K)')
        ax2.set_ylabel('Probability')
        ax2.set_title('Posterior vs Prior Distribution')
        ax2.set_xticks(x_pos)
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
        # Summary statistics
        posterior_mean = sum(k * p for k, p in zip(k_range, posterior))
        prior_mean = sum(k * p for k, p in zip(k_range, crp_prior))
        
        print(f"\n📈 Summary Statistics:")
        print(f"   Sample mean K: {sample_mean:.2f}")
        print(f"   Posterior mean K: {posterior_mean:.2f}")
        print(f"   Prior mean K: {prior_mean:.2f}")
        
        # Show top posterior probabilities
        print(f"\n   Top posterior probabilities:")
        sorted_post = sorted(zip(k_range, posterior), key=lambda x: x[1], reverse=True)
        for k, prob in sorted_post[:5]:
            if prob > 0.001:
                count = count_dist.get(k, 0)
                print(f"     K={k}: {prob:.3f} ({count:,} samples)")
        
        return {
            'cluster_counts': cluster_counts,
            'k_range': k_range,
            'posterior': posterior,
            'crp_prior': crp_prior,
            'sample_mean': sample_mean,
            'posterior_mean': posterior_mean,
            'prior_mean': prior_mean,
            'n_assessors': n_assessors,
            'alpha': alpha
        }
    @staticmethod
    def visualize_generated_data(
        data: dict,
        title_prefix: str = "Generated HPO Data",
        max_po_plots: int = 3,
        save_dir: str = None,
        show: bool = True,
    ):
        """
        Visualize key aspects of the dataset produced by GenerationUtils.generate_assessor_cluster_data.

        Parameters
        ----------
        data : dict
            Output dict from generate_assessor_cluster_data(...).
            Expected keys (gracefully handled if missing):
            - 'assessor_data' : Dict[int, List[List[int]]]
            - 'assessor_choice_sets' : Dict[int, List[List[int]]]
            - 'assessor_to_cluster' : Dict[int, int]
            - 'cluster_to_assessors' : Dict[int, List[int]]
            - 'cluster_ids' : List[int]
            - 'M_a_dict' : Dict[int, List[int]]
            - 'h_U_true' : Dict[int, np.ndarray]   (adjacency / partial order matrix per cluster)
            - 'true_params' : Dict[str, Any] (optional: contains alpha/beta)
            - 'M0' : List[int]
        title_prefix : str
            Prefix for figure titles.
        max_po_plots : int
            Max number of clusters for which to plot the partial order matrix (heatmap-like).
        save_dir : str or None
            If provided, save each figure as a PNG inside this directory.
        show : bool
            Whether to call plt.show() for each figure.
        """
        import os
        import math
        import numpy as np
        import matplotlib.pyplot as plt

        def _maybe_save(fig, name: str):
            if save_dir:
                os.makedirs(save_dir, exist_ok=True)
                fig.savefig(os.path.join(save_dir, f"{name}.png"), dpi=150, bbox_inches="tight")

        # ---------- Extract pieces with fallbacks ----------
        assessor_data         = data.get("assessor_data", {})
        assessor_choice_sets  = data.get("assessor_choice_sets", {})
        assessor_to_cluster   = data.get("assessor_to_cluster", {})
        cluster_to_assessors  = data.get("cluster_to_assessors", {})
        cluster_ids           = data.get("cluster_ids", sorted(cluster_to_assessors.keys()))
        M_a_dict              = data.get("M_a_dict", {})
        h_U_true              = data.get("h_U_true", {})
        true_params           = data.get("true_params", {})
        M0                    = data.get("M0", [])
        alpha                 = true_params.get("alpha", None)
        beta                  = true_params.get("beta", None)

        # ---------- Derived stats ----------
        # cluster sizes (assessor-level)
        cluster_sizes = {k: len(cluster_to_assessors.get(k, [])) for k in cluster_ids}

        # lists per assessor
        lists_per_assessor = [len(assessor_data.get(a, [])) for a in sorted(assessor_data.keys())]

        # choice set sizes across all lists
        choice_sizes = []
        for a, csets in assessor_choice_sets.items():
            for cs in csets:
                choice_sizes.append(len(cs))

        # items per cluster
        items_per_cluster = {k: len(M_a_dict.get(k, [])) for k in cluster_ids}
        # (5) build list-level true cluster labels c_true so we can do np.sum(c_true==t)
        #     For each assessor, append one label per list they contributed.
        c_true_listlevel = []
        for a, orders in assessor_data.items():
            k = assessor_to_cluster.get(a, None)
            if k is None:
                # fallback: if we have observed_orders_cl, we'll use that later for counts
                continue
            c_true_listlevel.extend([k] * len(orders))
        c_true = np.array(c_true_listlevel, dtype=int) if c_true_listlevel else np.array([], dtype=int)
        observed_orders_cl    = data.get("observed_orders_cl", {})     # k -> list of orders (optional)

        # ---------- 0) Console summary ----------
        print("=== Data Summary ===")
        print(f"Assessors: {len(assessor_data)}")
        print(f"Clusters : {len(cluster_ids)} -> sizes {cluster_sizes}")
        total_lists = sum(lists_per_assessor) if lists_per_assessor else 0
        print(f"Total lists: {total_lists}")
        if choice_sizes:
            print(f"Choice set size: mean={np.mean(choice_sizes):.2f}, "
                f"min={np.min(choice_sizes)}, max={np.max(choice_sizes)}")
        if alpha is not None:
            print(f"alpha (len={len(alpha)}), mean={np.mean(alpha):.3f}, std={np.std(alpha):.3f}")
        if beta is not None:
            print(f"beta  (len={len(beta)}), mean={np.mean(beta):.3f}, std={np.std(beta):.3f}")

        # ---------- 1) Bar: cluster sizes ----------
        if cluster_sizes:
            fig = plt.figure()
            xs = list(cluster_sizes.keys())
            ys = [cluster_sizes[k] for k in xs]
            plt.bar([str(x) for x in xs], ys)
            plt.title(f"{title_prefix}: Cluster sizes (#assessors)")
            plt.xlabel("Cluster ID")
            plt.ylabel("# Assessors")
            _maybe_save(fig, "cluster_sizes")
            if show: plt.show()

        # ---------- 2) Histogram: lists per assessor ----------
        if lists_per_assessor:
            fig = plt.figure()
            plt.hist(lists_per_assessor, bins=min(20, max(5, int(math.sqrt(len(lists_per_assessor))))))
            plt.title(f"{title_prefix}: Lists per assessor")
            plt.xlabel("# Lists for an assessor")
            plt.ylabel("Count of assessors")
            _maybe_save(fig, "lists_per_assessor")
            if show: plt.show()

        # ---------- 3) Histogram: choice set sizes ----------
        if choice_sizes:
            fig = plt.figure()
            plt.hist(choice_sizes, bins=range(int(min(choice_sizes)), int(max(choice_sizes))+2))
            plt.title(f"{title_prefix}: Choice set size distribution")
            plt.xlabel("Choice set size")
            plt.ylabel("Count of lists")
            _maybe_save(fig, "choice_set_sizes")
            if show: plt.show()

        # ---------- 4) Bar: items per cluster ----------
        if items_per_cluster:
            fig = plt.figure()
            xs = list(items_per_cluster.keys())
            ys = [items_per_cluster[k] for k in xs]
            plt.bar([str(x) for x in xs], ys)
            plt.title(f"{title_prefix}: Items per cluster")
            plt.xlabel("Cluster ID")
            plt.ylabel("# Items in cluster")
            _maybe_save(fig, "items_per_cluster")
            if show: plt.show()

        # ---------- 5) Heatmaps: partial order matrices (up to max_po_plots) ----------
        # ---------- 5) Per-cluster partial orders using PO_plot (no heatmap) ----------
        if h_U_true and max_po_plots > 0:
            plotted = 0
            for t in sorted(h_U_true.keys()):
                Ht = h_U_true[t]
                if not hasattr(Ht, "shape") or Ht.size == 0:
                    continue
                items_t = M_a_dict.get(t, list(range(Ht.shape[0])))

                # lists count for this cluster (prefer list-level c_true; fallback to observed_orders_cl)
                if c_true.size:
                    lists_count_t = int(np.sum(c_true == t))
                else:
                    lists_count_t = len(observed_orders_cl.get(t, [])) if observed_orders_cl else 0

                # >>> This is the line you asked to integrate <<<
                PO_plot.visualize_partial_order(
                    Ht, items_t,
                    title=f"True Cluster {t}  (lists={lists_count_t}, |items|={len(items_t)})"
                )

                plotted += 1
                if plotted >= max_po_plots:
                    break


        # ---------- 6) Optional: alpha per item (bar) ----------
        if alpha is not None and len(alpha) == len(M0) and len(M0) > 0:
            fig = plt.figure()
            plt.bar([str(i) for i in M0], alpha)
            plt.title(f"{title_prefix}: alpha per item")
            plt.xlabel("Global item ID")
            plt.ylabel("alpha")
            _maybe_save(fig, "alpha_per_item")
            if show: plt.show()

        print("✅ Visualization complete.")

    @staticmethod
    def plot_cluster_diagnostics(mcmc_result, config=None, data=None, burn_in=10000, max_clusters=10, figsize=(12, 4)):
        """
        Plot MCMC diagnostics for the number of clusters:
        1. Trace of the number of unique clusters over time.
        2. Distribution (histogram) of the number of unique clusters with CRP prior overlay.
        
        Parameters
        ----------
        mcmc_result : dict
            MCMC results dictionary containing 'assessor_cluster_trace'
        config : dict, optional
            Configuration dictionary with CRP parameters
        data : dict, optional
            Data dictionary (fallback for parameters)
        burn_in : int
            Number of burn-in iterations to discard
        max_clusters : int
            Maximum number of clusters to plot
        figsize : tuple
            Figure size (width, height)
        
        Returns
        -------
        matplotlib.figure.Figure
            The plot figure
        """
        import matplotlib.pyplot as plt
        import numpy as np
        from collections import Counter
        from scipy.special import gammaln
        import warnings
        
        # Get assessor cluster trace after burn-in
        assessor_cluster_trace = mcmc_result.get('assessor_cluster_trace', [])
        
        if not assessor_cluster_trace:
            print("❗ No assessor_cluster_trace found")
            return None
        
        # Apply burn-in
        trace_filtered = assessor_cluster_trace[burn_in:]
        print(f"   Using {len(trace_filtered)} iterations after burn-in")
        
        # Count number of clusters in each iteration
        cluster_counts = []
        for iteration_clusters in trace_filtered:
            if hasattr(iteration_clusters, 'values'):
                unique_clusters = len(set(iteration_clusters.values()))
                cluster_counts.append(unique_clusters)
        
        if not cluster_counts:
            print("❗ No cluster counts could be extracted")
            return None
            
        print(f"   Cluster count range: {min(cluster_counts)} to {max(cluster_counts)}")
        
        # Get CRP parameters
        n_assessors = None
        concentration_theta = None
        discount_d = 0.0  # Default for pure CRP
        
        # Try to get number of assessors from various sources
        if config is not None:
            if 'assessors' in config:
                n_assessors = len(config['assessors'])
            elif 'data' in config and 'assessors' in config['data']:
                n_assessors = len(config['data']['assessors'])
            
            # Get CRP parameters from config
            if 'prior' in config:
                concentration_theta = config['prior'].get('Dri_alpha', config['prior'].get('concentration_theta', None))
                discount_d = config['prior'].get('Dri_theta', config['prior'].get('discount_d', 0.0))
        
        # Fallback to data dictionary
        if data is not None:
            if n_assessors is None:
                n_assessors = len(data.get('assessors', []))
            if concentration_theta is None:
                true_params = data.get('true_params', {})
                concentration_theta = true_params.get('concentration_theta', true_params.get('Dri_alpha', None))
                discount_d = true_params.get('discount_d', true_params.get('Dri_theta', 0.0))
        
        # Final fallback: infer from MCMC trace
        if n_assessors is None:
            if trace_filtered:
                n_assessors = len(set(trace_filtered[0].keys()))
        
        # Default concentration parameter if still None
        if concentration_theta is None:
            concentration_theta = 1.0  # Default CRP concentration
            print(f"   Using default concentration parameter: {concentration_theta}")
        
        if n_assessors is None:
            print("❗ Could not determine number of assessors")
            return None
        
        print(f"   CRP parameters: n={n_assessors}, θ={concentration_theta}, d={discount_d}")
        
        # Calculate CRP prior probabilities
        def calculate_crp_prior(n, theta, d, max_k):
            """Calculate CRP prior probabilities for number of clusters."""
            priors = np.zeros(max_k)
            
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                
                for k in range(1, min(max_k + 1, n + 1)):
                    try:
                        if d == 0:  # Pure CRP (Dirichlet Process)
                            log_prob = k * np.log(max(theta, 1e-10))
                            if k == 1:
                                log_prob += 0  # S(n,1) = 1
                            else:
                                log_prob += gammaln(n) - gammaln(k)
                            priors[k-1] = np.exp(min(log_prob, 700))
                        else:  # Pitman-Yor process
                            log_prob = 0
                            for j in range(k):
                                log_prob += np.log(max(theta + j * d, 1e-10))
                            log_prob += gammaln(n) - gammaln(k)
                            priors[k-1] = np.exp(min(log_prob, 700))
                    except (ValueError, OverflowError):
                        priors[k-1] = 0
            
            # Normalize
            total = np.sum(priors)
            if total > 0:
                priors = priors / total
            else:
                priors = np.ones(max_k) / max_k
            
            return priors
        
        # Calculate prior probabilities
        crp_priors = calculate_crp_prior(n_assessors, concentration_theta, discount_d, max_clusters)
        
        # Create the plot
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
        fig.suptitle('Sound Data Cluster Diagnostics', fontsize=16, fontweight='bold')
        
        # Left plot: Trace of number of clusters over iterations
        ax1.plot(range(len(cluster_counts)), cluster_counts, color='teal', alpha=0.8, linewidth=0.8)
        ax1.set_xlabel('Iteration')
        ax1.set_ylabel('Number of Clusters')
        ax1.set_title('Trace: Number of Clusters')
        ax1.grid(True, alpha=0.3)
        
        # Add horizontal line for mean
        sample_mean = np.mean(cluster_counts)
        ax1.axhline(sample_mean, color='red', linestyle='--', linewidth=2, 
                    label=f'Mean: {sample_mean:.1f}')
        ax1.legend()
        
        # Right plot: Distribution with prior overlay
        ax2.hist(cluster_counts, bins=range(min(cluster_counts), max(cluster_counts)+2), 
                 density=False, alpha=0.7, color='teal', edgecolor='white', linewidth=0.5)
        
        # Overlay theoretical prior as line
        k_values = list(range(1, max_clusters + 1))
        cluster_range = np.array(k_values)
        prior_scaled = crp_priors * len(cluster_counts)  # Scale to match histogram
        ax2.plot(cluster_range, prior_scaled, 'k--', linewidth=2, label='CRP Prior')
        
        # Add vertical line for mean
        ax2.axvline(sample_mean, color='red', linestyle='--', linewidth=2, 
                    label=f'Sample Mean: {sample_mean:.1f}')
        
        ax2.set_xlabel('Number of Clusters')
        ax2.set_ylabel('Count')
        ax2.set_title('Distribution: Number of Clusters')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
        # Print summary statistics
        print(f"\n📈 Summary Statistics:")
        print(f"   Sample mean clusters: {sample_mean:.2f}")
        print(f"   Min clusters: {min(cluster_counts)}")
        print(f"   Max clusters: {max(cluster_counts)}")
        print(f"   Std clusters: {np.std(cluster_counts):.2f}")
        
        # Calculate and print prior mean
        prior_mean = np.sum([k * p for k, p in zip(k_values, crp_priors)])
        print(f"   Prior mean clusters: {prior_mean:.2f}")
        
        return fig



    @staticmethod
    def plot_effective_list_cluster_diagnostics(
        mcmc_result: Dict[str, Any],
        config: Optional[Dict[str, Any]] = None,
        burn_in: int = 10_000,
        max_clusters: int = 20,
        figsize: Tuple[int, int] = (12, 4),
        include_h0: bool = False,          # whether to include cluster id 0 (global) in C_eff
        dag_project: bool = True,          # safety: remove cycles before TR
        show_original_c: bool = True,      # overlay original C from c_vec_trace if available
        random_seed: int = 42,
    ):
        """
        Diagnostics for *effective* list clusters derived from H_trace.

        Effective C at draw t:
            C_eff(t) = number of UNIQUE canonical H among all clusters in H_trace[t],
            where canonical(H) = TR( DAG_project(H) ).

        Notes
        -----
        - This is different from c_vec_trace C (assignment-based).
        - OR/thresholding can create cycles; dag_project=True makes it robust.
        """
        import matplotlib.pyplot as plt
        from collections import Counter

        # ---- pull H_trace
        H_trace = mcmc_result.get("H_trace", [])
        if not H_trace:
            print("❗ No 'H_trace' in mcmc_result.")
            return None

        post = H_trace[burn_in:]
        if not post:
            print("❗ No draws remain after burn-in.")
            return None

        # lazy import
        from src.utils.po_fun import BasicUtils

        # infer n
        n = None
        for d in post:
            if d:
                any_key = next(iter(d))
                M = np.asarray(d[any_key])
                if M.ndim == 2 and M.shape[0] == M.shape[1]:
                    n = int(M.shape[0])
                    break
        if n is None:
            print("❗ Could not infer matrix size from H_trace.")
            return None

        def _project_dag(A: np.ndarray) -> np.ndarray:
            """Project adjacency to DAG by score-ordering."""
            A = (A > 0).astype(np.int8)
            np.fill_diagonal(A, 0)
            score = A.sum(axis=1).astype(np.float64) - A.sum(axis=0).astype(np.float64)
            order = np.argsort(-score)
            rank = np.empty(A.shape[0], dtype=int)
            rank[order] = np.arange(A.shape[0])
            A[rank[:, None] >= rank[None, :]] = 0
            np.fill_diagonal(A, 0)
            return A

        def _canonical_signature(M: np.ndarray) -> bytes:
            """Canonicalize matrix then return bytes signature for dedup."""
            A = np.asarray(M)
            if A.shape != (n, n):
                return b""  # will be ignored
            A = (A > 0).astype(np.int8)
            np.fill_diagonal(A, 0)
            if dag_project:
                A = _project_dag(A)

            # transitive reduction (expects DAG; dag_project helps)
            if hasattr(BasicUtils, "trans_mem_safe"):
                Hc = BasicUtils.trans_mem_safe(A.astype(np.int8))
            else:
                Hc = BasicUtils.transitive_reduction(A.astype(int))

            Hc = np.asarray(Hc, dtype=np.uint8)
            return Hc.tobytes()

        # ---- compute C_eff trace
        c_eff_trace: List[int] = []
        for d in post:
            keys = list(d.keys())
            if not include_h0:
                keys = [cid for cid in keys if cid != 0]

            sigs = set()
            for cid in keys:
                sig = _canonical_signature(d[cid])
                if sig:
                    sigs.add(sig)

            c_eff_trace.append(len(sigs))

        if not c_eff_trace:
            print("❗ C_eff trace is empty after processing.")
            return None

        c_min, c_max_obs = int(min(c_eff_trace)), int(max(c_eff_trace))
        mean_eff = float(np.mean(c_eff_trace))
        std_eff = float(np.std(c_eff_trace))

        print(f"Draws used (post burn-in): {len(c_eff_trace)}")
        print(f"Effective C range        : {c_min} .. {c_max_obs}")
        print(f"Effective C mean (std)   : {mean_eff:.2f} ({std_eff:.2f})")
        if not include_h0:
            print("   (H0 excluded from effective C)")

        # ---- optional original C from c_vec_trace
        c_trace = None
        if show_original_c and ("c_vec_trace" in mcmc_result) and mcmc_result["c_vec_trace"]:
            trace0 = mcmc_result["c_vec_trace"][burn_in:]
            if trace0:
                c_trace = [len(set(np.asarray(z).tolist())) for z in trace0]
                print(f"Original C mean (c_vec): {float(np.mean(c_trace)):.2f}")

        # ---- plot
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
        fig.suptitle("Effective List-Cluster Diagnostics (from H_trace)", fontsize=15, fontweight="bold")

        ax1.plot(np.arange(len(c_eff_trace)), c_eff_trace, color="teal", alpha=0.85, lw=0.9, label="C_eff")
        ax1.axhline(mean_eff, color="red", ls="--", lw=2, label=f"Mean C_eff: {mean_eff:.2f}")
        if c_trace is not None:
            ax1.plot(np.arange(len(c_trace)), c_trace, color="black", alpha=0.35, lw=0.8, label="C (c_vec)")

        ax1.set_title("Trace: Effective #Clusters (unique H)")
        ax1.set_xlabel("Draw (post burn-in)")
        ax1.set_ylabel("Number of clusters")
        ax1.grid(True, alpha=0.3)
        ax1.legend(fontsize=8, loc='upper right')

        bins = np.arange(min(c_eff_trace), max(c_eff_trace) + 2)
        ax2.hist(c_eff_trace, bins=bins, color="teal", alpha=0.7, ec="white", lw=0.5)
        ax2.axvline(mean_eff, color="red", ls="--", lw=2, label=f"Mean: {mean_eff:.2f}")

        if c_trace is not None:
            ax2.hist(c_trace, bins=np.arange(min(c_trace), max(c_trace) + 2),
                     color="black", alpha=0.15, ec="white", lw=0.3, label="C (c_vec)")

        ax2.set_title("Distribution: Effective #Clusters")
        ax2.set_xlabel("Number of clusters")
        ax2.set_ylabel("Count")
        ax2.grid(True, alpha=0.3)
        ax2.legend()

        plt.tight_layout()
        plt.show()
        return fig
