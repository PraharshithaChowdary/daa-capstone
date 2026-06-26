import networkx as nx
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
import logging
from collections import defaultdict
import json

logger = logging.getLogger(__name__)


class GraphVisualizer:
    def __init__(self, config: Dict = None):
        self.config = config or {}
        plt.style.use('seaborn-v0_8-darkgrid')

    def plot_graph(
        self,
        graph: nx.Graph,
        node_colors: Dict[str, float] = None,
        node_sizes: Dict[str, float] = None,
        title: str = "Transaction Graph",
        figsize: Tuple[int, int] = (12, 8),
        save_path: str = None
    ):
        fig, ax = plt.subplots(1, 1, figsize=figsize)
        
        if graph.number_of_nodes() == 0:
            ax.text(0.5, 0.5, 'Empty Graph', ha='center', va='center')
            return fig
        
        pos = nx.spring_layout(graph, k=2, iterations=50, seed=42)
        
        sizes = []
        for node in graph.nodes():
            if node_sizes:
                sizes.append(node_sizes.get(node, 100) * 1000)
            else:
                sizes.append(200)
        
        colors = []
        for node in graph.nodes():
            if node_colors:
                colors.append(node_colors.get(node, 0.5))
            else:
                colors.append(0.5)
        
        nx.draw_networkx_edges(
            graph, pos, alpha=0.3, edge_color='gray',
            width=[d.get('weight', 0.5) / 1000 for _, _, d in graph.edges(data=True)]
        )
        
        nodes = nx.draw_networkx_nodes(
            graph, pos, node_size=sizes,
            node_color=colors, cmap=plt.cm.YlOrRd,
            alpha=0.8, ax=ax
        )
        
        if len(graph.nodes()) <= 50:
            labels = {n: n.split('_')[-1] if len(n) > 8 else n for n in graph.nodes()}
            nx.draw_networkx_labels(graph, pos, labels, font_size=8, ax=ax)
        
        plt.colorbar(nodes, ax=ax, label='Score')
        ax.set_title(title)
        ax.axis('off')
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        
        return fig

    def plot_community_graph(
        self,
        graph: nx.Graph,
        partition: Dict[str, int],
        title: str = "Community Structure",
        figsize: Tuple[int, int] = (14, 10),
        save_path: str = None
    ):
        fig, ax = plt.subplots(1, 1, figsize=figsize)
        
        if not partition:
            ax.text(0.5, 0.5, 'No communities detected', ha='center', va='center')
            return fig
        
        pos = nx.spring_layout(graph, k=3, iterations=100, seed=42)
        
        communities = defaultdict(list)
        for node, comm_id in partition.items():
            communities[comm_id].append(node)
        
        num_communities = len(communities)
        colors = plt.cm.tab20(np.linspace(0, 1, max(num_communities, 1)))
        
        for i, (comm_id, members) in enumerate(communities.items()):
            nx.draw_networkx_nodes(
                graph, pos,
                nodelist=members,
                node_color=[colors[i % len(colors)]],
                node_size=200,
                alpha=0.8,
                ax=ax,
                label=f'C{comm_id} ({len(members)})'
            )
        
        nx.draw_networkx_edges(graph, pos, alpha=0.2, ax=ax)
        
        ax.set_title(title)
        ax.legend(loc='upper right', fontsize=8)
        ax.axis('off')
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        
        return fig

    def plot_interactive_graph(
        self,
        graph: nx.Graph,
        node_scores: Dict[str, float] = None,
        communities: Dict[str, int] = None,
        title: str = "Interactive Transaction Graph"
    ) -> go.Figure:
        if graph.number_of_nodes() == 0:
            return go.Figure()
        
        pos = nx.spring_layout(graph, k=3, iterations=50, seed=42)
        
        edge_trace_x, edge_trace_y = [], []
        for u, v in graph.edges():
            x0, y0 = pos[u]
            x1, y1 = pos[v]
            edge_trace_x.extend([x0, x1, None])
            edge_trace_y.extend([y0, y1, None])
        
        edge_trace = go.Scatter(
            x=edge_trace_x, y=edge_trace_y,
            line=dict(width=0.5, color='#888'),
            hoverinfo='none',
            mode='lines'
        )
        
        node_x, node_y, node_text, node_color, node_size = [], [], [], [], []
        
        for node in graph.nodes():
            x, y = pos[node]
            node_x.append(x)
            node_y.append(y)
            
            score = node_scores.get(node, 0) if node_scores else 0
            comm = communities.get(node, -1) if communities else -1
            deg = graph.degree(node)
            
            node_text.append(
                f"ID: {node}<br>"
                f"Community: {comm}<br>"
                f"Score: {score:.4f}<br>"
                f"Degree: {deg}"
            )
            
            if node_scores:
                node_color.append(node_scores.get(node, 0))
            elif communities:
                node_color.append(communities.get(node, -1))
            else:
                node_color.append(0)
            
            if node_scores:
                node_size.append(max(5, min(50, node_scores.get(node, 0) * 200)))
            else:
                node_size.append(15)
        
        node_trace = go.Scatter(
            x=node_x, y=node_y,
            mode='markers+text' if len(graph.nodes()) <= 30 else 'markers',
            text=[n[:10] for n in graph.nodes()],
            textposition="top center",
            textfont=dict(size=8),
            hoverinfo='text',
            hovertext=node_text,
            marker=dict(
                size=node_size,
                color=node_color,
                colorscale='YlOrRd',
                showscale=True,
                colorbar=dict(
                    title="Score",
                    thickness=15,
                    len=0.5
                ),
                line=dict(width=1, color='black')
            )
        )
        
        fig = go.Figure(
            data=[edge_trace, node_trace],
            layout=go.Layout(
                title=title,
                showlegend=False,
                hovermode='closest',
                margin=dict(b=20, l=5, r=5, t=40),
                xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                plot_bgcolor='rgba(0,0,0,0)'
            )
        )
        
        return fig

    def plot_centrality_distribution(
        self,
        centralities: Dict[str, Dict[str, float]],
        title: str = "Centrality Distributions",
        save_path: str = None
    ) -> go.Figure:
        n_metrics = len(centralities)
        fig = make_subplots(
            rows=n_metrics, cols=1,
            subplot_titles=list(centralities.keys()),
            vertical_spacing=0.05
        )
        
        for i, (metric_name, values) in enumerate(centralities.items()):
            scores = list(values.values())
            fig.add_trace(
                go.Histogram(x=scores, name=metric_name, nbinsx=50),
                row=i+1, col=1
            )
        
        fig.update_layout(
            title=title,
            height=200 * n_metrics,
            showlegend=False
        )
        
        if save_path:
            fig.write_html(save_path)
        
        return fig

    def plot_fraud_network(
        self,
        graph: nx.Graph,
        fraud_nodes: List[str],
        suspicious_edges: List[Tuple[str, str]] = None,
        title: str = "Fraud Network",
        save_path: str = None
    ) -> go.Figure:
        if graph.number_of_nodes() == 0:
            return go.Figure()
        
        pos = nx.spring_layout(graph, k=2, iterations=100, seed=42)
        
        fraud_set = set(fraud_nodes)
        
        edge_traces = []
        
        normal_edges = [(u, v) for u, v in graph.edges()
                       if u not in fraud_set and v not in fraud_set]
        if normal_edges:
            xe, ye = [], []
            for u, v in normal_edges:
                x0, y0 = pos[u]
                x1, y1 = pos[v]
                xe.extend([x0, x1, None])
                ye.extend([y0, y1, None])
            edge_traces.append(go.Scatter(
                x=xe, y=ye, mode='lines',
                line=dict(width=0.5, color='lightgray'),
                hoverinfo='none', showlegend=False
            ))
        
        fraud_related_edges = [(u, v) for u, v in graph.edges()
                              if u in fraud_set or v in fraud_set]
        if fraud_related_edges:
            xe, ye = [], []
            for u, v in fraud_related_edges:
                x0, y0 = pos[u]
                x1, y1 = pos[v]
                xe.extend([x0, x1, None])
                ye.extend([y0, y1, None])
            edge_traces.append(go.Scatter(
                x=xe, y=ye, mode='lines',
                line=dict(width=1.5, color='red'),
                hoverinfo='none', showlegend=False
            ))
        
        node_trace_x, node_trace_y, node_text, node_color, node_size = [], [], [], [], []
        for node in graph.nodes():
            x, y = pos[node]
            node_trace_x.append(x)
            node_trace_y.append(y)
            
            is_fraud = node in fraud_set
            node_text.append(
                f"ID: {node}<br>"
                f"Status: {'FRAUD' if is_fraud else 'Normal'}<br>"
                f"Degree: {graph.degree(node)}"
            )
            node_color.append('red' if is_fraud else 'lightblue')
            node_size.append(25 if is_fraud else 10)
        
        node_trace = go.Scatter(
            x=node_trace_x, y=node_trace_y,
            mode='markers',
            text=[n[:10] for n in graph.nodes()],
            hoverinfo='text',
            hovertext=node_text,
            marker=dict(
                size=node_size,
                color=node_color,
                line=dict(width=1, color='black')
            ),
            showlegend=False
        )
        
        fig = go.Figure(
            data=[*edge_traces, node_trace],
            layout=go.Layout(
                title=title,
                showlegend=False,
                hovermode='closest',
                margin=dict(b=20, l=5, r=5, t=40),
                xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)
            )
        )
        
        if save_path:
            fig.write_html(save_path)
        
        return fig

    def create_dashboard(self, graph: nx.Graph, mining_results: Dict) -> go.Figure:
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                'Network Overview',
                'Community Distribution',
                'Top Features',
                'Risk Scores'
            ),
            specs=[
                [{'type': 'scatter'}, {'type': 'pie'}],
                [{'type': 'bar'}, {'type': 'histogram'}]
            ],
            vertical_spacing=0.12,
            horizontal_spacing=0.1
        )
        
        pos = nx.spring_layout(graph, k=2, iterations=50, seed=42)
        edge_x, edge_y = [], []
        for u, v in graph.edges():
            x0, y0 = pos[u]
            x1, y1 = pos[v]
            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])
        
        fig.add_trace(
            go.Scatter(x=edge_x, y=edge_y, mode='lines', line=dict(width=0.5, color='gray'), hoverinfo='none'),
            row=1, col=1
        )
        
        node_x, node_y = [], []
        for node in graph.nodes():
            x, y = pos[node]
            node_x.append(x)
            node_y.append(y)
        
        fig.add_trace(
            go.Scatter(x=node_x, y=node_y, mode='markers', marker=dict(size=8, color='steelblue', opacity=0.7)),
            row=1, col=1
        )
        
        communities = mining_results.get('communities', {}).get('louvain', {})
        if communities:
            comm_counts = defaultdict(int)
            for c in communities.values():
                if c >= 0:
                    comm_counts[f'C{c}'] += 1
            
            fig.add_trace(
                go.Pie(labels=list(comm_counts.keys()), values=list(comm_counts.values())),
                row=1, col=2
            )
        
        centralities = mining_results.get('centralities', {})
        if 'pagerank' in centralities:
            pr_values = list(centralities['pagerank'].values())
            fig.add_trace(
                go.Histogram(x=pr_values, nbinsx=30, name='PageRank'),
                row=2, col=1
            )
        
        fig.update_layout(
            title="Fraud Detection Dashboard",
            height=800,
            showlegend=False
        )
        
        return fig


def visualize_fraud_patterns(
    graph: nx.Graph,
    fraud_nodes: List[str],
    patterns: List[Dict]
):
    visualizer = GraphVisualizer()
    
    interactive_fig = visualizer.plot_fraud_network(graph, fraud_nodes)
    interactive_fig.show()


def create_interactive_report(
    graph: nx.Graph,
    mining_results: Dict,
    output_path: str = "fraud_report.html"
):
    visualizer = GraphVisualizer()
    
    with open(output_path, 'w') as f:
        f.write("<html><head><title>Fraud Detection Report</title>")
        f.write("<script src='https://cdn.plot.ly/plotly-latest.min.js'></script>")
        f.write("</head><body>")
        
        dashboard = visualizer.create_dashboard(graph, mining_results)
        f.write(dashboard.to_html(full_html=False, include_plotlyjs='cdn'))
        
        centralities = mining_results.get('centralities', {})
        dist_fig = visualizer.plot_centrality_distribution(centralities)
        f.write(dist_fig.to_html(full_html=False, include_plotlyjs='cdn'))
        
        communities = mining_results.get('communities', {}).get('louvain', {})
        comm_fig = visualizer.plot_interactive_graph(graph, communities=communities)
        f.write(comm_fig.to_html(full_html=False, include_plotlyjs='cdn'))
        
        f.write("</body></html>")
    
    logger.info(f"Interactive report saved to {output_path}")
