"""
Visualization Utilities for Evo-CRISPR

This module provides functions for creating interactive visualizations
and plots to display CRISPR design data, variant predictions, and
evolutionary simulation results.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import altair as alt
from typing import Dict, List, Tuple, Optional, Union, Any
import io
import base64
import json
import os
import logging
from collections import Counter

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Set Altair rendering options
alt.data_transformers.disable_max_rows()

def create_delta_score_gauge(score: float, min_val: float = 0, max_val: float = 1) -> alt.Chart:
    """
    Create a gauge chart showing a score value.
    
    Args:
        score: Score value to display
        min_val: Minimum value for the gauge
        max_val: Maximum value for the gauge
        
    Returns:
        Altair chart object
    """
    # Normalize score
    norm_score = min(max(score, min_val), max_val)
    
    # Create a dataframe with the score
    df = pd.DataFrame({'category': ['score'], 'value': [norm_score]})
    
    # Calculate the angle for the gauge
    def calculate_angle(val):
        return (val - min_val) / (max_val - min_val) * 180
    
    # Create gauge chart
    gauge = alt.Chart(df).mark_arc(
        theta=calculate_angle(norm_score),
        stroke="#aaa",
        strokeWidth=3,
        innerRadius=80,
        outerRadius=100
    ).encode(
        theta=alt.Theta(field='value', scale=alt.Scale(domain=[min_val, max_val], range=[0, 180])),
        color=alt.Color(
            'value:Q',
            scale=alt.Scale(
                domain=[min_val, (min_val + max_val) / 2, max_val],
                range=['red', 'yellow', 'green']
            ),
            legend=None
        )
    )
    
    # Add background to gauge
    background = alt.Chart(pd.DataFrame({'category': ['background'], 'value': [max_val]})).mark_arc(
        theta=180,
        stroke="#ddd",
        strokeWidth=3,
        innerRadius=80,
        outerRadius=100,
        opacity=0.2
    ).encode(
        theta=alt.Theta(field='value', scale=alt.Scale(domain=[min_val, max_val], range=[0, 180]))
    )
    
    # Add text with score value
    text = alt.Chart(df).mark_text(
        align='center',
        baseline='middle',
        fontSize=24,
        fontWeight='bold'
    ).encode(
        text=alt.Text('value:Q', format='.2f')
    )
    
    # Combine all elements
    chart = (background + gauge + text).properties(
        width=200,
        height=120,
        title=f"Score: {norm_score:.2f}"
    ).configure_view(
        strokeWidth=0
    )
    
    return chart

def create_model_comparison_chart(
    predictions: List[Dict[str, Any]], 
    model_names: Optional[List[str]] = None
) -> alt.Chart:
    """
    Create a chart comparing predictions from different models.
    
    Args:
        predictions: List of prediction dictionaries
        model_names: List of model names to include (defaults to all)
        
    Returns:
        Altair chart object
    """
    # Extract model scores from predictions
    scores_data = []
    
    for pred in predictions:
        variant_id = pred.get('variant', 'Unknown')
        
        # Get model scores
        model_scores = pred.get('model_scores', {})
        if not model_scores:
            continue
            
        for model, score in model_scores.items():
            if model_names is None or model in model_names:
                scores_data.append({
                    'variant': variant_id,
                    'model': model,
                    'score': float(score)
                })
    
    if not scores_data:
        logger.warning("No model scores found in predictions")
        return alt.Chart().mark_text(text="No model scores available")
    
    # Create dataframe
    df = pd.DataFrame(scores_data)
    
    # Create chart
    chart = alt.Chart(df).mark_bar().encode(
        x='model:N',
        y='score:Q',
        color='model:N',
        column='variant:N'
    ).properties(
        width=150,
        title="Model Comparison"
    )
    
    return chart

def create_clinvar_significance_chart(clinvar_data: List[Dict[str, Any]]) -> alt.Chart:
    """
    Create a chart showing ClinVar clinical significance counts.
    
    Args:
        clinvar_data: List of ClinVar data dictionaries
        
    Returns:
        Altair chart object
    """
    # Extract clinical significance values
    significance_data = []
    
    for data in clinvar_data:
        if isinstance(data, dict) and 'clinical_significance' in data:
            sig = data['clinical_significance']
            significance_data.append({'significance': sig})
        elif isinstance(data, dict) and 'clinvar' in data and 'clinical_significance' in data['clinvar']:
            sig = data['clinvar']['clinical_significance']
            significance_data.append({'significance': sig})
    
    if not significance_data:
        logger.warning("No ClinVar significance data found")
        return alt.Chart().mark_text(text="No ClinVar data available")
    
    # Create dataframe and count occurrences
    df = pd.DataFrame(significance_data)
    counts = df['significance'].value_counts().reset_index()
    counts.columns = ['significance', 'count']
    
    # Create color mapping for clinical significance
    color_scale = {
        'Pathogenic': 'red',
        'Likely pathogenic': 'orange',
        'Uncertain significance': 'yellow',
        'Likely benign': 'lightgreen',
        'Benign': 'green',
        'Not provided': 'gray',
        'other': 'lightgray'
    }
    
    # Create chart
    chart = alt.Chart(counts).mark_bar().encode(
        x='significance:N',
        y='count:Q',
        color=alt.Color('significance:N', scale=alt.Scale(
            domain=list(color_scale.keys()),
            range=list(color_scale.values())
        ))
    ).properties(
        width=300,
        height=200,
        title="ClinVar Clinical Significance"
    )
    
    return chart

def plot_guide_positions(
    guides: List[Dict[str, Any]], 
    sequence_length: int, 
    target_gene: Optional[str] = None
) -> plt.Figure:
    """
    Plot the positions of guide RNAs along a target sequence.
    
    Args:
        guides: List of guide RNA dictionaries
        sequence_length: Length of the target sequence
        target_gene: Name of the target gene
        
    Returns:
        Matplotlib figure object
    """
    # Extract guide positions and scores
    positions = []
    scores = []
    labels = []
    
    for i, guide in enumerate(guides):
        start = guide.get('start_position', 0)
        end = start + len(guide.get('sequence', ''))
        score = guide.get('overall_score', 0)
        
        positions.append((start, end))
        scores.append(score)
        labels.append(f"Guide {i+1}")
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 4))
    
    # Plot sequence as a line
    ax.plot([0, sequence_length], [0, 0], 'k-', linewidth=2)
    
    # Plot guides as colored blocks
    for i, ((start, end), score) in enumerate(zip(positions, scores)):
        # Color based on score (red to green)
        color = plt.cm.RdYlGn(score)
        
        # Plot guide
        ax.add_patch(plt.Rectangle((start, -0.2), end - start, 0.4, color=color, alpha=0.7))
        
        # Add label
        ax.text(start + (end - start) / 2, 0.5, labels[i], ha='center', va='center', fontsize=8)
    
    # Set labels and title
    ax.set_xlim(0, sequence_length)
    ax.set_ylim(-1, 1)
    ax.set_xlabel('Position in sequence')
    ax.set_yticks([])
    
    title = f"Guide RNA positions"
    if target_gene:
        title += f" for {target_gene}"
    ax.set_title(title)
    
    # Add a colorbar for the scores
    cax = fig.add_axes([0.92, 0.2, 0.03, 0.6])
    sm = plt.cm.ScalarMappable(cmap=plt.cm.RdYlGn, norm=plt.Normalize(0, 1))
    sm.set_array([])
    cbar = plt.colorbar(sm, cax=cax)
    cbar.set_label('Guide score')
    
    plt.tight_layout()
    return fig

def fig_to_base64(fig: plt.Figure) -> str:
    """
    Convert a matplotlib figure to a base64 string.
    
    Args:
        fig: Matplotlib figure
        
    Returns:
        Base64-encoded string of the figure
    """
    # Save figure to a BytesIO object
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=100, bbox_inches='tight')
    buf.seek(0)
    
    # Convert to base64
    img_str = base64.b64encode(buf.read()).decode('utf-8')
    
    # Close figure to prevent memory leaks
    plt.close(fig)
    
    return img_str

def create_evolutionary_trajectory_plot(
    generations: List[int],
    frequencies: List[float],
    resistant_frequencies: Optional[List[float]] = None,
    wild_type_frequencies: Optional[List[float]] = None,
    title: str = "Allele Frequency Trajectory"
) -> plt.Figure:
    """
    Create a plot showing evolutionary trajectories of allele frequencies.
    
    Args:
        generations: List of generation numbers
        frequencies: List of edited allele frequencies
        resistant_frequencies: Optional list of resistant allele frequencies
        wild_type_frequencies: Optional list of wild type frequencies
        title: Plot title
        
    Returns:
        Matplotlib figure object
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot edited allele frequency
    ax.plot(generations, frequencies, 'g-', linewidth=2, label='Edited Allele')
    
    # Plot resistant allele frequency if provided
    if resistant_frequencies is not None:
        ax.plot(generations, resistant_frequencies, 'r-', linewidth=2, label='Resistant Allele')
    
    # Plot wild type frequency if provided
    if wild_type_frequencies is not None:
        ax.plot(generations, wild_type_frequencies, 'k-', linewidth=2, label='Wild Type')
    
    # Set axis labels and title
    ax.set_xlabel('Generation')
    ax.set_ylabel('Allele Frequency')
    ax.set_title(title)
    
    # Set y-axis limits
    ax.set_ylim(0, 1)
    
    # Add grid
    ax.grid(True, linestyle='--', alpha=0.6)
    
    # Add legend
    ax.legend()
    
    plt.tight_layout()
    return fig

def create_fitness_landscape_heatmap(
    fitness_landscape: Dict[str, float],
    allele_names: Optional[List[str]] = None,
    title: str = "Fitness Landscape"
) -> plt.Figure:
    """
    Create a heatmap visualizing a fitness landscape.
    
    Args:
        fitness_landscape: Dictionary mapping genotypes to fitness values
        allele_names: Optional list of allele names
        title: Plot title
        
    Returns:
        Matplotlib figure object
    """
    # Convert fitness landscape to a matrix if needed
    if isinstance(fitness_landscape, dict):
        # Simple case - just a dictionary of values
        values = list(fitness_landscape.values())
        labels = list(fitness_landscape.keys())
        
        if len(labels) <= 10:
            # Create a simple bar chart
            fig, ax = plt.subplots(figsize=(10, 6))
            bars = ax.bar(labels, values, color=plt.cm.viridis(np.array(values)))
            
            ax.set_xlabel('Genotype')
            ax.set_ylabel('Fitness')
            ax.set_title(title)
            
            # Add values on top of bars
            for bar, value in zip(bars, values):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width() / 2, height + 0.01,
                        f'{value:.2f}', ha='center', va='bottom')
            
            return fig
        else:
            # Too many values for a bar chart - create a heatmap
            # Create a square grid size that fits all values
            grid_size = int(np.ceil(np.sqrt(len(values))))
            
            # Create a grid and fill with values
            grid = np.zeros((grid_size, grid_size))
            for i, value in enumerate(values):
                row = i // grid_size
                col = i % grid_size
                grid[row, col] = value
            
            # Create heatmap
            fig, ax = plt.subplots(figsize=(10, 8))
            im = ax.imshow(grid, cmap='viridis')
            
            # Add colorbar
            cbar = fig.colorbar(im, ax=ax)
            cbar.set_label('Fitness')
            
            ax.set_title(title)
            
            return fig
    
    # Case for 2D fitness landscape
    elif isinstance(fitness_landscape, (list, np.ndarray)) and len(fitness_landscape) > 0:
        if isinstance(fitness_landscape[0], (list, np.ndarray)):
            # Convert to numpy array if it's a list of lists
            fitness_array = np.array(fitness_landscape)
            
            # Create labels if not provided
            if allele_names is None:
                allele_names = [f"Allele {i+1}" for i in range(fitness_array.shape[0])]
            
            # Create heatmap
            fig, ax = plt.subplots(figsize=(10, 8))
            im = ax.imshow(fitness_array, cmap='viridis')
            
            # Add labels
            ax.set_xticks(np.arange(fitness_array.shape[1]))
            ax.set_yticks(np.arange(fitness_array.shape[0]))
            ax.set_xticklabels(allele_names)
            ax.set_yticklabels(allele_names)
            
            # Rotate x labels for better readability
            plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
            
            # Add colorbar
            cbar = fig.colorbar(im, ax=ax)
            cbar.set_label('Fitness')
            
            ax.set_title(title)
            
            # Add text annotations
            for i in range(fitness_array.shape[0]):
                for j in range(fitness_array.shape[1]):
                    text = ax.text(j, i, f"{fitness_array[i, j]:.2f}",
                                   ha="center", va="center", color="white" if fitness_array[i, j] < 0.5 else "black")
            
            plt.tight_layout()
            return fig
    
    # Fallback for unrecognized format
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.text(0.5, 0.5, "Cannot visualize fitness landscape", ha='center', va='center')
    ax.set_title(title)
    ax.set_xticks([])
    ax.set_yticks([])
    
    return fig

def visualize_guide_efficiency(guides: List[Dict[str, Any]], top_n: int = 10) -> alt.Chart:
    """
    Create an interactive visualization of guide RNA efficiency scores.
    
    Args:
        guides: List of guide RNA dictionaries
        top_n: Number of top guides to display
        
    Returns:
        Altair chart object
    """
    # Extract data from guides
    guide_data = []
    
    for i, guide in enumerate(guides[:top_n]):
        guide_data.append({
            'id': i + 1,
            'sequence': guide.get('sequence', ''),
            'overall_score': guide.get('overall_score', 0),
            'on_target_score': guide.get('on_target_score', 0),
            'off_target_score': guide.get('off_target_score', 0),
            'gc_content': guide.get('gc_content', 0),
            'position': guide.get('start_position', 0)
        })
    
    if not guide_data:
        return alt.Chart().mark_text(text="No guide data available")
    
    # Create dataframe
    df = pd.DataFrame(guide_data)
    
    # Create chart
    bars = alt.Chart(df).mark_bar().encode(
        x=alt.X('id:O', title='Guide'),
        y=alt.Y('overall_score:Q', title='Overall Score'),
        color=alt.Color('overall_score:Q', scale=alt.Scale(scheme='viridis')),
        tooltip=['id', 'sequence', 'overall_score', 'on_target_score', 'off_target_score', 'gc_content', 'position']
    )
    
    # Add text labels
    text = bars.mark_text(
        align='center',
        baseline='bottom',
        dy=-5
    ).encode(
        text=alt.Text('overall_score:Q', format='.2f')
    )
    
    chart = (bars + text).properties(
        width=400,
        height=300,
        title=f"Top {len(guide_data)} Guide RNA Efficiency Scores"
    )
    
    return chart

def create_allele_frequency_map(
    populations: Dict[str, List[float]],
    generations: List[int]
) -> alt.Chart:
    """
    Create a heatmap showing allele frequencies across populations over time.
    
    Args:
        populations: Dictionary mapping population names to lists of allele frequencies
        generations: List of generation numbers
        
    Returns:
        Altair chart object
    """
    # Create long-form dataframe
    data = []
    
    for pop_name, freqs in populations.items():
        for gen, freq in zip(generations, freqs):
            data.append({
                'population': pop_name,
                'generation': gen,
                'frequency': freq
            })
    
    if not data:
        return alt.Chart().mark_text(text="No population data available")
    
    # Create dataframe
    df = pd.DataFrame(data)
    
    # Create heatmap
    chart = alt.Chart(df).mark_rect().encode(
        x=alt.X('generation:O', title='Generation'),
        y=alt.Y('population:N', title='Population'),
        color=alt.Color('frequency:Q', scale=alt.Scale(scheme='viridis')),
        tooltip=['population', 'generation', 'frequency']
    ).properties(
        width=500,
        height=300,
        title="Allele Frequency Across Populations"
    )
    
    return chart

def save_chart_to_html(chart: alt.Chart, filepath: str) -> str:
    """
    Save an Altair chart to an HTML file.
    
    Args:
        chart: Altair chart object
        filepath: Path to save the HTML file
        
    Returns:
        Path to the saved file
    """
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    
    # Save chart to HTML
    chart.save(filepath)
    
    logger.info(f"Saved chart to {filepath}")
    return filepath

def save_plots_to_report(
    plots: Dict[str, Union[plt.Figure, alt.Chart]],
    output_dir: str,
    base_filename: str = "evo_crispr_plots"
) -> Dict[str, str]:
    """
    Save multiple plots to files for reporting.
    
    Args:
        plots: Dictionary mapping plot names to Matplotlib figures or Altair charts
        output_dir: Directory to save files
        base_filename: Base filename for saved files
        
    Returns:
        Dictionary mapping plot names to file paths
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    saved_files = {}
    
    for name, plot in plots.items():
        # Create a safe filename
        safe_name = name.lower().replace(' ', '_').replace('-', '_')
        
        if isinstance(plot, plt.Figure):
            # Save Matplotlib figure
            filepath = os.path.join(output_dir, f"{base_filename}_{safe_name}.png")
            plot.savefig(filepath, dpi=100, bbox_inches='tight')
            plt.close(plot)
            saved_files[name] = filepath
            
        elif isinstance(plot, alt.Chart):
            # Save Altair chart
            filepath = os.path.join(output_dir, f"{base_filename}_{safe_name}.html")
            save_chart_to_html(plot, filepath)
            saved_files[name] = filepath
            
        else:
            logger.warning(f"Unsupported plot type for '{name}': {type(plot)}")
    
    logger.info(f"Saved {len(saved_files)} plots to {output_dir}")
    return saved_files 