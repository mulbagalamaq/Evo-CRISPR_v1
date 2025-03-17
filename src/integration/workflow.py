"""
Integration Workflow Module for GenetiXplorer

This module provides functions to integrate CRISPR design and evolutionary
simulation components, facilitating seamless data flow between modules.
"""

import os
import json
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional, Union, Any
import logging
import base64
import io
from datetime import datetime

# Import from other modules
from src.crispr.crispr_designer import (
    analyze_target_sequence, 
    get_sequence_from_genome,
    GuideRNA,
    export_results_to_csv
)
from src.evolution.population_sim import (
    PopulationParams,
    SimulationParams,
    PopulationSimulator,
    create_gene_drive_simulation,
    fitness_landscape_from_variants
)
from src.variant_predictor.evo_predictor import predict_variant_effect, predict_variant_effect_mock
from src.variant_predictor.ensembl_api import query_ensembl_vep
from src.clinvar.clinvar_api import get_clinvar_data
from src.utils.data_utils import (
    format_variant_identifier,
    validate_variant_input,
    save_analysis_results
)
from src.utils.visualizations import (
    create_delta_score_gauge,
    create_model_comparison_chart,
    create_clinvar_significance_chart
)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class GenetiXplorerWorkflow:
    """Class for integrating CRISPR design and evolutionary simulation"""
    
    def __init__(self, project_name: str = ""):
        """Initialize the workflow manager"""
        self.project_name = project_name or f"GenetiXplorer_Project_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.results_dir = os.path.join("results", self.project_name)
        self.crispr_results = {}
        self.variant_results = {}
        self.simulation_results = {}
        self.integrated_results = {}
        
        # Create results directory
        os.makedirs(self.results_dir, exist_ok=True)
        
    def design_crispr_guides(
        self,
        target_sequence: str = "",
        chromosome: str = "",
        start: int = 0,
        end: int = 0,
        genome_build: str = "GRCh38",
        target_gene: str = "",
        pam: str = "NGG"
    ) -> Dict[str, Any]:
        """
        Design CRISPR guides for a target sequence or genomic region.
        
        Args:
            target_sequence: DNA sequence to target
            chromosome: Chromosome (if no sequence provided)
            start: Start position (if no sequence provided)
            end: End position (if no sequence provided)
            genome_build: Reference genome build
            target_gene: Name of the target gene
            pam: PAM sequence
            
        Returns:
            Dictionary with CRISPR design results
        """
        # Get sequence if not provided
        if not target_sequence and chromosome and start and end:
            target_sequence = get_sequence_from_genome(genome_build, chromosome, start, end)
            logger.info(f"Retrieved {len(target_sequence)} bp sequence from {chromosome}:{start}-{end}")
        
        # Validate sequence
        if not target_sequence:
            return {"error": "No target sequence provided or retrieved"}
        
        # Design guides
        logger.info(f"Designing CRISPR guides for {len(target_sequence)} bp sequence")
        crispr_results = analyze_target_sequence(target_sequence, target_gene, pam)
        
        # Save guide designs to CSV
        guides_file = os.path.join(self.results_dir, "crispr_guides.csv")
        export_results_to_csv(crispr_results, guides_file)
        
        # Add metadata
        crispr_results["meta"] = {
            "chromosome": chromosome,
            "start": start,
            "end": end,
            "genome_build": genome_build,
            "sequence_length": len(target_sequence),
            "pam": pam,
            "guides_file": guides_file
        }
        
        # Store results
        self.crispr_results = crispr_results
        
        return crispr_results
    
    def predict_variant_effects(
        self,
        guides: Optional[List[GuideRNA]] = None,
        variants: Optional[List[Dict[str, Any]]] = None,
        genome_build: str = "GRCh38",
        use_mock: bool = False
    ) -> Dict[str, List[Dict[str, Any]]]:
        """
        Predict effects of variants generated from guide RNAs.
        
        Args:
            guides: List of GuideRNA objects
            variants: List of variant dictionaries (alternative to guides)
            genome_build: Reference genome build
            use_mock: Whether to use mock predictions
            
        Returns:
            Dictionary with variant effect predictions
        """
        # Use guides from previous step if not provided
        if guides is None and variants is None:
            if "guides" in self.crispr_results:
                guides = self.crispr_results["guides"]
            else:
                return {"error": "No guides or variants provided"}
        
        # Convert guides to variants if needed
        if variants is None:
            variants = []
            for guide in guides:
                # For simplicity, assume edits at the PAM site
                chromosome = self.crispr_results.get("meta", {}).get("chromosome", "1")
                position = guide.start_position + len(guide.sequence)
                ref_allele = "A"  # Placeholder - would need sequence context
                alt_allele = "G"  # Placeholder - would be based on edit type
                
                variant = {
                    "chromosome": chromosome,
                    "position": position,
                    "ref_allele": ref_allele,
                    "alt_allele": alt_allele,
                    "guide": guide.sequence
                }
                variants.append(variant)
        
        # Predict effects for each variant
        logger.info(f"Predicting effects for {len(variants)} variants")
        predictions = []
        
        for variant in variants:
            # Get variant info
            chromosome = variant.get("chromosome", "1")
            position = variant.get("position", 0)
            ref_allele = variant.get("ref_allele", "A")
            alt_allele = variant.get("alt_allele", "G")
            
            # Create variant identifier
            variant_id = format_variant_identifier(chromosome, position, ref_allele, alt_allele)
            
            # Get predictions
            predict_func = predict_variant_effect_mock if use_mock else predict_variant_effect
            prediction = predict_func(
                chromosome=chromosome,
                position=position,
                ref_allele=ref_allele,
                alt_allele=alt_allele,
                reference_genome=genome_build
            )
            
            # Add variant identifier
            prediction["variant"] = variant_id
            
            # Get ClinVar data
            try:
                clinvar_data = get_clinvar_data(
                    chromosome=chromosome,
                    position=position,
                    ref_allele=ref_allele,
                    alt_allele=alt_allele,
                    assembly=genome_build
                )
                prediction["clinvar"] = clinvar_data
            except Exception as e:
                logger.error(f"Error fetching ClinVar data: {str(e)}")
                prediction["clinvar"] = {"error": str(e)}
            
            # Get Ensembl VEP data
            try:
                ensembl_data = query_ensembl_vep(
                    chromosome=chromosome,
                    position=position,
                    ref_allele=ref_allele,
                    alt_allele=alt_allele,
                    assembly=genome_build
                )
                prediction["ensembl"] = ensembl_data
            except Exception as e:
                logger.error(f"Error fetching Ensembl data: {str(e)}")
                prediction["ensembl"] = {"error": str(e)}
            
            # Store additional guide info if available
            if "guide" in variant:
                prediction["guide"] = variant["guide"]
            
            predictions.append(prediction)
        
        # Save predictions to file
        predictions_file = os.path.join(self.results_dir, "variant_predictions.json")
        with open(predictions_file, "w") as f:
            json.dump(predictions, f, indent=2)
        
        # Store results
        self.variant_results = {
            "predictions": predictions,
            "meta": {
                "genome_build": genome_build,
                "num_variants": len(predictions),
                "predictions_file": predictions_file
            }
        }
        
        return self.variant_results
    
    def run_evolutionary_simulation(
        self,
        simulation_type: str = "gene_drive",
        population_size: int = 1000,
        initial_frequency: float = 0.01,
        generations: int = 100,
        fitness_landscape: Optional[Dict[str, float]] = None,
        custom_params: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Run evolutionary simulation based on CRISPR edits.
        
        Args:
            simulation_type: Type of simulation ("standard", "gene_drive")
            population_size: Size of the population
            initial_frequency: Initial frequency of the edited allele
            generations: Number of generations to simulate
            fitness_landscape: Custom fitness landscape
            custom_params: Additional parameters for specific simulation types
            
        Returns:
            Dictionary with simulation results
        """
        # Use variant effects to create fitness landscape if not provided
        if fitness_landscape is None and "predictions" in self.variant_results:
            # Create a consequence to fitness mapping
            consequence_fitness_map = {
                "missense_variant": 0.9,
                "synonymous_variant": 1.0,
                "stop_gained": 0.5,
                "frameshift_variant": 0.4,
                "intron_variant": 1.0,
                "downstream_gene_variant": 1.0,
                "upstream_gene_variant": 1.0
            }
            
            fitness_landscape = fitness_landscape_from_variants(
                self.variant_results["predictions"],
                consequence_fitness_map
            )
        
        # Default params if none provided
        if custom_params is None:
            custom_params = {}
        
        # Run appropriate simulation
        logger.info(f"Running {simulation_type} simulation for {generations} generations")
        
        if simulation_type == "gene_drive":
            # Get gene drive specific parameters
            drive_efficiency = custom_params.get("drive_efficiency", 0.9)
            fitness_cost = custom_params.get("fitness_cost", 0.05)
            resistance_rate = custom_params.get("resistance_rate", 0.01)
            
            # Run gene drive simulation
            results = create_gene_drive_simulation(
                population_size=population_size,
                initial_frequency=initial_frequency,
                drive_efficiency=drive_efficiency,
                fitness_cost=fitness_cost,
                resistance_rate=resistance_rate,
                generations=generations
            )
            
        else:  # Standard simulation
            # Create simulation parameters
            pop_params = PopulationParams(
                size=population_size,
                carrying_capacity=population_size,
                initial_edited_frequency=initial_frequency,
                num_loci=custom_params.get("num_loci", 1),
                spatial_structure=custom_params.get("spatial_structure", False),
                num_demes=custom_params.get("num_demes", 1)
            )
            
            sim_params = SimulationParams(
                generations=generations,
                mutation_rate=custom_params.get("mutation_rate", 1e-8),
                selection_model=custom_params.get("selection_model", "multiplicative"),
                fitness_landscape=fitness_landscape,
                stochastic=custom_params.get("stochastic", True)
            )
            
            # Create and run simulator
            simulator = PopulationSimulator(pop_params, sim_params)
            results = simulator.run_simulation()
            
            # Generate plots and convert to base64 strings
            try:
                # Allele frequency plot
                fig = simulator.plot_allele_frequency(0, 1)
                buf = io.BytesIO()
                fig.savefig(buf, format='png', dpi=100)
                buf.seek(0)
                allele_freq_plot = base64.b64encode(buf.read()).decode('utf-8')
                plt.close(fig)
                
                # Fitness trajectory plot
                fig = simulator.plot_fitness_trajectory()
                buf = io.BytesIO()
                fig.savefig(buf, format='png', dpi=100)
                buf.seek(0)
                fitness_plot = base64.b64encode(buf.read()).decode('utf-8')
                plt.close(fig)
                
                # Add plots to results
                results['plots'] = {
                    'allele_frequency': allele_freq_plot,
                    'fitness_trajectory': fitness_plot
                }
            except Exception as e:
                logger.error(f"Error generating plots: {str(e)}")
        
        # Save simulation results
        sim_file = os.path.join(self.results_dir, "simulation_results.json")
        with open(sim_file, "w") as f:
            # Convert some non-serializable objects
            sim_results = results.copy()
            
            # Remove any matplotlib figures and large data
            if 'parameters' in sim_results:
                sim_results['parameters'] = str(sim_results['parameters'])
            
            # Only keep important data for history
            if 'history' in sim_results:
                history = sim_results['history']
                # Keep generation and edited_frequency, summarize the rest
                sim_results['history'] = {
                    'generation': history.get('generation', []),
                    'edited_frequency': history.get('edited_frequency', []),
                    'summary': {
                        'mean_fitness': {
                            'min': min(history.get('mean_fitness', [0])),
                            'max': max(history.get('mean_fitness', [1])),
                            'final': history.get('mean_fitness', [1])[-1] if history.get('mean_fitness') else 1
                        }
                    }
                }
            
            json.dump(sim_results, f, indent=2)
        
        # Store results
        self.simulation_results = results
        
        return results
    
    def integrate_results(self) -> Dict[str, Any]:
        """
        Integrate CRISPR design, variant prediction, and simulation results.
        
        Returns:
            Dictionary with integrated results
        """
        # Check if we have all components
        if not self.crispr_results or not self.variant_results or not self.simulation_results:
            missing = []
            if not self.crispr_results:
                missing.append("CRISPR design")
            if not self.variant_results:
                missing.append("variant prediction")
            if not self.simulation_results:
                missing.append("evolutionary simulation")
            
            return {"error": f"Missing results: {', '.join(missing)}"}
        
        # Create integrated results
        integrated = {
            "project_name": self.project_name,
            "timestamp": datetime.now().isoformat(),
            "summary": {
                "crispr_guides": {
                    "total_guides": len(self.crispr_results.get("guides", [])),
                    "top_guide_sequence": self.crispr_results.get("guides", [{}])[0].get("sequence", "") if self.crispr_results.get("guides") else "",
                    "top_guide_score": self.crispr_results.get("guides", [{}])[0].get("overall_score", 0) if self.crispr_results.get("guides") else 0
                },
                "variant_predictions": {
                    "total_variants": len(self.variant_results.get("predictions", [])),
                    "pathogenic_count": sum(1 for p in self.variant_results.get("predictions", []) if p.get("effect") in ["Pathogenic", "Likely Pathogenic"]),
                    "benign_count": sum(1 for p in self.variant_results.get("predictions", []) if p.get("effect") in ["Benign", "Likely Benign"])
                },
                "simulation": {
                    "final_edited_frequency": self.simulation_results.get("final_state", {}).get("edited_frequency", 0),
                    "generations_simulated": len(self.simulation_results.get("history", {}).get("generation", [])),
                    "fixation_reached": self.simulation_results.get("final_state", {}).get("edited_frequency", 0) > 0.99
                }
            },
            "components": {
                "crispr_design": {
                    "meta": self.crispr_results.get("meta", {}),
                    "top_guides": [g.__dict__ for g in self.crispr_results.get("guides", [])[:3]] if self.crispr_results.get("guides") else []
                },
                "variant_predictions": {
                    "meta": self.variant_results.get("meta", {}),
                    "top_predictions": self.variant_results.get("predictions", [])[:3]
                },
                "simulation": {
                    "final_state": self.simulation_results.get("final_state", {}),
                    "parameters": self.simulation_results.get("parameters", {})
                }
            }
        }
        
        # Add plots if available
        if "plots" in self.simulation_results:
            integrated["plots"] = self.simulation_results["plots"]
        
        # Save integrated results
        integrated_file = os.path.join(self.results_dir, "integrated_results.json")
        with open(integrated_file, "w") as f:
            # Convert non-serializable objects
            # Extract only serializable data from dictionaries
            serializable = {
                "project_name": integrated["project_name"],
                "timestamp": integrated["timestamp"],
                "summary": integrated["summary"]
            }
            json.dump(serializable, f, indent=2)
        
        # Store results
        self.integrated_results = integrated
        
        logger.info(f"Integrated results saved to {integrated_file}")
        
        return integrated
    
    def generate_report(self, format: str = "html") -> str:
        """
        Generate a comprehensive report of the integrated results.
        
        Args:
            format: Output format ("html", "md", "json")
            
        Returns:
            Path to the generated report file
        """
        # Generate report based on integrated results
        if not self.integrated_results:
            logger.warning("No integrated results available, generating from components")
            self.integrate_results()
        
        # Create report filename
        report_file = os.path.join(self.results_dir, f"report.{format}")
        
        if format == "json":
            # For JSON, just save the integrated results
            with open(report_file, "w") as f:
                json.dump(self.integrated_results, f, indent=2)
                
        elif format == "html":
            # Generate HTML report
            html_content = self._generate_html_report()
            with open(report_file, "w") as f:
                f.write(html_content)
                
        elif format == "md":
            # Generate Markdown report
            md_content = self._generate_markdown_report()
            with open(report_file, "w") as f:
                f.write(md_content)
                
        else:
            raise ValueError(f"Unsupported report format: {format}")
        
        logger.info(f"Report generated: {report_file}")
        return report_file
    
    def _generate_html_report(self) -> str:
        """Generate an HTML report of the integrated results"""
        # Simple HTML report template
        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>GenetiXplorer Report: {self.project_name}</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 40px; }}
                h1, h2, h3 {{ color: #2c3e50; }}
                .section {{ margin-bottom: 30px; }}
                .card {{ border: 1px solid #ddd; border-radius: 5px; padding: 20px; margin-bottom: 20px; }}
                .summary {{ display: flex; justify-content: space-between; }}
                .summary-card {{ background-color: #f8f9fa; width: 30%; padding: 15px; border-radius: 5px; }}
                table {{ border-collapse: collapse; width: 100%; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                th {{ background-color: #f2f2f2; }}
                img {{ max-width: 100%; border: 1px solid #ddd; margin: 10px 0; }}
                .footer {{ margin-top: 50px; font-size: 0.8em; color: #7f8c8d; text-align: center; }}
            </style>
        </head>
        <body>
            <h1>GenetiXplorer Integrated Report</h1>
            <p>Project: {self.project_name}</p>
            <p>Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
            
            <div class="section">
                <h2>Executive Summary</h2>
                <div class="summary">
                    <div class="summary-card">
                        <h3>CRISPR Design</h3>
                        <p>Total Guides: {self.integrated_results.get("summary", {}).get("crispr_guides", {}).get("total_guides", 0)}</p>
                        <p>Top Guide: {self.integrated_results.get("summary", {}).get("crispr_guides", {}).get("top_guide_sequence", "")}</p>
                    </div>
                    <div class="summary-card">
                        <h3>Variant Predictions</h3>
                        <p>Total Variants: {self.integrated_results.get("summary", {}).get("variant_predictions", {}).get("total_variants", 0)}</p>
                        <p>Pathogenic: {self.integrated_results.get("summary", {}).get("variant_predictions", {}).get("pathogenic_count", 0)}</p>
                        <p>Benign: {self.integrated_results.get("summary", {}).get("variant_predictions", {}).get("benign_count", 0)}</p>
                    </div>
                    <div class="summary-card">
                        <h3>Evolutionary Outcome</h3>
                        <p>Final Frequency: {self.integrated_results.get("summary", {}).get("simulation", {}).get("final_edited_frequency", 0):.2%}</p>
                        <p>Generations: {self.integrated_results.get("summary", {}).get("simulation", {}).get("generations_simulated", 0)}</p>
                        <p>Fixation: {"Yes" if self.integrated_results.get("summary", {}).get("simulation", {}).get("fixation_reached", False) else "No"}</p>
                    </div>
                </div>
            </div>
            
            <div class="section">
                <h2>CRISPR Guide Design</h2>
                <div class="card">
                    <h3>Top Guides</h3>
                    <table>
                        <tr>
                            <th>Rank</th>
                            <th>Sequence</th>
                            <th>PAM</th>
                            <th>Score</th>
                        </tr>
        """
        
        # Add guide rows
        top_guides = self.integrated_results.get("components", {}).get("crispr_design", {}).get("top_guides", [])
        for i, guide in enumerate(top_guides, 1):
            html += f"""
                        <tr>
                            <td>{i}</td>
                            <td>{guide.get("sequence", "")}</td>
                            <td>{guide.get("pam", "")}</td>
                            <td>{guide.get("overall_score", 0):.2f}</td>
                        </tr>
            """
        
        html += """
                    </table>
                </div>
            </div>
            
            <div class="section">
                <h2>Variant Effect Predictions</h2>
                <div class="card">
                    <h3>Predicted Effects</h3>
                    <table>
                        <tr>
                            <th>Variant</th>
                            <th>Effect</th>
                            <th>Confidence</th>
                            <th>ClinVar</th>
                        </tr>
        """
        
        # Add variant prediction rows
        top_predictions = self.integrated_results.get("components", {}).get("variant_predictions", {}).get("top_predictions", [])
        for pred in top_predictions:
            html += f"""
                        <tr>
                            <td>{pred.get("variant", "")}</td>
                            <td>{pred.get("effect", "")}</td>
                            <td>{pred.get("confidence", 0):.1%}</td>
                            <td>{pred.get("clinvar", {}).get("clinical_significance", "N/A")}</td>
                        </tr>
            """
        
        html += """
                    </table>
                </div>
            </div>
            
            <div class="section">
                <h2>Evolutionary Simulation</h2>
                <div class="card">
        """
        
        # Add simulation plots
        if "plots" in self.integrated_results:
            if "allele_frequency" in self.integrated_results["plots"]:
                html += f"""
                    <h3>Allele Frequency Trajectory</h3>
                    <img src="data:image/png;base64,{self.integrated_results['plots']['allele_frequency']}" alt="Allele Frequency Plot">
                """
            
            if "gene_drive_dynamics" in self.integrated_results["plots"]:
                html += f"""
                    <h3>Gene Drive Dynamics</h3>
                    <img src="data:image/png;base64,{self.integrated_results['plots']['gene_drive_dynamics']}" alt="Gene Drive Dynamics">
                """
            
            if "fitness_trajectory" in self.integrated_results["plots"]:
                html += f"""
                    <h3>Fitness Trajectory</h3>
                    <img src="data:image/png;base64,{self.integrated_results['plots']['fitness_trajectory']}" alt="Fitness Trajectory">
                """
        
        html += """
                </div>
            </div>
            
            <div class="footer">
                <p>Generated by GenetiXplorer platform on """ + datetime.now().strftime('%Y-%m-%d %H:%M:%S') + """</p>
            </div>
        </body>
        </html>
        """
        
        return html
    
    def _generate_markdown_report(self) -> str:
        """Generate a Markdown report of the integrated results"""
        # Simple Markdown report template
        md = f"""# GenetiXplorer Integrated Report

## Project: {self.project_name}
**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Executive Summary

### CRISPR Design
- **Total Guides:** {self.integrated_results.get("summary", {}).get("crispr_guides", {}).get("total_guides", 0)}
- **Top Guide:** {self.integrated_results.get("summary", {}).get("crispr_guides", {}).get("top_guide_sequence", "")}

### Variant Predictions
- **Total Variants:** {self.integrated_results.get("summary", {}).get("variant_predictions", {}).get("total_variants", 0)}
- **Pathogenic:** {self.integrated_results.get("summary", {}).get("variant_predictions", {}).get("pathogenic_count", 0)}
- **Benign:** {self.integrated_results.get("summary", {}).get("variant_predictions", {}).get("benign_count", 0)}

### Evolutionary Outcome
- **Final Frequency:** {self.integrated_results.get("summary", {}).get("simulation", {}).get("final_edited_frequency", 0):.2%}
- **Generations:** {self.integrated_results.get("summary", {}).get("simulation", {}).get("generations_simulated", 0)}
- **Fixation:** {"Yes" if self.integrated_results.get("summary", {}).get("simulation", {}).get("fixation_reached", False) else "No"}

## CRISPR Guide Design

### Top Guides

| Rank | Sequence | PAM | Score |
|------|----------|-----|-------|
"""
        
        # Add guide rows
        top_guides = self.integrated_results.get("components", {}).get("crispr_design", {}).get("top_guides", [])
        for i, guide in enumerate(top_guides, 1):
            md += f"| {i} | {guide.get('sequence', '')} | {guide.get('pam', '')} | {guide.get('overall_score', 0):.2f} |\n"
        
        md += """
## Variant Effect Predictions

### Predicted Effects

| Variant | Effect | Confidence | ClinVar |
|---------|--------|------------|---------|
"""
        
        # Add variant prediction rows
        top_predictions = self.integrated_results.get("components", {}).get("variant_predictions", {}).get("top_predictions", [])
        for pred in top_predictions:
            md += f"| {pred.get('variant', '')} | {pred.get('effect', '')} | {pred.get('confidence', 0):.1%} | {pred.get('clinvar', {}).get('clinical_significance', 'N/A')} |\n"
        
        md += """
## Evolutionary Simulation

"""
        
        # Add simulation results
        final_state = self.integrated_results.get("components", {}).get("simulation", {}).get("final_state", {})
        md += f"""### Final State
- **Population Size:** {final_state.get('pop_size', 0)}
- **Edited Allele Frequency:** {final_state.get('edited_frequency', 0):.2%}
- **Mean Fitness:** {final_state.get('mean_fitness', 0):.2f}

"""
        
        if "gene_drive_analysis" in self.simulation_results:
            drive_params = self.simulation_results["gene_drive_analysis"]["drive_parameters"]
            md += f"""### Gene Drive Parameters
- **Drive Efficiency:** {drive_params.get('efficiency', 0):.1%}
- **Fitness Cost:** {drive_params.get('fitness_cost', 0):.1%}
- **Resistance Rate:** {drive_params.get('resistance_rate', 0):.1%}

### Gene Drive Results
- **Maximum Drive Frequency:** {self.simulation_results["gene_drive_analysis"].get('max_drive_frequency', 0):.1%}
- **Final Drive Frequency:** {self.simulation_results["gene_drive_analysis"].get('final_drive_frequency', 0):.1%}
- **Final Resistant Frequency:** {self.simulation_results["gene_drive_analysis"].get('final_resistant_frequency', 0):.1%}
"""
            if self.simulation_results["gene_drive_analysis"].get('fixation_generation'):
                md += f"- **Fixation Generation:** {self.simulation_results['gene_drive_analysis']['fixation_generation']}\n"
            else:
                md += "- **Fixation Generation:** Not reached\n"
        
        md += "\n\n*Generated by GenetiXplorer platform on " + datetime.now().strftime('%Y-%m-%d %H:%M:%S') + "*"
        
        return md 