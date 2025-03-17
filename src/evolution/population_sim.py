"""
Evolutionary Population Simulator for Evo-CRISPR

This module provides functions for simulating how genetic edits propagate 
through populations over time, accounting for selection, drift, and other 
evolutionary forces.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional, Union, Any, Callable
from dataclasses import dataclass, field
import time
import logging
from scipy.stats import norm
import networkx as nx

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Constants
DEFAULT_POPULATION_SIZE = 1000
DEFAULT_GENERATIONS = 100
DEFAULT_MUTATION_RATE = 1e-8
DEFAULT_RECOMBINATION_RATE = 1e-8
DEFAULT_MIGRATION_RATE = 0.01

@dataclass
class Individual:
    """Class representing an individual in the simulation"""
    genotype: List[int]  # List of alleles (0=wild-type, 1=edited, 2=other variant)
    fitness: float = 1.0
    sex: str = "F"      # M or F
    age: int = 0
    location: int = 0   # For spatial models
    
    def __post_init__(self):
        """Calculate fitness based on genotype if not provided"""
        if self.fitness == 1.0:
            # Default calculation - will be overridden by fitness functions
            pass

@dataclass
class PopulationParams:
    """Parameters for population simulation"""
    size: int = DEFAULT_POPULATION_SIZE
    carrying_capacity: int = DEFAULT_POPULATION_SIZE
    initial_edited_frequency: float = 0.0
    sex_ratio: float = 0.5  # Proportion of males
    num_loci: int = 1       # Number of genetic loci to track
    spatial_structure: bool = False
    num_demes: int = 1      # Number of subpopulations if spatial
    
@dataclass
class SimulationParams:
    """Parameters for evolution simulation"""
    generations: int = DEFAULT_GENERATIONS
    mutation_rate: float = DEFAULT_MUTATION_RATE
    recombination_rate: float = DEFAULT_RECOMBINATION_RATE
    migration_rate: float = DEFAULT_MIGRATION_RATE
    selection_model: str = "multiplicative"  # additive, multiplicative, epistatic
    fitness_landscape: Dict[str, float] = field(default_factory=dict)
    stochastic: bool = True  # Whether to include random genetic drift
    record_frequency: int = 1  # Record data every N generations

class PopulationSimulator:
    """Class for evolutionary population simulations"""
    
    def __init__(
        self, 
        pop_params: PopulationParams, 
        sim_params: SimulationParams
    ):
        """Initialize the population simulator"""
        self.pop_params = pop_params
        self.sim_params = sim_params
        self.population = []
        self.history = {
            'generation': [],
            'pop_size': [],
            'edited_frequency': [],
            'mean_fitness': [],
            'allele_frequencies': []
        }
        self._initialize_population()
        
    def _initialize_population(self):
        """Create the initial population"""
        # Determine how many individuals have the edited allele
        num_edited = int(self.pop_params.size * self.pop_params.initial_edited_frequency)
        
        # Create individuals
        self.population = []
        for i in range(self.pop_params.size):
            # Determine genotype (for haploid model for simplicity)
            genotype = [0] * self.pop_params.num_loci  # All wild-type
            
            # For first locus, assign edited allele to some individuals
            if i < num_edited:
                genotype[0] = 1  # Edited allele at first locus
            
            # Assign sex
            sex = "M" if np.random.random() < self.pop_params.sex_ratio else "F"
            
            # Assign location if using spatial structure
            location = np.random.randint(0, self.pop_params.num_demes) if self.pop_params.spatial_structure else 0
            
            # Create individual
            individual = Individual(
                genotype=genotype,
                sex=sex,
                location=location
            )
            
            # Calculate fitness based on genotype
            individual.fitness = self._calculate_fitness(individual)
            
            self.population.append(individual)
        
        # Record initial state
        self._record_state(0)
    
    def _calculate_fitness(self, individual: Individual) -> float:
        """Calculate the fitness of an individual based on its genotype"""
        # Get genotype as a string for lookup in fitness landscape
        genotype_str = ''.join(map(str, individual.genotype))
        
        # Check if this specific genotype has a defined fitness value
        if genotype_str in self.sim_params.fitness_landscape:
            return self.sim_params.fitness_landscape[genotype_str]
        
        # Otherwise calculate based on selection model
        fitness = 1.0
        
        if self.sim_params.selection_model == "multiplicative":
            # Multiplicative fitness effects
            for locus, allele in enumerate(individual.genotype):
                allele_key = f"{locus}_{allele}"
                if allele_key in self.sim_params.fitness_landscape:
                    fitness *= self.sim_params.fitness_landscape[allele_key]
                    
        elif self.sim_params.selection_model == "additive":
            # Additive fitness effects
            fitness = 1.0
            for locus, allele in enumerate(individual.genotype):
                allele_key = f"{locus}_{allele}"
                if allele_key in self.sim_params.fitness_landscape:
                    fitness += self.sim_params.fitness_landscape[allele_key] - 1.0
        
        # Ensure fitness is not negative
        return max(0.0, fitness)
    
    def _select_parents(self) -> Tuple[Individual, Individual]:
        """Select two parents based on fitness"""
        # Calculate selection probabilities based on fitness
        fitnesses = np.array([ind.fitness for ind in self.population])
        prob = fitnesses / np.sum(fitnesses)
        
        # Select parents
        parent_indices = np.random.choice(
            len(self.population), 
            size=2, 
            replace=True,  # Allow self-mating for simplicity
            p=prob
        )
        
        return self.population[parent_indices[0]], self.population[parent_indices[1]]
    
    def _reproduce(self, parent1: Individual, parent2: Individual) -> Individual:
        """Create a new individual from two parents"""
        # Determine offspring genotype through recombination and mutation
        offspring_genotype = []
        
        for locus in range(self.pop_params.num_loci):
            # Randomly select allele from one parent
            allele = parent1.genotype[locus] if np.random.random() < 0.5 else parent2.genotype[locus]
            
            # Possible mutation
            if np.random.random() < self.sim_params.mutation_rate:
                # Simple model: mutate to a random different allele
                current_options = list(range(3))  # 0, 1, 2
                current_options.remove(allele)
                allele = np.random.choice(current_options)
            
            offspring_genotype.append(allele)
        
        # Determine sex
        offspring_sex = "M" if np.random.random() < self.pop_params.sex_ratio else "F"
        
        # Determine location (default to parent1's location or allow migration)
        offspring_location = parent1.location
        if self.pop_params.spatial_structure and np.random.random() < self.sim_params.migration_rate:
            # Migrate to a different deme
            options = list(range(self.pop_params.num_demes))
            if len(options) > 1:  # Only if there are multiple demes
                options.remove(offspring_location)
                offspring_location = np.random.choice(options)
        
        # Create new individual
        offspring = Individual(
            genotype=offspring_genotype,
            sex=offspring_sex,
            location=offspring_location
        )
        
        # Calculate fitness
        offspring.fitness = self._calculate_fitness(offspring)
        
        return offspring
    
    def _record_state(self, generation: int):
        """Record the current state of the population"""
        if generation % self.sim_params.record_frequency != 0:
            return
            
        # Calculate allele frequencies
        all_allele_freqs = []
        for locus in range(self.pop_params.num_loci):
            allele_counts = {}
            for ind in self.population:
                allele = ind.genotype[locus]
                if allele not in allele_counts:
                    allele_counts[allele] = 0
                allele_counts[allele] += 1
            
            # Convert to frequencies
            total = len(self.population)
            allele_freqs = {allele: count/total for allele, count in allele_counts.items()}
            all_allele_freqs.append(allele_freqs)
        
        # Calculate mean fitness
        mean_fitness = np.mean([ind.fitness for ind in self.population])
        
        # Calculate edited allele frequency (for locus 0)
        edited_freq = 0
        if self.population:
            edited_count = sum(1 for ind in self.population if ind.genotype[0] == 1)
            edited_freq = edited_count / len(self.population)
        
        # Record data
        self.history['generation'].append(generation)
        self.history['pop_size'].append(len(self.population))
        self.history['edited_frequency'].append(edited_freq)
        self.history['mean_fitness'].append(mean_fitness)
        self.history['allele_frequencies'].append(all_allele_freqs)

    def run_simulation(self, callback: Optional[Callable] = None) -> Dict[str, Any]:
        """
        Run the evolutionary simulation for the specified number of generations.
        
        Args:
            callback: Optional callback function called after each generation
            
        Returns:
            Dictionary with simulation results
        """
        for generation in range(1, self.sim_params.generations + 1):
            # Create new generation
            new_population = []
            
            # Determine number of offspring based on carrying capacity
            target_size = self.pop_params.carrying_capacity
            
            # Produce offspring until we reach target size
            while len(new_population) < target_size:
                # Select parents
                parent1, parent2 = self._select_parents()
                
                # Create offspring
                offspring = self._reproduce(parent1, parent2)
                
                # Add to new population
                new_population.append(offspring)
            
            # Replace old population with new generation
            self.population = new_population
            
            # Record the state
            self._record_state(generation)
            
            # Call callback if provided
            if callback:
                callback(self, generation)
            
            # Log progress periodically
            if generation % 10 == 0 or generation == self.sim_params.generations:
                logger.info(f"Generation {generation}/{self.sim_params.generations} complete")
        
        # Return results
        return self.get_results()
    
    def get_results(self) -> Dict[str, Any]:
        """Get the results of the simulation"""
        return {
            'parameters': {
                'population': self.pop_params,
                'simulation': self.sim_params
            },
            'history': self.history,
            'final_state': {
                'pop_size': len(self.population),
                'edited_frequency': self.history['edited_frequency'][-1] if self.history['edited_frequency'] else 0,
                'mean_fitness': self.history['mean_fitness'][-1] if self.history['mean_fitness'] else 0
            }
        }
    
    def plot_allele_frequency(self, locus: int = 0, allele: int = 1) -> plt.Figure:
        """
        Plot the frequency of a specific allele over time.
        
        Args:
            locus: Locus index to plot
            allele: Allele value to plot (0=wild-type, 1=edited)
            
        Returns:
            Matplotlib figure
        """
        generations = self.history['generation']
        
        # Extract frequencies for the specified allele at the specified locus
        frequencies = []
        for gen_idx, gen in enumerate(generations):
            allele_freqs = self.history['allele_frequencies'][gen_idx][locus]
            frequencies.append(allele_freqs.get(allele, 0))
        
        # Create plot
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(generations, frequencies, '-o', label=f'Allele {allele}')
        ax.set_xlabel('Generation')
        ax.set_ylabel('Allele Frequency')
        ax.set_title(f'Frequency of Allele {allele} at Locus {locus}')
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 1)
        ax.legend()
        
        return fig
    
    def plot_fitness_trajectory(self) -> plt.Figure:
        """
        Plot the mean fitness of the population over time.
        
        Returns:
            Matplotlib figure
        """
        generations = self.history['generation']
        mean_fitness = self.history['mean_fitness']
        
        # Create plot
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(generations, mean_fitness, '-o')
        ax.set_xlabel('Generation')
        ax.set_ylabel('Mean Fitness')
        ax.set_title('Population Mean Fitness Over Time')
        ax.grid(True, alpha=0.3)
        
        return fig


def create_gene_drive_simulation(
    population_size: int = DEFAULT_POPULATION_SIZE,
    initial_frequency: float = 0.01,
    drive_efficiency: float = 0.9,
    fitness_cost: float = 0.05,
    resistance_rate: float = 0.01,
    generations: int = DEFAULT_GENERATIONS
) -> Dict[str, Any]:
    """
    Create a simulation specifically modeling a CRISPR gene drive.
    
    Args:
        population_size: Size of the population
        initial_frequency: Initial frequency of the gene drive
        drive_efficiency: Efficiency of the gene drive (conversion rate)
        fitness_cost: Fitness cost of the gene drive
        resistance_rate: Rate at which resistance develops
        generations: Number of generations to simulate
        
    Returns:
        Dictionary with simulation results
    """
    # Create population parameters
    pop_params = PopulationParams(
        size=population_size,
        carrying_capacity=population_size,
        initial_edited_frequency=initial_frequency,
        num_loci=2  # Locus 0: gene drive, Locus 1: resistance
    )
    
    # Create fitness landscape
    # 0: wild-type, 1: gene drive, 2: resistant
    fitness_landscape = {
        # First locus fitness effects (gene drive)
        "0_0": 1.0,     # Wild-type
        "0_1": 1.0 - fitness_cost,  # Gene drive with fitness cost
        "0_2": 1.0,     # Resistant allele
        
        # Second locus fitness effects (no effect in this model)
        "1_0": 1.0,
        "1_1": 1.0,
        "1_2": 1.0
    }
    
    # Custom reproduction function to model gene drive
    def gene_drive_reproduce(simulator, parent1, parent2):
        # Original reproduce method
        offspring = simulator._reproduce(parent1, parent2)
        
        # Apply gene drive mechanism
        # If one parent has the gene drive and the other doesn't,
        # the drive can convert the wild-type allele to drive
        p1_has_drive = parent1.genotype[0] == 1
        p2_has_drive = parent2.genotype[0] == 1
        
        if (p1_has_drive or p2_has_drive) and offspring.genotype[0] == 0:
            # Gene drive can convert wild-type to drive with certain efficiency
            if np.random.random() < drive_efficiency:
                # Check for resistance development
                if np.random.random() < resistance_rate:
                    offspring.genotype[0] = 2  # Resistant allele
                else:
                    offspring.genotype[0] = 1  # Converted to drive
        
        # Recalculate fitness
        offspring.fitness = simulator._calculate_fitness(offspring)
        return offspring
    
    # Create simulation parameters
    sim_params = SimulationParams(
        generations=generations,
        selection_model="multiplicative",
        fitness_landscape=fitness_landscape,
        record_frequency=1
    )
    
    # Create simulator
    simulator = PopulationSimulator(pop_params, sim_params)
    
    # Replace reproduce method with custom one
    simulator._reproduce = lambda p1, p2: gene_drive_reproduce(simulator, p1, p2)
    
    # Run simulation
    results = simulator.run_simulation()
    
    # Add some gene drive specific analysis
    # Extract frequencies of all alleles at locus 0
    generations = results['history']['generation']
    allele_freqs_by_gen = results['history']['allele_frequencies']
    
    drive_freqs = []
    resistant_freqs = []
    wildtype_freqs = []
    
    for gen_idx, gen in enumerate(generations):
        locus0_freqs = allele_freqs_by_gen[gen_idx][0]  # Alleles at locus 0
        drive_freqs.append(locus0_freqs.get(1, 0))      # Frequency of drive allele
        resistant_freqs.append(locus0_freqs.get(2, 0))  # Frequency of resistant allele
        wildtype_freqs.append(locus0_freqs.get(0, 0))   # Frequency of wild-type allele
    
    # Create allele trajectory plot
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(generations, drive_freqs, '-o', label='Gene Drive')
    ax.plot(generations, resistant_freqs, '-o', label='Resistant')
    ax.plot(generations, wildtype_freqs, '-o', label='Wild-type')
    ax.set_xlabel('Generation')
    ax.set_ylabel('Allele Frequency')
    ax.set_title('Gene Drive Dynamics')
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1)
    ax.legend()
    
    # Convert figure to image
    import io
    import base64
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=100)
    buf.seek(0)
    img_str = base64.b64encode(buf.read()).decode('utf-8')
    
    # Add plot to results
    results['plots'] = {
        'gene_drive_dynamics': img_str
    }
    
    # Add summary metrics
    results['gene_drive_analysis'] = {
        'max_drive_frequency': max(drive_freqs),
        'final_drive_frequency': drive_freqs[-1],
        'final_resistant_frequency': resistant_freqs[-1],
        'fixation_generation': next((i for i, f in enumerate(drive_freqs) if f >= 0.99), None),
        'drive_parameters': {
            'efficiency': drive_efficiency,
            'fitness_cost': fitness_cost,
            'resistance_rate': resistance_rate
        }
    }
    
    return results


def fitness_landscape_from_variants(
    variants: List[Dict[str, Any]],
    consequence_fitness_map: Dict[str, float]
) -> Dict[str, float]:
    """
    Create a fitness landscape based on variant effect predictions.
    
    Args:
        variants: List of variant prediction results
        consequence_fitness_map: Mapping of consequence types to fitness effects
        
    Returns:
        Fitness landscape dictionary for simulation
    """
    fitness_landscape = {}
    
    for i, variant in enumerate(variants):
        # Skip variants without predictions
        if 'effect' not in variant:
            continue
            
        # Map effect prediction to fitness effect
        effect = variant['effect']
        if effect == "Benign":
            fitness_effect = 1.0  # No effect
        elif effect == "Likely Benign":
            fitness_effect = 0.95  # Slight negative effect
        elif effect == "Variant of Uncertain Significance":
            fitness_effect = 0.9  # Moderate negative effect
        elif effect == "Likely Pathogenic":
            fitness_effect = 0.7  # Substantial negative effect
        elif effect == "Pathogenic":
            fitness_effect = 0.5  # Severe negative effect
        else:
            fitness_effect = 0.9  # Default moderate effect
            
        # If confidence is provided, adjust based on confidence
        if 'confidence' in variant:
            confidence = variant['confidence']
            # Move fitness effect toward 1.0 for low confidence
            fitness_effect = 1.0 - (1.0 - fitness_effect) * confidence
            
        # Add to fitness landscape
        allele_key = f"{i}_1"  # locus i, allele 1 (edited)
        fitness_landscape[allele_key] = fitness_effect
        
        # Wild-type allele has fitness 1.0 by default
        fitness_landscape[f"{i}_0"] = 1.0
    
    return fitness_landscape 