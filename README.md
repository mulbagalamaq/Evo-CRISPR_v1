# GenetiXplorer

> An integrated platform for CRISPR gene editing design and evolutionary outcome simulation

## Overview

GenetiXplorer is a comprehensive software platform that integrates CRISPR gene editing design tools with evolutionary simulation models. This powerful combination enables researchers to design precise genetic modifications and predict their evolutionary consequences in populations over time.

### Key Capabilities

- **CRISPR Design Module**: Sophisticated tools for designing optimal CRISPR-Cas9 experiments
  - Guide RNA selection and evaluation
  - Off-target prediction and scoring
  - Primer design for validation
  - CRISPR construct visualization

- **Variant Analysis Module**: Advanced tools for genetic variant interpretation
  - AI-powered variant effect prediction
  - Integration with ClinVar, Ensembl, and other databases
  - Protein structure impact visualization

- **Evolutionary Simulation Module**: Predictive modeling of gene edits over time
  - Population-level simulation of edited gene spread
  - Fitness landscape modeling
  - Natural selection and genetic drift simulation
  - Visualization of evolutionary trajectories
  - Powered by Evo2 DNA language model

- **Integration Features**: Seamless workflow from design to prediction
  - Import/export between modules
  - Comprehensive reporting
  - AI-assisted interpretation of results

## Applications

GenetiXplorer is designed for researchers in:

- **Synthetic Biology**: Design organisms with specific traits and understand how those traits might evolve
- **Conservation Biology**: Assess the impact of gene drives on controlling invasive species or preserving endangered populations
- **Medical Research**: Model the evolutionary implications of genetic therapies
- **Agricultural Biotechnology**: Develop and assess genetic modifications for crop improvement

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/genetixplorer.git
cd genetixplorer

# Create a virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Usage

```bash
# Start the web application
streamlit run app.py
```

## Project Structure

```
genetixplorer/
├── app.py                 # Main Streamlit application
├── requirements.txt       # Project dependencies
├── src/
│   ├── crispr/           # CRISPR design and analysis modules
│   ├── evolution/        # Evolutionary simulation modules
│   ├── integration/      # Integration between CRISPR and evolution
│   ├── ui/               # User interface components
│   └── utils/            # Shared utilities
├── tests/                # Test suite
├── data/                 # Sample and reference data
└── docs/                 # Documentation
```

## API Keys

Some features of GenetiXplorer require API keys for external services:
- Groq API (for AI-powered analysis)
- GenX API (for enhanced variant prediction)
- Evo2 API (for DNA language modeling and evolutionary simulations)

Store your API keys in `.streamlit/secrets.toml`:

```toml
GROQ_API_KEY = "your_groq_api_key"
GENX_API_KEY = "your_genx_api_key"
EVO2_API_KEY = "your_evo2_api_key"
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgements

- [ARC Institute](https://github.com/ArcInstitute/evo2) for the Evo2 DNA language model
- Groq for AI assistant capabilities 