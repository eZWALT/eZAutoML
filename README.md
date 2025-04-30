# eZAutoML 

## Overview

## Installation 

Latest published version can be installed through PyPI using:

```bash 
pip install ezautoml
ezautoml --help
```

### Install from source
To install from source, you can clone this repo and install with `pip`:

```
pip install -e .
```

## Usage

### Command Line Interface 

Usage:

```bash
ezautoml --dataset <path_to_data> --target <target_name> --task <classification|regression> --models <model1,model2,...> --cv <folds> --output <path_to_output>
```

Options:
- dataset: Path to the dataset file (CSV, parquet...)
- target: The target column name for prediction
- task: Task type: classification or regression
- search: Black-box optimization algorithm to perform
- models: Comma-separated list of models to use (e.g., lr,rf,xgb). Use initials!
- cv: Number of cross-validation folds (if needed)
- output: Directory to save the output models/results
- trials: Maximum number of trials inside an optimiation algorithm
- preprocess: Whether to perform minimal preprocessing (Scaling, Encoding...) or not
- verbose: Increase logging verbosity 
- version: Show the current version 

For more detailed help, use:

```bash
ezautoml --help
```

There are future features that are still a work-in-progress and will be enabled in the future such as scheduling, metalearning, pipelines...

### Python Script


## Features & WIP
3 core components:

## Contributing

## License 