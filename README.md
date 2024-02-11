# AquaNet
## Introduction

AquaNet is a comprehensive toolkit designed for analyzing and visualizing aquatic food web networks.
This project provides a dataset of 173 graphs in GraphML format, alongside Python scripts for conducting experiments of these networks.
The measurements include analysis of Core/Periphery structures within the Food Web, robustness, identification of critical nodes that hold the network together, and a detailed 3-nodes motif representation.

## Installation
To install the necessary packages for AquaNet, make sure you have Python installed on your system and run the following command to install dependencies:

```bash
pip install -r requirements.txt
```

## Usage

### Configuration
The config.ini file in bin/ directory allows users to specify paths relative to the main folder for various inputs and outputs. Here's an overview of the configuration sections and their purposes:

[dataset]: Define the path to the Food Web dataset.
[eda]: Specify where to store the exploratory data analysis table.
[core and periphery]: Paths for storing measurements of core and periphery sizes, node classifications, and node vulnerabilities and generalities.
[critical nodes]: Define where to store the sequence of critical nodes and the robustness index for each food web.
[motifs representation]: Specify paths for real and swapped triad census data, motif representations, and motif roles.
[plots]: Paths for output images visualizing core and periphery results.

You can edit or create your own configuration file accoding to the format specified.

### Running experiments
To run experiments with AquaNet, a bash script run_experiments.sh is provided, which executes the analysis based on configurations defined in config.ini. To start the experiments, use:

```bash
bash run_experiments.sh -c bin/config.ini
```

### Visualizations

```bash
bash plot_images.sh -c bin/config.ini
```

At the moment only to plots are supported.

### Print pickle function
A bash script to print the table within the terminal
```bash
bash display_table.sh -t <table_path>
```

## Results of the experiments
The results of our experiments are stored in the directory results

- ```core_and_periphery_size.pickle```
  
  This file contains ...
- ```eda.csv```
  
  This file contains ...
- ```eda.pickle```
  
  This file contains ...
- ```generality_vulnerability_living_nodes.pickle```
  
  This file contains ...
- ```motif_representation.pickle```
  
  This file contains ...
- ```motif_representation.pickle```
  
  This file contains ...
- ```motif_representation_table.xlsx```
  
  This file contains ...
- ```node_classification_core_periphery_dataframe.pickle```
  
  This file contains ...
- ```node_sequence.pickle```
  
  This file contains ...
- ```real_networks_triad_census_living.pickle```
  
  This file contains ...
- ```robustness.pickle```
  
  This file contains ...
- ```swapped_triad_census_living.pickle```
  
  This file contains ...
