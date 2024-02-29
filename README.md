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

## Dataset content

Each graph in the dataset represents a food web, detailing the feeding interactions between different compartments within an ecological system. These interactions are representated as a *directed edge* between two nodes. In particular the interaction A->B corresponds to a flow of biomass from A to B (e.g. A is eaten by B) [1].
The graphs are structured using .graphml format. For the entire analysis, we used Python's igraph library.

### Graph Level Attributes:
- **name**: This attribute represents the name of the food web, which typically includes the geographical location and the year of the study when available (e.g., "Yucatan (1987)"). It provides context and identification for the ecological network being represented.
- **Author**: When available, indicates the primary researcher(s) or the author(s) of the study from which the food web data was derived.
- **Citation**: When available, provides the formal citation for the study or publication that presented the food web. This enables users of the dataset to reference the original work accurately in their own publications or research.
Regardless this attribute, for each graph we've added the whole bibliograph for this dataset in the Supplementary Materials.
- **URL**: When available, offers a direct link additional to the resource.

### Node (Vertex) Level Attributes:
- **ECO**: Stands for the ecological type of the node. It differentiates between living beings (1) and nonliving organic deposit (2) like detritus, debris, Particulate Organic Carbon (POC), Dissolved Organic Carbon (DOC), etc.
- **id**: A unique identifier for each node within the graph.
- **name**: The name of the node, which could be the species name, a group of species, or nonliving organic deposit components within the ecological system.
- **trophic_level**: Represents the trophic level of the node within the food web, indicating the node's position in the ecosystem's food chain. Trophic levels range from primary producers at the base (trophic level 1), through various levels of consumers, to apex predators at the top. Decomposers and detritivores are also included in this spectrum, typically occupying specific trophic levels based on their feeding behavior.
The trophic level ($TL_i$) of node $i$ is calculated using the formula:

$TL_i = 1 + 1 / k_{i} \sum_j A_{ij} TL_j$

Here, $k_{i}$ represents the weighted in-degree of node $i$, which is the number of different species (or nodes) that node $i$ feeds upon. $A_{ij}$ is the weighted interaction matrix. If the ```weight``` or ```proportion``` is not an attribute of the edge, we'll set the $A_{ij}$ elements to 1 when there is a feeding interaction from $j$ to $i$. 
Primary producers (organisms that do not consume others for biomass, such as plants and algae) and nonliving organic deposits are assigned a trophic level ($TL$) of 1, reflecting their foundational role in the food web as sources of energy and matter, according to [2].

**Biomass**: When available, represents the biomass of the node, quantified in milligrams of a specified medium (commonly carbon) per unit of surface area (typically square meters). This measure reflects the content of the biomass within a defined area of the ecosystem, offering insights into the ecosystem's productivity and the role of different nodes within the carbon cycling dynamics.
  
### Edge Level Attributes:
**weight**: When available, it indicates the flow of biomass between compartments. It is quantified in Biomass per time units.
**proportion**: When available, it indicates the proportion of the flow of biomass between compartments.

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

### Print pickle function
A bash script to print the table within the terminal an well as a description of the table.
```bash
bash display_table.sh -t <table_path>
```

## Results of the experiments
The results of our experiments are stored in the directory ```results/```. We provided the outputs both in ```.pickle``` and ```.xlsx``` format.

- ```eda.xsls```
    A file containing global metrics of the graphs in the dataset. These metrics are S - the number of nodes; L - the number of directed interactions or links; C - the connectivity value (C := L/S^2); B/N - the ratio between the basal nodes (the one that have zero in degree) and all the nodes of the graph; det/N - the ratio between the other compartmenrs and all the nodes in the graph; clustering - the average local undirected clustering coefficient.


- ```core_and_periphery_size.xlsx```
    The cells contain the size of each core/periphery structure - i.e. the number of nodes of that graph beloning to the given structure and the number of vertices of the whole graph.
  
- ```node_classification_core_periphery_dataframe.xlsx```
    For each node the classification in the node and periphery structure as well as their type (either living compartment - ECO = 1, or other compartment - ECO = 2).

- ```generality_vulnerability_living_nodes.xlsx```
    For each node in each graph, the structure it belongs to and the value of generality and vulnerability.
  

- ```motif_representation.xlsx```
    The z-score value of each motif for each graph in the dataset. Each value is evaluated with an ensamble of 50 randomized graph with the swap algorithm.
    
- ```motif_representation_table.xlsx```
    A schematic representation of the z-score profile: an up-arrow means that the motif is over-represented, a down-arrow means that the motif is under-represented, a minus means that the motif is not present in the dataset nor in the random ensamble.

- ```real_networks_triad_census_living.xlsx```
    Number of 3-nodes motifs for each graph.

- ```swapped_triad_census_living.xlsx```
    Number of triads in the randomized network. 


- ```node_sequence.xlsx```
    Sequence of critical nodes evaluated with the algorithm described in the paper. ```Fraction of reachable``` pairs represent the fraction of reachable pairs measured after the removal of the links adjacent and incident to the node, while ```N reachable pairs``` is its absolute value. ```Edge-Removed Nodes Fraction``` represents the fraction of nodes from which adjacent and incident arcs have been removed.

- ```robustness.xlsx```
    Robustness of each graph with the equation in the paper.

- ```robustness_and_top_5_most_critical_nodes.xlsx```
    Robustness of each graph and its top 5 most critical nodes.
    

---
# References
[1] Dunne, Jennifer A. "Food Webs." Encyclopedia of complexity and systems science 1 (2009): 3661-3682.

[2] Williams, Richard J., and Neo D. Martinez. "Limits to trophic levels and omnivory in complex food webs: theory and data." The American Naturalist 163.3 (2004): 458-468.