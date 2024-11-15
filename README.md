# AquaNet
## Introduction

AquaNet is a comprehensive toolkit designed for analyzing and visualizing aquatic food web networks.
This project provides a dataset of 173 food webs in GraphML format, alongside Python scripts for conducting experiments of these networks.
The measurements include analysis of Core/Periphery structures within the Food Web, robustness, identification of critical nodes that hold the network together, and a detailed 3-nodes motif representation.

## Installation
To install the necessary packages for AquaNet, make sure you have Python installed on your system and run the following command to install dependencies:

```bash
pip install -r requirements.txt
```

## Dataset content

Each file in the dataset represents a food web, detailing the feeding interactions between different compartments within an ecological system. These interactions are representated as a *directed edge* between two nodes. In particular the interaction A->B corresponds to a flow of biomass from A to B (e.g. A is eaten by B) [1].
The food webs are structured using .graphml format. For the entire analysis, we used Python's igraph library.

### Downloading and reading the dataset
The dataset is hosted within the ```dataset``` directory.
To download the datasets, you can either clone the AquaNet repository or directly download the files from the GitHub web interface.
1) Cloning the Repository:
```bash
git clone https://github.com/davidetorre92/AquaNet.git
```
2) Navigate to ```dataset```

Alternatively, you can download individual GraphML files directly from the GitHub web interface at AquaNet's dataset directory. Click on the desired file and use the Download button.

### Reading the Datasets in Python
The food web networks are saved in GraphML format, which can be easily read using the igraph Python library. If you haven't already, install ```igraph``` using pip:
```bash
pip install igraph
```

Here's how to read a GraphML file and print a summary of the food web network:

```python
from igraph import Graph

# Replace 'path/to/your/file.graphml' with the actual file path
file_path = 'path/to/your/file.graphml'

# Load the GraphML file
g = Graph.Read_GraphML(file_path)

# Print a summary of the food web
print(g.summary())
```


### Graph Level Attributes:
- **name**: This attribute represents the name of the food web, which typically includes the geographical location and the year of the study when available (e.g., "Yucatan (1987)"). It provides context and identification for the ecological network being represented.
- **Author**: When available, indicates the primary researcher(s) or the author(s) of the study from which the food web data was derived.
- **Citation**: When available, provides the formal citation for the study or publication that presented the food web. This enables users of the dataset to reference the original work accurately in their own publications or research.
Regardless this attribute, for each food web we've added the whole bibliograph for this dataset in the Supplementary Materials.
- **URL**: When available, offers a direct link additional to the resource.

### Node (Vertex) Level Attributes:
- **ECO**: Stands for the ecological type of the node. It differentiates between living beings (1) and nonliving organic deposit (2) like detritus, debris, Particulate Organic Carbon (POC), Dissolved Organic Carbon (DOC), etc.
- **id**: A unique identifier for each node within the food web.
- **name**: The name of the node, which could be the species name, a group of species, or nonliving organic deposit components within the ecological system.
- **trophic_level**: Represents the trophic level of the node within the food web, indicating the node's position in the ecosystem's food chain. Trophic levels range from primary producers at the base (trophic level 1), through various levels of consumers, to apex predators at the top. Decomposers and detritivores are also included in this spectrum, typically occupying specific trophic levels based on their feeding behavior.
The trophic level ($TL_i$) of node $i$ is calculated using the formula:

$TL_i = 1 + 1 / k_{i} \sum_j A_{ji} TL_j$

Here, $k_{i}$ represents the weighted in-degree of node $i$, which is the number of different species (or nodes) that node $i$ feeds upon. $A_{ij}$ is the weighted interaction matrix. If the ```weight``` or ```proportion``` is not an attribute of the edge, we'll set the $A_{ij}$ elements to 1 when there is a feeding interaction from $i$ to $j$. 
Primary producers (organisms that do not consume others for biomass, such as plants and algae) and nonliving organic deposits are assigned a trophic level ($TL$) of 1, reflecting their foundational role in the food web as sources of energy and matter, according to [2].

**Biomass**: When available, represents the biomass of the node, quantified in milligrams of a specified medium (commonly carbon) per unit of surface area (typically square meters). This measure reflects the content of the biomass within a defined area of the ecosystem, offering insights into the ecosystem's productivity and the role of different nodes within the carbon cycling dynamics.
 
### Edge Level Attributes:
**weight**: When available, it indicates the flow of biomass between compartments. It is quantified in Biomass per time units.
**proportion**: When available, it indicates the proportion of the flow of biomass between compartments.

## Usage

This section is for those who want to perform the experiments reported in the article.
The codes given here have been tested on a computer with the Ubuntu 22.04 operating system with Python v.3.10

### Prerequisites
1) Clone the repository
```bash
git clone https://github.com/davidetorre92/AquaNet.git
```

Navigate in the directory.
2) Set the environmet
```bash
pip -m venv your_virtual_environment_name # or pip3
source your_virtual_environment_name/bin/activate
pip install -r requirements.txt # or pip3
bash set_environment.sh
```

Now you are ready to use the package
### Settings
The ```settings.py``` file in the main directory allows users to specify paths relative to the main folder for various inputs and outputs.

You can edit or create your own configuration file accoding to the format specified.

### Running experiments
The experiments are organized in the folder ```measurements```.
Each script is an independent code in which the settings are managed in the settings file.
### Visualizations

The visualization scripts are organized in the folder ```visualization_scripts```

### Print pickle function
A bash script to print the table within the terminal an well as a description of the table.
```bash
bash display_table.sh -t <table_path>
```

---
# References
[1] Dunne, Jennifer A. "Food Webs." Encyclopedia of complexity and systems science 1 (2009): 3661-3682.

[2] Williams, Richard J., and Neo D. Martinez. "Limits to trophic levels and omnivory in complex food webs: theory and data." The American Naturalist 163.3 (2004): 458-468.
