# General Settings
aquanet_folder = '/home/davide/AI/Projects/AquaNet'
dataframe_output_folder = f'{aquanet_folder}/dataframes'
dataset_folder = f'{aquanet_folder}/dataset'
verbose = True
article_images_folder = f'{aquanet_folder}/assets/article/images'
article_table_folder = f'{aquanet_folder}/assets/article/tables'
sm_images_folder = f'{aquanet_folder}/assets/supplementary_material/images'
sm_table_folder = f'{aquanet_folder}/assets/supplementary_material/tables'
seed = 12345

# EDA Settings
df_eda_path = f'{dataframe_output_folder}/eda/eda.pickle'

# Core and Periphery Settings
df_nodes_core_periphery_classification_path = f'{dataframe_output_folder}/core_periphery/node_classification_core_periphery.pickle'
df_links_core_periphery_classification_path = f'{dataframe_output_folder}/core_periphery/link_classification_core_periphery.pickle'
seasonal_food_webs_report_path = f'{dataframe_output_folder}/core_periphery/seasonal_food_webs.txt'
df_core_periphery_generality_living_proportion_path = f'{dataframe_output_folder}/core_periphery/generality_living_proportion.pickle'
df_core_periphery_generality_all_proportion_path = f'{dataframe_output_folder}/core_periphery/generality_all_proportion.pickle'

# Critical Nodes Settings
df_critical_nodes_path = f'{dataframe_output_folder}/critical_nodes/critical_nodes.pickle'
df_robustness_path = f'{dataframe_output_folder}/critical_nodes/robustness.pickle'
top_3_critical_nodes_path = f'{dataframe_output_folder}/critical_nodes/top_3_critical_nodes.tex'
df_kendalltau_path = f'{dataframe_output_folder}/critical_nodes/kendalls_tau_outdegree.pickle'

# Three-node Motifs
num_random_graph_samples_triad = 50
df_motif_Z_scores_path = f"{dataframe_output_folder}/three_node_motif/z_score_swap.pickle"
df_motif_randomization_data_path = f"{dataframe_output_folder}/three_node_motif/motif_randomization_data.pickle"
df_biome_path = f"{dataframe_output_folder}/biomes.csv"