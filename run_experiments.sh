#!/bin/bash
function framed_title() {
    TITLE="$1"

    # Calculate the length of the frame based on the length of the TITLE
    TITLE_LENGTH=${#TITLE}
    FRAME_LENGTH=$((TITLE_LENGTH + 4))  # 2 spaces on each side of the title

    # Create the top frame
    echo
    for ((i = 0; i < FRAME_LENGTH; i++)); do
        echo -n "-"
    done
    echo

    # Display the title in the frame
    echo "  $TITLE"

    # Create the bottom frame
    for ((i = 0; i < FRAME_LENGTH; i++)); do
        echo -n "-"
    done
    echo
}

while getopts ":c:-:" opt; do
  case "$opt" in
    c)
      CONFIG_PATH="$OPTARG"
      ;;
    -)
      case "${OPTARG}" in
        config=*)
          CONFIG_PATH="${OPTARG#*=}"
          ;;
        *)
          echo "Invalid option: --$OPTARG" >&2
          exit 1
          ;;
      esac
      ;;
    *)
      echo "Invalid option: -$opt" >&2
      exit 1
      ;;
  esac
done

# Check if CONFIG_PATH is set and exists
if [ -z "$CONFIG_PATH" ]; then
  echo "Please provide a config file using the -c or --config flag."
  exit 1
fi

if [ ! -f "$CONFIG_PATH" ]; then
  echo "Config file '$CONFIG_PATH' does not exist."
  exit 1
fi

echo "Using config file: $CONFIG_PATH"

framed_title "EDA"
python -m bin.measurements.eda -c ${CONFIG_PATH}

framed_title "Core and periphery structure"
python -m bin.measurements.core_periphery_classification -c ${CONFIG_PATH}

framed_title "Generality and vulnerability of each node"
python -m bin.measurements.generality_vulnerability -c ${CONFIG_PATH}

framed_title "Most critical node: sequence and robustness"
python -m bin.measurements.node_sequence_robustness_index -c ${CONFIG_PATH}

framed_title "Triad census - real food webs"
python -m bin.measurements.triad_census_real -c ${CONFIG_PATH}

framed_title "Triad census - randomized food webs (swap)"
python -m bin.measurements.triad_census_swap -c ${CONFIG_PATH}

framed_title "Z-score evaluation"
python -m bin.measurements.z_score -c ${CONFIG_PATH}
