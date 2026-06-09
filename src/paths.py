"""Central path and constant configuration for the whole pipeline."""
import os

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
DATA = os.path.join(REPO, "data")
INPUTS = os.path.join(DATA, "inputs")
OUTPUTS = os.path.join(DATA, "outputs")
FIGURES = os.path.join(REPO, "figures")

# inputs
WEIGHT = os.path.join(INPUTS, "weight.xlsx")                       # DRG master: Revenue, MeanLOS, ClusterId
WEIGHT_SEVERITY = os.path.join(INPUTS, "weight_with_severity.xlsx")
DISCHARGE = os.path.join(INPUTS, "discharge_20.xlsx")             # per-region per-DRG demand
DISCHARGE_CLUSTER = os.path.join(INPUTS, "dischargebycluster_20.xlsx")
BEDS = os.path.join(INPUTS, "beds_20.xlsx")                        # synthetic bed-day capacity

# consolidated outputs (gzip CSV)
BASE_OUTPUTS = os.path.join(OUTPUTS, "base_outputs.csv.gz")        # availability 0.8, all equity targets
AVAIL_OUTPUTS = os.path.join(OUTPUTS, "availability_outputs.csv.gz")  # sweep over availability 0.6-1.0
SOLVER_LOG = os.path.join(OUTPUTS, "solver_log.csv.gz")           # objective / gap / runtime per scenario

BASE_RATE = 5000.0          # assumed DRG base payment rate (USD)
N_CLUSTERS = 6
CLUSTER_COLORS = ["#0474ed", "#08a29e", "#91f0fa", "#a2b458", "#a6cabd", "#326164"]
