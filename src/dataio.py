"""Loaders for inputs and consolidated outputs."""
import pandas as pd

import paths


def load_weight(as_int_cluster=False):
    drg = pd.read_excel(paths.WEIGHT, dtype={"DRGCode": "string", "ClusterId": "string"})
    if as_int_cluster:
        drg["ClusterId"] = drg["ClusterId"].astype(int)
    return drg


def load_discharge():
    return pd.read_excel(paths.DISCHARGE, dtype={"DRGCode": "string"})


def load_discharge_cluster():
    return pd.read_excel(paths.DISCHARGE_CLUSTER, dtype={"ClusterId": "string"})


def load_beds():
    return pd.read_excel(paths.BEDS)


def load_base_outputs():
    """Canonical base-case (availability 0.8) MILP allocations."""
    return pd.read_csv(paths.BASE_OUTPUTS, dtype={"DRG": "string"})


def load_availability_outputs():
    """Resource-availability sweep allocations (availability 0.6-1.0)."""
    return pd.read_csv(paths.AVAIL_OUTPUTS, dtype={"DRG": "string"})


def load_solver_log():
    return pd.read_csv(paths.SOLVER_LOG)


def revenue_map():
    return dict(load_weight()[["DRGCode", "Revenue"]].values)


def cluster_map(as_int=False):
    drg = load_weight()
    return dict(zip(drg["DRGCode"], drg["ClusterId"].astype(int) if as_int else drg["ClusterId"]))
