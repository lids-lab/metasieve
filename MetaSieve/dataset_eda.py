import os
import time
import math
import torch
import duckdb
import numpy as np
import pandas as pd
from tqdm import tqdm
from pathlib import Path
from typing import Dict, Tuple, List, Optional, Iterable, Any
from concurrent.futures import ThreadPoolExecutor, as_completed

from relbench.tasks import get_task
from relbench.datasets import get_dataset
from relbench.modeling.utils import get_stype_proposal
from relbench.modeling.graph import make_pkey_fkey_graph

from torch_geometric.seed import seed_everything
from torch_frame.config.text_embedder import TextEmbedderConfig

from candidate_generation_utils import *
from embeddings import GloveTextEmbedding


# TASK_NAME = "user-engagement"
# DATASET_NAME = "rel-stack"

TASK_NAME = "site-success"
DATASET_NAME = "rel-trial"

def count_zeros_ones(table, col="did_not_finish"):
    counts = table.df[col].value_counts(dropna=False)
    return {"0s": int(counts.loc[0]), "1s": int(counts.loc[1])}


def count_zeros_nonzeros(table, col="ltv"):
    s = table.df[col]
    nans = int(s.isna().sum())
    s = s.dropna()
    zeros = int((s == 0).sum())
    non_zeros = int((s != 0).sum())
    return {"zeros": zeros, "non_zeros": non_zeros, "nans": nans}


device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
root_dir = "./data"

dataset = get_dataset(DATASET_NAME, download=True)
task = get_task(DATASET_NAME, TASK_NAME, download=True)


train_table = task.get_table("train")
val_table   = task.get_table("val")
test_table  = task.get_table("test")

print(train_table)
# print(val_table)

# print("train:", count_zeros_ones(train_table, col="outcome"))
# print("val:  ", count_zeros_ones(val_table, col="outcome"))

print("train:", count_zeros_nonzeros(train_table, col="popularity"))
print("val:  ", count_zeros_nonzeros(val_table, col="popularity"))