import copy
import torch
from torch_geometric.data import Data
from relbench.modeling.nn import HeteroEncoder
from relbench.modeling.graph import get_node_train_table_input

def relbench_hetero_to_homo_for_mps(
    data_hetero,
    col_stats_dict,
    channels: int,
    device: torch.device,
) -> Data:
    """
    Returns a homogeneous PyG Data object for MPS:
      - x: [N, channels]
      - edge_index: [2, E]
      - node_type: [N] (int type ids)
      - edge_type: [E] (int relation ids)

    Also attaches:
      - node_type_names: list(data_hetero.node_types)
      - edge_type_names: list(data_hetero.edge_types)
    """
    enc_device = torch.device("cpu")
    data_tmp = copy.copy(data_hetero)

    encoder = HeteroEncoder(
        channels=channels,
        node_to_col_names_dict={nt: data_tmp[nt].tf.col_names_dict for nt in data_tmp.node_types},
        node_to_col_stats=col_stats_dict,
    ).to(enc_device)
    encoder.eval()

    with torch.no_grad():
        x_dict = encoder({nt: data_tmp[nt].tf for nt in data_tmp.node_types})

    for nt, x in x_dict.items():
        data_tmp[nt].x = x.cpu()

    data_homo = data_tmp.to_homogeneous(
        node_attrs=["x","time"],
        add_node_type=True,
        add_edge_type=True,
    )

    # Attach mappings (public + internal). PyG uses internal _node/_edge_type_names for ordering. :contentReference[oaicite:1]{index=1}
    node_type_names = list(data_hetero.node_types)
    edge_type_names = list(data_hetero.edge_types)

    data_homo.node_type_names = node_type_names
    data_homo.edge_type_names = edge_type_names
    data_homo._node_type_names = node_type_names
    data_homo._edge_type_names = edge_type_names
    try:
        data_homo._store._node_type_names = node_type_names
        data_homo._store._edge_type_names = edge_type_names
    except Exception:
        pass

    return data_homo


def _build_type_offsets(data_hetero, data_homo):
    # to_homogeneous concatenates node types in node_type_names order.
    offsets = {}
    cur = 0
    for nt in data_homo.node_type_names:
        offsets[nt] = cur
        cur += data_hetero[nt].num_nodes
    return offsets


def attach_relbench_labels_and_masks_to_homo(
    data_homo: Data,
    data_hetero,
    task,
    train_table,
    val_table,
    test_table,
    device,
):
    """
    Creates:
      data_homo.mpgnn_y: LongTensor [num_nodes] (only entity nodes are labeled, others stay 0)
      data_homo.{train,val,test}_mask: BoolTensor [num_nodes]
    """
    # EntityTask exposes entity_col / target_col / time_col, etc. :contentReference[oaicite:1]{index=1}
    offsets = _build_type_offsets(data_hetero, data_homo)

    n = data_homo.num_nodes
    data_homo.mpgnn_y = torch.zeros(n, dtype=torch.long)
    data_homo.train_mask = torch.zeros(n, dtype=torch.bool)
    data_homo.val_mask   = torch.zeros(n, dtype=torch.bool)
    data_homo.test_mask  = torch.zeros(n, dtype=torch.bool)

    def _fill(split_table, mask_tensor):
        table_input = get_node_train_table_input(table=split_table, task=task)
        entity_type, seed_ids = table_input.nodes  # ('user', tensor([...]))
        seed_ids = seed_ids.to(torch.long).cpu()

        # global ids in homogeneous space:
        gidx = seed_ids + offsets[entity_type]

        # labels from task.target_col (binary for user-badge)
        y = torch.tensor(split_table.df[task.target_col].to_numpy(), dtype=torch.long)

        mask_tensor[gidx] = True
        data_homo.mpgnn_y[gidx] = y

    _fill(train_table, data_homo.train_mask)
    _fill(val_table,   data_homo.val_mask)

    return data_homo


# ----------------------------
# Helper: make the MPS "scoring" object (bags/alpha/target_ids) for ONLY labeled entity nodes
# ----------------------------
def make_mps_scoring_data(data_homo: Data, use_split: str = "train"):
    """
    MPS authors assume y aligns with bags and target_ids.
    In RelBench, only entity nodes have labels → we build bags only for those nodes.
    """
    if use_split == "train":
        seed = torch.where(data_homo.train_mask)[0]
    elif use_split == "val":
        seed = torch.where(data_homo.val_mask)[0]
    else:
        raise ValueError("use_split must be 'train' or 'val'")

    # bag labels must be 0/1 ints for their sampling_bags() logic
    bag_y = data_homo.mpgnn_y[seed].to(torch.long).cpu()

    bags = [[int(i)] for i in seed.tolist()]
    alpha = {(int(i), tuple([int(i)])): 1.0 for i in seed.tolist()}

    data_score = Data(
        x=data_homo.x,
        edge_index=data_homo.edge_index,
        edge_type=data_homo.edge_type,
        y=bag_y,
        num_nodes=data_homo.num_nodes,
        bags=bags,
        alpha=alpha,
        target_ids=seed.tolist(),
    )
    return data_score



