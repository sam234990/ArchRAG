from src.dataset.webqsp import WebQDataset
from src.dataset.webqsp_baseline import WebQSPBaselineDataset
from src.dataset.mintaka import MintakaDataset

load_dataset = {
    "webq": WebQDataset,
    "mintaka": MintakaDataset,
    "webqsp_baseline": WebQSPBaselineDataset,
}

dataset_output_dir = {
    "webq":"/mnt/data/wangshu/hcarag/FB15k/KG",
    "mintaka":"/mnt/data/wangshu/hcarag/mintaka/KG"
}