from src.dataset.webqsp import WebQDataset
from src.dataset.webqsp_baseline import WebQSPBaselineDataset
from src.dataset.mintaka import MintakaDataset

load_dataset = {
    "webqsp": WebQDataset,
    "mintaka": MintakaDataset,
    "webqsp_baseline": WebQSPBaselineDataset,
}
