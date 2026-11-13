from os import environ
from os.path import join
from pathlib import Path

# Default: DIETI cluster NAS path. Override for local/CI/Colab runs:
#   export DATA_ROOT=/path/to/dataset
_BASE_DATA_PATH = environ.get('DATA_ROOT', '/nas/data/datasets/')
_REPO_ROOT = Path(__file__).resolve().parent.parent.parent

def _resolve_data_path(rel_cluster_path, filename):
    """Resolve dataset path transparently across HPC cluster, Colab, and local environments."""
    if 'DATA_ROOT' in environ:
        p_env_sub = Path(environ['DATA_ROOT']) / rel_cluster_path
        if p_env_sub.exists():
            return str(p_env_sub)
        p_env = Path(environ['DATA_ROOT']) / filename
        if p_env.exists():
            return str(p_env)

    # Check local paths relative to repository root
    candidates = [
        _REPO_ROOT / 'dataset' / filename,
        _REPO_ROOT / 'data' / rel_cluster_path,
        _REPO_ROOT / 'data' / filename,
        Path('/nas/data/datasets/') / rel_cluster_path,
    ]
    for c in candidates:
        if c.exists():
            return str(c)

    return join(_BASE_DATA_PATH, rel_cluster_path)

dataset_config = {
    'cic-iomt': {  # class order
        'path': _resolve_data_path(
            join('cic-iomt', 'cic-iomt_firstDIRfixed_100pkts_7f_1p-mt1m_10p-mt100k_clean_uniform-label.parquet'),
            'cic-iomt_firstDIRfixed_100pkts_7f_1p-mt1m_10p-mt100k_clean_uniform-label.parquet'
        ),
    },
    'edge-iiot': {  # class order
        'path': _resolve_data_path(
            join('edge-iiot', 'edge-iiot_firstDIRfixed_100pkts_6f_50p-benign_1p-mt100k_clean_uniform-label.parquet'),
            'edge-iiot_firstDIRfixed_100pkts_6f_50p-benign_1p-mt100k_clean_uniform-label.parquet'
        ),
    },
    'ton-iot': {  # class order
        'path': _resolve_data_path(
            join('ton-iot', 'ton-iot_100pkts_firstDIRfixed_PLfixed_6f_1p-mt1m_10p-mt100k_clean_uniform-label.parquet'),
            'ton-iot_100pkts_firstDIRfixed_PLfixed_6f_1p-mt1m_10p-mt100k_clean_uniform-label.parquet'
        ),
    },
}

min_max_config = {
    'PL': (-1,1500), 'IAT': (-1,60000), 'DIR': (0,1), 'WIN': (-1,65535),
}
