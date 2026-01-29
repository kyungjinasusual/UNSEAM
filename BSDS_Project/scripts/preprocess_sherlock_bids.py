#!/usr/bin/env python3
"""
Sherlock BIDS 데이터 전처리 스크립트

BIDS 형식의 raw Sherlock fMRI 데이터에서 ROI/Network 시계열을 추출합니다.

Mode 1: Network-level (Yang 2023 스타일)
- Schaefer-Yeo 7 Networks: Vis, SomMot, DorsAttn, SalVentAttn, Limbic, Cont, Default
- 각 네트워크별 평균 또는 PCA 추출

Mode 2: ROI-level (Baldassano 2017 스타일)
- 특정 ROI (AG, PMC, PCC, EAC, V1, mPFC)

Mode 3: Parcel-level (전체 파셀)
- 모든 Schaefer 파셀의 시계열

Usage:
    # 7 Networks (권장)
    python preprocess_sherlock_bids.py --bids-dir /storage/bigdata/Sherlock \\
        --output-dir ./data/sherlock_7networks --mode networks

    # 특정 ROI
    python preprocess_sherlock_bids.py --bids-dir /storage/bigdata/Sherlock \\
        --output-dir ./data/sherlock_roi --mode roi --roi AG

    # 전체 파셀
    python preprocess_sherlock_bids.py --bids-dir /storage/bigdata/Sherlock \\
        --output-dir ./data/sherlock_parcels --mode parcels
"""

import argparse
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

# fMRI processing
try:
    import nibabel as nib
    from nilearn import datasets, image, signal
    from nilearn.maskers import NiftiLabelsMasker
    NILEARN_AVAILABLE = True
except ImportError:
    NILEARN_AVAILABLE = False
    print("Warning: nilearn not available. Install with: pip install nilearn")


# =============================================================================
# Schaefer-Yeo 7 Networks Definition
# =============================================================================
# Network names in Schaefer atlas labels (7 networks version)
SCHAEFER_7_NETWORKS = [
    'Vis',        # Visual
    'SomMot',     # Somatomotor
    'DorsAttn',   # Dorsal Attention
    'SalVentAttn',# Salience/Ventral Attention
    'Limbic',     # Limbic
    'Cont',       # Control (Frontoparietal)
    'Default',    # Default Mode Network
]

# Short names for output
NETWORK_SHORT_NAMES = {
    'Vis': 'VIS',
    'SomMot': 'SMN',
    'DorsAttn': 'DAN',
    'SalVentAttn': 'VAN',
    'Limbic': 'LIM',
    'Cont': 'FPC',
    'Default': 'DMN',
}

# Specific ROI mapping (approximate parcel indices)
SCHAEFER_ROI_MAPPING = {
    'AG': {  # Angular Gyrus - Default Mode Network
        'networks': ['Default'],
        'keywords': ['IPL', 'Ang', 'PGp', 'PGa', 'Angular'],
        'schaefer_ids_200': list(range(89, 95)) + list(range(189, 195)),
    },
    'PMC': {  # Posterior Medial Cortex
        'networks': ['Default'],
        'keywords': ['PCC', 'Precun', 'RSC'],
        'schaefer_ids_200': list(range(77, 85)) + list(range(177, 185)),
    },
    'PCC': {  # Posterior Cingulate Cortex
        'networks': ['Default'],
        'keywords': ['PCC', 'pCun'],
        'schaefer_ids_200': list(range(77, 82)) + list(range(177, 182)),
    },
    'EAC': {  # Early Auditory Cortex
        'networks': ['SomMot'],
        'keywords': ['Aud', 'A1', 'STG', 'TE1'],
        'schaefer_ids_200': list(range(20, 25)) + list(range(120, 125)),
    },
    'V1': {  # Primary Visual Cortex
        'networks': ['Vis'],
        'keywords': ['V1', 'Vis', 'Striate'],
        'schaefer_ids_200': list(range(0, 10)) + list(range(100, 110)),
    },
    'mPFC': {  # Medial Prefrontal Cortex
        'networks': ['Default'],
        'keywords': ['mPFC', 'MPFC', 'PFCm'],
        'schaefer_ids_200': list(range(65, 72)) + list(range(165, 172)),
    },
}


def load_sherlock_bids(bids_dir: Path,
                       task: str = 'sherlockPart1',
                       subjects: Optional[List[str]] = None) -> Dict[str, Dict]:
    """
    BIDS 형식의 Sherlock 데이터 로드

    Args:
        bids_dir: BIDS 데이터 루트 디렉토리
        task: 'sherlockPart1', 'sherlockPart2', 또는 'freerecall'
        subjects: 피험자 리스트 (None이면 전체)

    Returns:
        Dictionary with subject data
    """
    bids_dir = Path(bids_dir)

    # Find all subjects
    subject_dirs = sorted(bids_dir.glob('sub-*'))

    if subjects:
        subject_dirs = [d for d in subject_dirs if d.name in subjects]

    print(f"Found {len(subject_dirs)} subjects")

    data = {}
    for subj_dir in subject_dirs:
        subj_id = subj_dir.name
        func_dir = subj_dir / 'func'

        # Find the BOLD file
        bold_files = list(func_dir.glob(f'*_task-{task}_bold.nii.gz'))

        if not bold_files:
            print(f"  Warning: {subj_id}: No {task} BOLD file found")
            continue

        bold_file = bold_files[0]

        # Check if file exists (might be git-annex symlink)
        if not bold_file.exists():
            print(f"  Warning: {subj_id}: BOLD file not available (git-annex?)")
            continue

        # Check file size (git-annex symlinks have small size)
        try:
            file_size = bold_file.stat().st_size
            if file_size < 1000000:  # Less than 1MB is suspicious for fMRI
                print(f"  Warning: {subj_id}: BOLD file too small ({file_size} bytes) - run 'git annex get'")
                continue
        except OSError:
            print(f"  Warning: {subj_id}: Cannot access BOLD file")
            continue

        data[subj_id] = {
            'bold_file': bold_file,
            'subj_dir': subj_dir,
        }
        print(f"  OK: {subj_id}: {bold_file.name}")

    return data


def get_network_parcel_mapping(labels: List[str]) -> Dict[str, List[int]]:
    """
    Schaefer 레이블에서 네트워크별 파셀 인덱스 매핑 생성

    Args:
        labels: Schaefer atlas labels (e.g., '7Networks_LH_Vis_1')

    Returns:
        Dictionary mapping network name to list of parcel indices
    """
    network_parcels = {net: [] for net in SCHAEFER_7_NETWORKS}

    for idx, label in enumerate(labels):
        # Handle bytes
        if isinstance(label, bytes):
            label = label.decode('utf-8')

        # Parse network from label (format: 7Networks_LH_NetworkName_RegionNum)
        for network in SCHAEFER_7_NETWORKS:
            if f'_{network}_' in label or label.endswith(f'_{network}'):
                network_parcels[network].append(idx)
                break

    # Print summary
    print("\nNetwork-Parcel Mapping:")
    for net, parcels in network_parcels.items():
        short = NETWORK_SHORT_NAMES[net]
        print(f"  {short} ({net}): {len(parcels)} parcels")

    return network_parcels


def extract_network_timeseries(bold_file: Path,
                                n_parcels: int = 200,
                                method: str = 'mean',
                                standardize: bool = True,
                                detrend: bool = True,
                                low_pass: float = 0.1,
                                high_pass: float = 0.01,
                                t_r: float = 1.5) -> Tuple[np.ndarray, List[str], Dict]:
    """
    7 Networks별 시계열 추출 (Yang 2023 스타일)

    Args:
        bold_file: NIfTI BOLD 파일 경로
        n_parcels: Schaefer 파셀 수 (100, 200, 400)
        method: 'mean' (평균), 'pca' (1st PC), 'all' (모든 파셀)
        standardize: Z-score 정규화
        detrend: 선형 추세 제거
        low_pass: 저주파 필터 컷오프 (Hz)
        high_pass: 고주파 필터 컷오프 (Hz)
        t_r: TR (초)

    Returns:
        Tuple of:
        - network_timeseries: (T x 7) for mean/pca, or (T x n_parcels) for all
        - network_names: List of network names
        - metadata: Additional information
    """
    if not NILEARN_AVAILABLE:
        raise ImportError("nilearn required. pip install nilearn")

    print(f"  Loading BOLD data from {bold_file.name}...")
    bold_img = nib.load(bold_file)

    # Get Schaefer atlas
    print(f"  Fetching Schaefer {n_parcels} atlas (7 networks)...")
    schaefer = datasets.fetch_atlas_schaefer_2018(
        n_rois=n_parcels,
        yeo_networks=7,
        resolution_mm=2,
        verbose=0
    )

    labels = [l.decode() if isinstance(l, bytes) else str(l)
              for l in schaefer['labels']]

    # Create masker
    masker = NiftiLabelsMasker(
        labels_img=schaefer['maps'],
        standardize=standardize,
        detrend=detrend,
        low_pass=low_pass,
        high_pass=high_pass,
        t_r=t_r,
        memory='nilearn_cache',
        verbose=0
    )

    # Extract all parcel timeseries
    print(f"  Extracting parcel timeseries...")
    all_timeseries = masker.fit_transform(bold_img)  # (T, n_parcels)
    T = all_timeseries.shape[0]
    print(f"  Raw timeseries shape: {all_timeseries.shape}")

    # Get network mapping
    network_parcels = get_network_parcel_mapping(labels)

    if method == 'all':
        # Return all parcels with network labels
        network_names = []
        for idx, label in enumerate(labels):
            for net in SCHAEFER_7_NETWORKS:
                if f'_{net}_' in label:
                    network_names.append(f"{NETWORK_SHORT_NAMES[net]}_{idx}")
                    break

        metadata = {
            'method': 'all_parcels',
            'n_parcels': n_parcels,
            'network_parcel_counts': {net: len(p) for net, p in network_parcels.items()},
            'labels': labels,
        }
        return all_timeseries, network_names, metadata

    # Aggregate by network
    network_timeseries = np.zeros((T, len(SCHAEFER_7_NETWORKS)))
    network_names = []

    for i, network in enumerate(SCHAEFER_7_NETWORKS):
        parcel_indices = network_parcels[network]
        short_name = NETWORK_SHORT_NAMES[network]
        network_names.append(short_name)

        if len(parcel_indices) == 0:
            print(f"  Warning: No parcels found for {network}")
            continue

        network_data = all_timeseries[:, parcel_indices]  # (T, n_parcels_in_network)

        if method == 'mean':
            # Average across parcels
            network_timeseries[:, i] = np.mean(network_data, axis=1)
        elif method == 'pca':
            # First principal component
            from sklearn.decomposition import PCA
            pca = PCA(n_components=1)
            network_timeseries[:, i] = pca.fit_transform(network_data).flatten()
        else:
            raise ValueError(f"Unknown method: {method}. Use 'mean', 'pca', or 'all'")

    metadata = {
        'method': method,
        'n_parcels': n_parcels,
        'network_parcel_counts': {net: len(p) for net, p in network_parcels.items()},
    }

    print(f"  Network timeseries shape: {network_timeseries.shape}")
    return network_timeseries, network_names, metadata


def extract_roi_timeseries(bold_file: Path,
                           roi: str = 'AG',
                           n_parcels: int = 200,
                           standardize: bool = True,
                           detrend: bool = True,
                           low_pass: float = 0.1,
                           high_pass: float = 0.01,
                           t_r: float = 1.5) -> np.ndarray:
    """
    특정 ROI의 시계열 추출 (Baldassano 2017 스타일)
    """
    if not NILEARN_AVAILABLE:
        raise ImportError("nilearn required. pip install nilearn")

    print(f"  Loading BOLD data...")
    bold_img = nib.load(bold_file)

    # Get Schaefer atlas
    print(f"  Fetching Schaefer {n_parcels} atlas...")
    schaefer = datasets.fetch_atlas_schaefer_2018(
        n_rois=n_parcels,
        yeo_networks=7,
        resolution_mm=2,
        verbose=0
    )

    # Get ROI indices
    if roi not in SCHAEFER_ROI_MAPPING:
        raise ValueError(f"Unknown ROI: {roi}. Available: {list(SCHAEFER_ROI_MAPPING.keys())}")

    roi_info = SCHAEFER_ROI_MAPPING[roi]
    roi_indices = roi_info[f'schaefer_ids_{n_parcels}']
    print(f"  ROI '{roi}': using {len(roi_indices)} parcels")

    # Create masker
    masker = NiftiLabelsMasker(
        labels_img=schaefer['maps'],
        standardize=standardize,
        detrend=detrend,
        low_pass=low_pass,
        high_pass=high_pass,
        t_r=t_r,
        memory='nilearn_cache',
        verbose=0
    )

    # Extract all parcels
    all_timeseries = masker.fit_transform(bold_img)  # (T, n_parcels)

    # Select ROI parcels
    valid_indices = [i for i in roi_indices if i < all_timeseries.shape[1]]
    roi_timeseries = all_timeseries[:, valid_indices]

    print(f"  Extracted: {roi_timeseries.shape} (T x V)")
    return roi_timeseries


def extract_all_parcels(bold_file: Path,
                        n_parcels: int = 200,
                        standardize: bool = True,
                        detrend: bool = True,
                        low_pass: float = 0.1,
                        high_pass: float = 0.01,
                        t_r: float = 1.5) -> Tuple[np.ndarray, List[str]]:
    """
    전체 파셀의 시계열 추출
    """
    if not NILEARN_AVAILABLE:
        raise ImportError("nilearn required. pip install nilearn")

    print(f"  Loading BOLD data...")
    bold_img = nib.load(bold_file)

    # Get atlas
    print(f"  Fetching Schaefer {n_parcels} atlas...")
    schaefer = datasets.fetch_atlas_schaefer_2018(
        n_rois=n_parcels,
        yeo_networks=7,
        resolution_mm=2,
        verbose=0
    )

    # Create masker
    masker = NiftiLabelsMasker(
        labels_img=schaefer['maps'],
        standardize=standardize,
        detrend=detrend,
        low_pass=low_pass,
        high_pass=high_pass,
        t_r=t_r,
        memory='nilearn_cache',
        verbose=0
    )

    # Extract
    timeseries = masker.fit_transform(bold_img)
    labels = [l.decode() if isinstance(l, bytes) else str(l)
              for l in schaefer['labels']]

    print(f"  Extracted: {timeseries.shape}")
    return timeseries, labels


def process_subjects_networks(bids_dir: Path,
                              output_dir: Path,
                              task: str = 'sherlockPart1',
                              n_parcels: int = 200,
                              method: str = 'mean',
                              subjects: Optional[List[str]] = None) -> Dict:
    """
    모든 피험자에 대해 7 Networks 시계열 추출
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"Processing Sherlock BIDS Data - 7 Networks Mode")
    print(f"{'='*60}")
    print(f"BIDS dir: {bids_dir}")
    print(f"Task: {task}")
    print(f"Parcels: {n_parcels}")
    print(f"Method: {method}")
    print(f"Output: {output_dir}")

    bids_data = load_sherlock_bids(bids_dir, task, subjects)

    if not bids_data:
        print("ERROR: No data found!")
        return {}

    results = {}
    network_names = None
    all_metadata = None

    for subj_id, subj_data in bids_data.items():
        print(f"\n--- {subj_id} ---")
        try:
            timeseries, names, metadata = extract_network_timeseries(
                subj_data['bold_file'],
                n_parcels=n_parcels,
                method=method,
                t_r=1.5  # Sherlock TR
            )
            results[subj_id] = timeseries
            network_names = names
            all_metadata = metadata
            print(f"  SUCCESS: {timeseries.shape}")
        except Exception as e:
            print(f"  FAILED: {e}")
            import traceback
            traceback.print_exc()

    # Save results
    if results:
        # BSDS format: List of (D x T) arrays
        data_list = []
        for subj_id in sorted(results.keys()):
            # Transpose to (D x T) for BSDS
            data_list.append(results[subj_id].T)

        # Save combined data
        output_file = output_dir / f"sherlock_{task}_7networks_{method}.npy"
        np.save(output_file, np.array(data_list, dtype=object), allow_pickle=True)
        print(f"\nSaved: {output_file}")
        print(f"  {len(data_list)} subjects")
        print(f"  Shape per subject: {data_list[0].shape} (D x T)")
        print(f"  Networks: {network_names}")

        # Save individual subjects
        subj_dir = output_dir / 'subjects'
        subj_dir.mkdir(exist_ok=True)
        for subj_id, ts in results.items():
            np.save(subj_dir / f"{subj_id}_7networks.npy", ts)

        # Save metadata
        import json
        full_metadata = {
            'task': task,
            'mode': 'networks',
            'n_parcels': n_parcels,
            'method': method,
            'n_subjects': len(results),
            'subjects': list(sorted(results.keys())),
            'network_names': network_names,
            'shapes': {k: list(v.shape) for k, v in results.items()},
            'TR': 1.5,
            'bids_dir': str(bids_dir),
            'preprocessing': {
                'standardize': True,
                'detrend': True,
                'low_pass': 0.1,
                'high_pass': 0.01,
            },
            **all_metadata
        }
        with open(output_dir / 'metadata.json', 'w') as f:
            json.dump(full_metadata, f, indent=2)

        print(f"\nMetadata saved to {output_dir / 'metadata.json'}")

    return results


def process_subjects_roi(bids_dir: Path,
                         output_dir: Path,
                         task: str = 'sherlockPart1',
                         roi: str = 'AG',
                         n_parcels: int = 200,
                         subjects: Optional[List[str]] = None) -> Dict:
    """
    모든 피험자에 대해 특정 ROI 시계열 추출
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"Processing Sherlock BIDS Data - ROI Mode")
    print(f"{'='*60}")
    print(f"BIDS dir: {bids_dir}")
    print(f"Task: {task}")
    print(f"ROI: {roi}")
    print(f"Output: {output_dir}")

    bids_data = load_sherlock_bids(bids_dir, task, subjects)

    if not bids_data:
        print("ERROR: No data found!")
        return {}

    results = {}

    for subj_id, subj_data in bids_data.items():
        print(f"\n--- {subj_id} ---")
        try:
            timeseries = extract_roi_timeseries(
                subj_data['bold_file'],
                roi=roi,
                n_parcels=n_parcels,
                t_r=1.5
            )
            results[subj_id] = timeseries
            print(f"  SUCCESS: {timeseries.shape}")
        except Exception as e:
            print(f"  FAILED: {e}")

    # Save results
    if results:
        data_list = []
        for subj_id in sorted(results.keys()):
            data_list.append(results[subj_id].T)  # (D x T)

        output_file = output_dir / f"sherlock_{task}_{roi}.npy"
        np.save(output_file, np.array(data_list, dtype=object), allow_pickle=True)
        print(f"\nSaved: {output_file}")
        print(f"  {len(data_list)} subjects, shape: {data_list[0].shape}")

        # Save individual subjects
        subj_dir = output_dir / 'subjects'
        subj_dir.mkdir(exist_ok=True)
        for subj_id, ts in results.items():
            np.save(subj_dir / f"{subj_id}_{roi}.npy", ts)

        # Save metadata
        import json
        metadata = {
            'task': task,
            'mode': 'roi',
            'roi': roi,
            'n_parcels': n_parcels,
            'n_subjects': len(results),
            'subjects': list(sorted(results.keys())),
            'shapes': {k: list(v.shape) for k, v in results.items()},
            'TR': 1.5,
            'bids_dir': str(bids_dir),
        }
        with open(output_dir / 'metadata.json', 'w') as f:
            json.dump(metadata, f, indent=2)

    return results


def process_subjects_parcels(bids_dir: Path,
                             output_dir: Path,
                             task: str = 'sherlockPart1',
                             n_parcels: int = 200,
                             subjects: Optional[List[str]] = None) -> Dict:
    """
    모든 피험자에 대해 전체 파셀 시계열 추출
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"Processing Sherlock BIDS Data - All Parcels Mode")
    print(f"{'='*60}")
    print(f"BIDS dir: {bids_dir}")
    print(f"Task: {task}")
    print(f"Parcels: {n_parcels}")
    print(f"Output: {output_dir}")

    bids_data = load_sherlock_bids(bids_dir, task, subjects)

    if not bids_data:
        print("ERROR: No data found!")
        return {}

    results = {}
    all_labels = None

    for subj_id, subj_data in bids_data.items():
        print(f"\n--- {subj_id} ---")
        try:
            timeseries, labels = extract_all_parcels(
                subj_data['bold_file'],
                n_parcels=n_parcels,
                t_r=1.5
            )
            results[subj_id] = timeseries
            all_labels = labels
            print(f"  SUCCESS: {timeseries.shape}")
        except Exception as e:
            print(f"  FAILED: {e}")

    # Save results
    if results:
        data_list = []
        for subj_id in sorted(results.keys()):
            data_list.append(results[subj_id].T)

        output_file = output_dir / f"sherlock_{task}_parcels{n_parcels}.npy"
        np.save(output_file, np.array(data_list, dtype=object), allow_pickle=True)
        print(f"\nSaved: {output_file}")
        print(f"  {len(data_list)} subjects, shape: {data_list[0].shape}")

        # Save individual subjects
        subj_dir = output_dir / 'subjects'
        subj_dir.mkdir(exist_ok=True)
        for subj_id, ts in results.items():
            np.save(subj_dir / f"{subj_id}_parcels.npy", ts)

        # Save labels
        np.save(output_dir / 'parcel_labels.npy', all_labels)

        # Save metadata
        import json
        metadata = {
            'task': task,
            'mode': 'parcels',
            'n_parcels': n_parcels,
            'n_subjects': len(results),
            'subjects': list(sorted(results.keys())),
            'shapes': {k: list(v.shape) for k, v in results.items()},
            'TR': 1.5,
            'bids_dir': str(bids_dir),
        }
        with open(output_dir / 'metadata.json', 'w') as f:
            json.dump(metadata, f, indent=2)

    return results


def main():
    parser = argparse.ArgumentParser(
        description='Preprocess Sherlock BIDS data for BSDS/HMM analysis',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Extract 7 Networks (recommended for BSDS)
  python preprocess_sherlock_bids.py --bids-dir /storage/bigdata/Sherlock \\
      --output-dir ./data/sherlock_7networks --mode networks

  # Extract with PCA instead of mean
  python preprocess_sherlock_bids.py --bids-dir /storage/bigdata/Sherlock \\
      --output-dir ./data/sherlock_7networks_pca --mode networks --method pca

  # Extract specific ROI
  python preprocess_sherlock_bids.py --bids-dir /storage/bigdata/Sherlock \\
      --output-dir ./data/sherlock_AG --mode roi --roi AG
        """
    )

    parser.add_argument('--bids-dir', type=str, required=True,
                        help='BIDS data directory (e.g., /storage/bigdata/Sherlock)')
    parser.add_argument('--output-dir', type=str, default='./data/sherlock_preprocessed',
                        help='Output directory')
    parser.add_argument('--task', type=str, default='sherlockPart1',
                        choices=['sherlockPart1', 'sherlockPart2', 'freerecall'],
                        help='Task to process')
    parser.add_argument('--mode', type=str, default='networks',
                        choices=['networks', 'roi', 'parcels'],
                        help='Extraction mode: networks (7 Yeo networks), roi (specific ROI), parcels (all)')
    parser.add_argument('--method', type=str, default='mean',
                        choices=['mean', 'pca', 'all'],
                        help='Aggregation method for networks mode')
    parser.add_argument('--roi', type=str, default='AG',
                        choices=['AG', 'PMC', 'PCC', 'EAC', 'V1', 'mPFC'],
                        help='ROI to extract (for roi mode)')
    parser.add_argument('--n-parcels', type=int, default=200,
                        choices=[100, 200, 400, 600, 800, 1000],
                        help='Number of Schaefer parcels')
    parser.add_argument('--subjects', type=str, nargs='+', default=None,
                        help='Specific subjects to process (e.g., sub-01 sub-02)')

    args = parser.parse_args()

    if not NILEARN_AVAILABLE:
        print("ERROR: nilearn is required. Install with: pip install nilearn")
        return

    # Process based on mode
    if args.mode == 'networks':
        process_subjects_networks(
            Path(args.bids_dir),
            Path(args.output_dir),
            task=args.task,
            n_parcels=args.n_parcels,
            method=args.method,
            subjects=args.subjects
        )
    elif args.mode == 'roi':
        process_subjects_roi(
            Path(args.bids_dir),
            Path(args.output_dir),
            task=args.task,
            roi=args.roi,
            n_parcels=args.n_parcels,
            subjects=args.subjects
        )
    elif args.mode == 'parcels':
        process_subjects_parcels(
            Path(args.bids_dir),
            Path(args.output_dir),
            task=args.task,
            n_parcels=args.n_parcels,
            subjects=args.subjects
        )


if __name__ == '__main__':
    main()
