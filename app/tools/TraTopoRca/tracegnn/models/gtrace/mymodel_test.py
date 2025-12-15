import os
import argparse
import warnings

import dgl
import torch
from loguru import logger

try:
    import mltk  # used for ExpConfig instantiation
except Exception as e:
    raise RuntimeError(f"mltk is required to run eval_only: {e}")

from tracegnn.models.gtrace.config import ExpConfig
from tracegnn.models.gtrace.dataset import TestDataset
from tracegnn.models.gtrace.models.mymodel import MyTraceAnomalyModel
from tracegnn.models.gtrace.mymodel_eval import evaluate


def init_seed(config: ExpConfig):
    if getattr(config, 'seed', None) is not None:
        import random
        import numpy as np
        torch.manual_seed(config.seed)
        torch.cuda.manual_seed_all(config.seed)
        torch.backends.cudnn.benchmark = False
        np.random.seed(config.seed)
        random.seed(config.seed)
        os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'


def main():
    warnings.filterwarnings("ignore")

    ap = argparse.ArgumentParser(description='Evaluate existing checkpoint without training')
    ap.add_argument('--model', default=None, help='Path to model checkpoint (state_dict); overrides config.model_path if set')
    ap.add_argument('--report-dir', default=None, help='Override report dir under processed (e.g., reports_1119) or absolute path')
    ap.add_argument('--export-debug', action='store_true', help='Enable extra RCA debug outputs and conflict CSV')
    ap.add_argument('--batch-size', type=int, default=None, help='Override eval batch size')
    ap.add_argument('--dataset', default=None, help='Override dataset name (default from config)')
    ap.add_argument('--test-dataset', default=None, help='Override test split name (default from config)')
    ap.add_argument('--dataset-root', default=None, help='Override dataset root directory (default from config)')
    args = ap.parse_args()

    # Build config via mltk experiment (to keep nested config behavior)
    with mltk.Experiment(ExpConfig) as exp:
        config: ExpConfig = exp.config

        # Optional overrides
        if args.dataset:
            config.dataset = args.dataset
        if args.test_dataset:
            config.test_dataset = args.test_dataset
        if args.dataset_root:
            config.dataset_root_dir = args.dataset_root
        if args.batch_size is not None:
            config.test_batch_size = int(args.batch_size)
        # Optional explicit model path override
        if args.model is not None:
            try:
                config.model_path = args.model
            except Exception:
                pass
        if args.report_dir:
            config.report_dir = args.report_dir
        if args.export_debug:
            try:
                config.RCA.export_debug = True
            except Exception:
                pass

        init_seed(config)
        logger.info(f"Device: {config.device}")
        logger.info(f"Dataset: {config.dataset} (test={config.test_dataset})")

        # Build test dataset/loader
        test_dataset = TestDataset(config)
        test_loader = dgl.dataloading.GraphDataLoader(
            test_dataset,
            batch_size=(config.test_batch_size or config.batch_size),
        )

        # Build model and load checkpoint
        model = MyTraceAnomalyModel(config).to(config.device)
        # Resolve checkpoint path (relative paths are under dataset_root_dir/dataset)
        try:
            rel_ckpt_path = getattr(config, 'model_path', None) or 'model.pth'
        except Exception:
            rel_ckpt_path = 'model.pth'
        if os.path.isabs(rel_ckpt_path):
            ckpt_path = rel_ckpt_path
        else:
            base_dir = os.path.join(config.dataset_root_dir, config.dataset)
            ckpt_path = os.path.join(base_dir, rel_ckpt_path)
        if not os.path.isfile(ckpt_path):
            logger.warning(f"Checkpoint not found: {ckpt_path}. Proceeding without loading state.")
        else:
            state = torch.load(ckpt_path, map_location=config.device)
            model.load_state_dict(state)
            logger.info(f"Loaded checkpoint from {ckpt_path}")

        # Run evaluation (epoch=None indicates final eval)
        evaluate(config, test_loader, model, epoch=None)


if __name__ == '__main__':
    main()
