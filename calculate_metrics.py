import argparse
import json
import os

from datetime import datetime
from loguru import logger

from main import read_run_config
from metrics import load_data_for_metrics_calculation, load_model_for_metrics_calculation, \
                calculate_metrics_and_save_result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='slay-net_metrics_calculation')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='enables CUDA training')
    parser.add_argument('--dataloader_workers', type=int, default=0,
                        help='the number of multi-processes for torch dataloader')
    parser.add_argument('--batch_size', type=int, default=20000, metavar='N',
                        help='input batch size for metrics calculation (default: 20000)')
    parser.add_argument('--checkpoint_meta_dir', type=str, default='model_checkpoints.json',
                        help='the paths to the best model for each configuration')
    parser.add_argument('--development_test', type=int, default=0,
                        help='to identify if the run is for development testing (1: yes, 0: no)')
    args = parser.parse_args()

    checkpoint_meta_dir = args.checkpoint_meta_dir
    assert os.path.isfile(checkpoint_meta_dir)

    if not os.path.isdir('logs/metrics_calculation'):
        os.mkdir('logs/metrics_calculation')
        logger.info("the following directory has been created: 'logs/metrics_calculation'")

    with open(checkpoint_meta_dir, 'r') as f:
        checkpoint_meta = json.load(f)
    logger_id = logger.add(f"logs/metrics_calculation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")

    logger.info("Checking if all specified checkpoint paths are valid")
    for _, this_checkpoint_meta in checkpoint_meta.items():
        assert os.path.isfile(this_checkpoint_meta["checkpoint"])
    logger.info("all checkpoint paths are valid")

    for model_name, this_checkpoint_meta in checkpoint_meta.items():
        output_path = f"logs/metrics_calculation/{model_name}.txt"
        if os.path.isfile(output_path):
            logger.info(f"metrics calculation for model '{model_name}' already exist")
            continue
        this_config_path = this_checkpoint_meta["config"]
        logger.info(f"Calculating metrics for model '{model_name}'")
        logger.info(f"The config is read from directory '{this_config_path}'")
        assert os.path.isfile(this_config_path)
        args.config_dir = this_config_path
        read_run_config(args)
        test_dataset = load_data_for_metrics_calculation(args, 'test')
        this_model = load_model_for_metrics_calculation(args, this_checkpoint_meta["checkpoint"])
        calculate_metrics_and_save_result(args, this_model, test_dataset, output_path)

    logger.remove(logger_id)


