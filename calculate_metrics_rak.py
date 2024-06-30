import argparse
import copy
from datetime import datetime
import json
import os
import time

import faiss
import faiss.contrib.torch_utils
from loguru import logger
import torch

from main import read_run_config
from metrics import load_model_for_metrics_calculation, load_data_for_metrics_calculation
from polyvore_outfits_set import item_embedding_collation, recall_at_k_collation


def calculate_one_config(args, output_path, input_model, input_dataset, development_test = False):
    start_time = time.time()

    embedding_dim = args.dim_embed_img + args.dim_embed_txt
    input_dataset.prepare_for_inference('embedding')

    finegrain2index = {}
    input_model.eval()
    with torch.no_grad():
        for each_finegrain in input_dataset.selected_finegrain_list:
            input_dataset.set_finegrain(each_finegrain)
            this_faiss_index = faiss.IndexFlatL2(embedding_dim)

            this_dataloader = torch.utils.data.DataLoader(
                input_dataset,
                batch_size=args.batch_size,
                shuffle=False,
                collate_fn=item_embedding_collation
            )

            for _, each_batch in enumerate(this_dataloader):
                img, txt, type_onehot, img_id, finegrain = each_batch
                embedding_tensors = input_model.forward_positive(img, type_onehot, txt=txt)
                this_faiss_index.add(embedding_tensors)
            finegrain2index[each_finegrain] = {
                "img_list": copy.deepcopy(input_dataset.selected_ims),
                "index": this_faiss_index
            }
    input_dataset.prepare_for_inference('recall@k')
    recall_at_k_loader = torch.utils.data.DataLoader(
        input_dataset,
        batch_size=16 if development_test else args.batch_size,
        shuffle=False,
        collate_fn=recall_at_k_collation
    )
    logger.info("executing item retrieval as per FITB questions")

    question_count = 0
    rak_results_raw = []

    input_model.eval()
    with torch.no_grad():
        for _, each_batch in enumerate(recall_at_k_loader):
            q_img, q_onehot, q_mask, q_text, q_outfit_len, a_onehot, a_img_id, a_type, a_finegrain = each_batch
            q_embeddings_tensor = input_model(
                q_img, q_onehot, a_onehot,
                feat_mask=q_mask, set_size=q_outfit_len, txt=q_text)

            for idx in range(q_embeddings_tensor.shape[0]):
                i_finegrain = a_finegrain[idx]
                answer_img_idx = finegrain2index[i_finegrain]['img_list'].index(a_img_id[idx])
                _, result_index = finegrain2index[i_finegrain]['index'].search(
                    torch.index_select(q_embeddings_tensor, 0, torch.tensor(idx)),
                    50
                )
                rak_results_raw.append([
                    1 if answer_img_idx in result_index[0][:10] else 0,
                    1 if answer_img_idx in result_index[0][:30] else 0,
                    1 if answer_img_idx in result_index[0][:50] else 0
                ])
                question_count += 1
                if question_count % 200 == 0:
                    print(question_count, end=', ')
            if development_test:
                break
    rak_results = {
        "10": 0,
        "30": 0,
        "50": 0
    }
    for each_result in rak_results_raw:
        rak_results["10"] += each_result[0]
        rak_results["30"] += each_result[1]
        rak_results["50"] += each_result[2]
    rak_results["10"] = rak_results["10"] / len(rak_results_raw)
    rak_results["30"] = rak_results["30"] / len(rak_results_raw)
    rak_results["50"] = rak_results["50"] / len(rak_results_raw)
    with open(output_path, 'w') as f:
        f.write(
            f"r@k-10: {rak_results['10']}\nr@k-30: {rak_results['30']}\nr@k-50: {rak_results['50']}"
        )
    logger.info(f"results are saved in '{output_path}'")
    end_time = time.time()
    time_taken = end_time - start_time
    logger.info(f"The R@k metrics calculation took: {round(time_taken, 1)} seconds")
    return True


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='slay-net_metrics_calculation_rak')
    parser.add_argument('--dataloader_workers', type=int, default=0,
                        help='the number of multi-processes for torch dataloader')
    parser.add_argument('--batch_size', type=int, default=8192,
                        help='input batch size for metrics calculation (default: 8192)')
    parser.add_argument('--checkpoint_meta_dir', type=str, default='model_checkpoints.json',
                        help='the paths to the best model for each configuration')
    parser.add_argument('--development_test', type=int, default=0,
                        help='to identify if the run is for development testing (1: yes, 0: no)')
    parser.add_argument('--no-cuda', action='store_true', default=True,
                        help='force to use CPU even GPU is detected '
                             '(GPU (hence, CUDA) is used by default when it is detected')
    args = parser.parse_args()

    checkpoint_meta_dir = args.checkpoint_meta_dir
    assert os.path.isfile(checkpoint_meta_dir)
    with open(checkpoint_meta_dir, 'r') as f:
        checkpoint_meta = json.load(f)

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
        if not args.development_test:
            output_path = f"logs/metrics_calculation/rak_{model_name}.txt"
            if os.path.isfile(output_path):
                logger.info(f"metrics calculation for model '{model_name}' already exist")
                continue
        else:
            output_path = f"logs/metrics_calculation/test_rak_{model_name}.txt"

        this_config_path = this_checkpoint_meta["config"]
        logger.info(f"Calculating Recall@top-k metrics for model '{model_name}'")
        logger.info(f"The config is read from directory '{this_config_path}'")
        assert os.path.isfile(this_config_path)
        args.config_dir = this_config_path
        read_run_config(args)

        # Manually set device to cpu as faiss only support cpu on windows
        args.cuda = False
        this_model = load_model_for_metrics_calculation(args, this_checkpoint_meta["checkpoint"])
        item_dataset = load_data_for_metrics_calculation(args, 'test', metric_type='rak')

        calculate_one_config(args, output_path, this_model, item_dataset,
                             development_test=args.development_test)

        if bool(args.development_test):
            break
    logger.remove(logger_id)


