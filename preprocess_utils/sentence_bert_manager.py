import json
import os
import time

import numpy as np
import torch

from sentence_transformers import SentenceTransformer


def get_text_v1(item_list: list, meta_data: dict):
    output = []
    not_found_count = 0
    for each_item in item_list:
        this_text = ''
        if each_item in meta_data.keys():
            this_content = meta_data[each_item]
            if this_content['description'] != '':
                this_text = this_content['description']
            elif this_content['title'] != '':
                this_text = this_content['title']
            elif this_content['url_name'] != '':
                this_text = this_content['url_name']
        else:
            not_found_count += 1
        output.append(this_text)
    print(f"There are {not_found_count} items not found")
    return output


def get_text_v2(item_list: list, meta_data: dict):
    '''
    This function consider only field 'url_name' for the text description of an item

    :param item_list:
    :param meta_data:
    :return:
    '''
    output = []
    not_found_count = 0
    for each_item in item_list:
        this_text = ''
        if each_item in meta_data.keys():
            this_content = meta_data[each_item]
            this_text = this_content['url_name']
        else:
            not_found_count += 1
        output.append(this_text)
    print(f"There are {not_found_count} items not found")
    return output


def convert_text_to_sentence_embedding(item_list_path: str, item_meta_data_path: str, base_data_folder_dir: str,
                                       get_text_version: str = 'v1'):
    assert os.path.isfile(item_list_path)
    assert os.path.isfile(item_meta_data_path)
    assert os.path.isdir(base_data_folder_dir)

    with open(item_list_path, 'r') as f:
        item_list = f.read().split(',')

    with open(item_meta_data_path, 'r') as f:
        item_meta_data = json.load(f)

    print(f"get_text version used: '{get_text_version}'")
    if get_text_version == 'v1':
        text_list = get_text_v1(item_list, item_meta_data)
    elif get_text_version == 'v2':
        text_list = get_text_v2(item_list, item_meta_data)
    else:
        raise Exception("Invalid value for argument 'get_text_version'")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"device used: '{device}'")

    print("Text feature extraction has started")
    model = SentenceTransformer(
        'sentence-transformers/distiluse-base-multilingual-cased-v2',
        device=device
    )

    start_time = time.time()
    text_embeddings = model.encode(text_list)
    output_name = f"text_sentence_bert_{get_text_version}.npy"
    output_path = os.path.join(base_data_folder_dir, output_name)
    np.save(output_path, text_embeddings)
    end_time = time.time()
    print(f"Text Embeddings are saved in '{output_path}'")
    print(f"The process took {round(end_time - start_time, 1)} seconds")
    return True


