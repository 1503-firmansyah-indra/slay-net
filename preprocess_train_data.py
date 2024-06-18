import argparse
from itertools import compress
import json
import os


def main(args):
    data_dir = f"{args.datadir}/polyvore_outfits/{args.polyvore_split}/"
    meta_data_path = f"{args.datadir}/polyvore_outfits/polyvore_item_metadata.json"

    raw_data_path = os.path.join(data_dir, f"{args.dataset_split}.json")

    if args.max_outfit_length >= 19:
        print("Processing full data (with max length of 19)")
        output_set_path = os.path.join(data_dir, f"set_based_{args.dataset_split}_full.json")
    else:
        print(f"Processing data with max length of {args.max_outfit_length}")
        output_set_path = os.path.join(data_dir, f"set_based_{args.dataset_split}_{str(args.max_outfit_length)}.json")

    with open(raw_data_path, 'r') as f:
        raw_data = json.load(f)

    with open(meta_data_path, 'r') as f:
        meta_data = json.load(f)

    set_data_arr = []
    item_types = set()

    for each in raw_data:
        if len(each['items']) > args.max_outfit_length:
            continue
        this_complete_outfit = []
        for item in each['items']:
            this_complete_outfit.append([
                item['item_id'],
                meta_data[item['item_id']]['semantic_category']
            ])
            item_types.add(meta_data[item['item_id']]['semantic_category'])

        for idx, outfit_item in enumerate(this_complete_outfit):
            bool_mask = list(map(lambda x: x != idx, range(len(each['items']))))
            set_data_arr.append([
                each['set_id'],
                outfit_item,
                list(compress(this_complete_outfit, bool_mask))
            ])

    with open(output_set_path, 'w') as outfile:
        json.dump(set_data_arr, outfile)
    print(f"Saving output at '{output_set_path}'")

    if args.dataset_split == 'train':
        output_item_type_path = os.path.join(data_dir, 'item_types.json')
        item_dict = {j: i for i, j in enumerate(item_types)}
        with open(output_item_type_path, 'w') as outfile:
            json.dump(item_dict, outfile)
        print(f"Saving item type dictionary at '{output_item_type_path}'")

        if args.max_outfit_length < 19:
            comp_output_filename = os.path.join(
                data_dir, f"compatibility_train_{str(args.max_outfit_length)}.txt")
            comp_data_path = os.path.join(data_dir, 'compatibility_train.txt')
            comp_small_data = []
            with open(comp_data_path, 'r') as f:
                for line in f.readlines():
                    this_len = len(line.split()) - 1
                    if this_len <= args.max_outfit_length:
                        comp_small_data.append(line)

            with open(comp_output_filename, 'w') as f:
                for output_line in comp_small_data:
                    f.write(output_line)
            print(f"Saving small compatibility training data at '{comp_output_filename}'")
    return True


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='data preparation')
    parser.add_argument('--polyvore_split', default='nondisjoint', type=str,
                        help='specifies the split of the polyvore data (either disjoint or nondisjoint)')
    parser.add_argument('--datadir', default='../data', type=str,
                        help='directory of the polyvore outfits dataset')
    parser.add_argument('--dataset_split', default='train', type=str,
                        help='specifies the dataset split to be processed (train, valid or test)')
    parser.add_argument('--max_outfit_length', default=20, type=int,
                        help='the max length outfit that will be included in the outfit set data')
    args = parser.parse_args()

    assert args.polyvore_split in ['nondisjoint', 'disjoint']
    assert args.dataset_split in ['train', 'valid', 'test']

    main(args)

