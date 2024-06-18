import argparse
from collections import defaultdict
import json
import os


def get_stats(input_outfit_data: list, input_meta_data: dict):
    outfit_data = input_outfit_data
    meta_data = input_meta_data
    type2im = defaultdict(set)
    type2finegrain = defaultdict(set)
    finegrain2type = defaultdict(set)
    finegrain2im = defaultdict(set)

    im2type = {}
    im2finegrain = {}
    full_outfit_set_data = {}
    for outfit in outfit_data:
        outfit_id = outfit['set_id']
        full_outfit_set_data[outfit_id] = {}
        for item in outfit['items']:
            im = item['item_id']
            category = meta_data[im]['semantic_category']
            finegrain = meta_data[im]['category_id']
            type2im[category].add(im)
            type2finegrain[category].add(finegrain)
            finegrain2im[finegrain].add(im)
            finegrain2type[finegrain].add(category)
            im2type[im] = category
            im2finegrain[im] = finegrain
            full_outfit_set_data[outfit_id][str(item['index'])] = im
    return type2im, type2finegrain, finegrain2im, finegrain2type, full_outfit_set_data, im2type, im2finegrain


def main(args: argparse.Namespace):
    data_dir = f"{args.datadir}/polyvore_outfits/{args.polyvore_split}"
    meta_data_path = f"{args.datadir}/polyvore_outfits/polyvore_item_metadata.json"

    test_fitb_path = f"{data_dir}/fill_in_blank_test.json"
    test_outfit_path = f"{data_dir}/test.json"
    train_outfit_path = f"{data_dir}/train.json"

    with open(meta_data_path, 'r') as f:
        meta_data = json.load(f)

    with open(test_fitb_path, 'r') as f:
        test_fitb_data = json.load(f)

    with open(test_outfit_path, 'r') as f:
        test_data = json.load(f)

    with open(train_outfit_path, 'r') as f:
        train_data = json.load(f)

    test_type2im, test_type2finegrain, test_finegrain2im, test_finegrain2type, test_full_outfit_set_data, \
        test_im2type, test_im2finegrain = get_stats(test_data, meta_data)
    train_type2im, train_type2finegrain, train_finegrain2im, train_finegrain2type, train_full_outfit_set_data, \
        train_im2type, train_im2finegrain = get_stats(train_data, meta_data)

    all_finegrain2im = defaultdict(set)

    for i, j in test_finegrain2im.items():
        all_finegrain2im[i] = j.union(train_finegrain2im[i])

    finegrain_count = 0
    selected_finegrain = []
    for i, j in all_finegrain2im.items():
        if len(list(j)) >= 3000:
            finegrain_count += 1
            selected_finegrain.append(i)
    print(f"{finegrain_count} out of {len(all_finegrain2im.keys())} fine-grained types are eligible")

    eligible_count = 0
    for i in test_fitb_data:
        outfit_id, index = i['answers'][0].split('_')
        i_img = test_full_outfit_set_data[outfit_id][index]
        i_finegrain = test_im2finegrain[i_img]
        if i_finegrain in selected_finegrain:
            eligible_count += 1
    print(f"{eligible_count} out of {len(test_fitb_data)} FTIB questions are eligible")

    output_dir = f"{data_dir}/rak"
    if not os.path.isdir(output_dir):
        os.mkdir(output_dir)

    for each_finegrain in selected_finegrain:
        ims = list(test_finegrain2im[each_finegrain])
        if len(ims) < 3000:
            diff_count = 3000 - len(ims)
            train_diff = train_finegrain2im[each_finegrain].difference(test_finegrain2im[each_finegrain])
            train_diff = list(train_diff)[:diff_count]
            ims += train_diff
        with open(f"{output_dir}/{each_finegrain}", "w") as f:
            f.write(" ".join(ims))

    with open(f"{output_dir}/selected_finegrain", "w") as f:
        f.write(" ".join(selected_finegrain))

    finegrain2type = {}
    for i, j in train_finegrain2type.items():
        if i in selected_finegrain:
            type_list = list(j)
            assert len(type_list) == 1
            finegrain2type[i] = type_list[0]
    with open(f"{output_dir}/finegrain2type.json", "w") as f:
        json.dump(finegrain2type, f)

    print("Data processing for Recall@k calculation is done")
    return True


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='data preparation for Recall@k calculation')
    parser.add_argument('--polyvore_split', default='nondisjoint', type=str,
                        help='specifies the split of the polyvore data (either disjoint or nondisjoint)')
    parser.add_argument('--datadir', default='../data', type=str,
                        help='directory of the polyvore outfits dataset')
    args = parser.parse_args()

    assert args.polyvore_split in ['nondisjoint', 'disjoint']

    main(args)

