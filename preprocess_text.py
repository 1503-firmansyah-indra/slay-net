import argparse

from preprocess_utils import convert_text_to_sentence_embedding


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='convert_txt_to_emb')
    parser.add_argument('--img_list_path', type=str, required=True,
                        default='../data/polyvore_outfits/images_name.txt',
                        help='path of list of images (items)')
    parser.add_argument('--metadata_path', type=str, required=True,
                        default='../data/polyvore_outfits/polyvore_item_metadata.json',
                        help='path of meta data of items')
    parser.add_argument('--datadir', type=str, required=True,
                        default='../data/polyvore_outfits/',
                        help='the base directory of the data')
    args = parser.parse_args()

    convert_text_to_sentence_embedding(
        args.img_list_path,
        args.metadata_path,
        args.datadir
    )

