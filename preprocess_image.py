import argparse

from preprocess_utils import convert_folder_to_fclip


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='convert_img_to_fclip')
    parser.add_argument('--imgdir', type=str, required=True,
                        default='../data/polyvore_outfits/images',
                        help='directory of images to be converted to FashionCLIP embeddings')
    parser.add_argument('--datadir', type=str, required=True,
                        default='../data/polyvore_outfits',
                        help='the base directory of the data')
    parser.add_argument('--development_test', type=int, default=0, metavar='N',
                        help='to identify if the run is for development testing (1: yes, 0: no)')
    args = parser.parse_args()

    convert_folder_to_fclip(
        args.imgdir,
        args.datadir,
        bool(args.development_test))

