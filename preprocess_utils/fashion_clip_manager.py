import os
import time

import numpy as np

from fashion_clip.fashion_clip import FashionCLIP


def convert_folder_to_fclip(image_folder_dir: str, base_data_folder_dir: str, development_test: bool = False,
                            batch_size: int = 512):
    assert os.path.isdir(image_folder_dir)
    assert os.path.isdir(base_data_folder_dir)
    img_paths = [os.path.join(image_folder_dir, k) for k in os.listdir(image_folder_dir)]
    img_name_list = [k.split('.jpg')[0] for k in os.listdir(image_folder_dir)]

    if not development_test:
        output_img_name_file = 'images_name.txt'
        output_embeddings_file = 'images_fclip.npy'
        input_batch_size = batch_size
    else:
        output_img_name_file = 'images_name_test.txt'
        img_name_list = img_name_list[:16]
        img_paths = img_paths[:16]
        output_embeddings_file = 'images_fclip_test.npy'
        input_batch_size = 16
    with open(os.path.join(base_data_folder_dir, output_img_name_file), 'w') as f:
        f.write(','.join(img_name_list))

    start_time = time.time()

    fclip = FashionCLIP('fashion-clip')
    image_embeddings = fclip.encode_images(img_paths, batch_size=input_batch_size)
    np.save(os.path.join(base_data_folder_dir, output_embeddings_file), image_embeddings)

    end_time = time.time()
    print(f"Embeddings are saved in '{os.path.join(base_data_folder_dir, output_embeddings_file)}'")
    print(f"The process took {round(end_time - start_time, 1)} seconds")
    return True

