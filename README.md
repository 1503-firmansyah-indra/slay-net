# Slay-Net : Sliced-Wasserstein Fashion Compatibility Network


# Abstract
In recent years, learning outfit compatibility patterns from human-generated fashion outfits has gained attention from both academia and industry due to its importance to the generation of recommendations on fashion e-Commerce platforms.
The researches in this area mainly tackle two relevant tasks; Fashion FITB and Outfit Complementary Item Retrieval (Outfit CIR).
This work presents *Sliced-Wasserstein Fashion Compatibility Network* or *Slay-Net* to tackle these tasks.
In the proposed approach, fashion outfits are modeled as set-structured data to capture the complex relationship between an item and the rest of the items within an outfit can be captured.
*Slay-Net* includes a novel approach to learn to generate the set embedding of a fashion outfit by using an attention-based set encoding and Pooling by Sliced-Wasserstein Embedding (PSWE).
Furthermore, the training of *Slay-Net* follows a curriculum learning framework that includes simultaneous training for binary classification and contrastive learning through multi-task learning.
Among recent related works, *Slay-Net* is able to achieve the best performance in the Outfit CIR task, as measured by the Recall@top-k metric.
Experiments conducted on a real-world dataset shows that Slay-Net improves the performance in Outfit CIR task by up to 27.16%.

# Paper Submission
The paper was accepted at the Strategic and Utility-aware REcommendations (SURE) Workshop at Recsys 2024 in Bari, Italy. 
You view the camera ready version of the paper [here](https://1503-firmansyah-indra.github.io/works/camera_ready-firmansyah_learning-set-embeddings.pdf).

# Running the code
## Installing Required Libraries
- Python version: 3.9.x
- Package manager: Anaconda
- Pytorch: the version that is used is 2.0.1 and if you are using Linux or Windows, you can execute the following
    ```commandline
    conda install pytorch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 pytorch-cuda=11.8 -c pytorch -c nvidia
    ```
- The following command is to install libraries needed for data pre-processing
  ```commandline
  pip install -r requirements_preprocessing.txt
  ```
  
- The following command is to install libraries needed for training
  ```commandline
  pip install -r requirements.txt
  ```
  
- The following command is to install libraries needed to calculate Recall@top-k
  ```commandline
  conda install -c pytorch faiss-cpu=1.8.0
  ```


## Downloading the data

- The data can be downloaded from the Github repo for [Vasileva et al., 2018]: [link to the repo](https://github.com/mvasil/fashion-compatibility) 
- The instructions provided here assumes that the Polyvore outfits data are stored on another folder called "data" outside the folder for this repo, such that the folder structure is as follows:
  ```
    ├── data
    │     └── polyvore_outfits
    └── slay-net
  ```

## Executing the data preprocessing

1. Run the script to generate image feature using FashionCLIP
    ```commandline
    python preprocess_image.py
    ```
   
2. Run the script to generate text feature using SentenceBERT
    ```commandline
    python preprocess_text.py
    ```

3. Run the script to create contrastive learning dataset for training
    ```commandline
    python preprocess_train_data.py --dataset_split train
    ```

4. Run the script to create metadata required for fine-grained negative samping
    ```commandline
    python preprocess_rak.py
    ```
   
## Executing Training

- The training can be run by executing the following command:
    ```commandline
    python main.py --batch_size 512 --config_dir <path to config>
    ```

- To run the training on **disjoint** dataset, the following command can be used:
    ```commandline
    python main.py --batch_size 512 --config_dir train_config/disjoint_proposed_curriculum.yaml
    ```

- To run the training on **nondisjoint** dataset, the following command can be used:
    ```commandline
    python main.py --batch_size 512 --config_dir train_config/nondisjoint_proposed_curriculum.yaml
    ```

- Please note that the commands given above only train the model once using one set of random seeds. 
The results on the paper are the average of 6 training runs with different sets of random seeds.
    - One set of random seeds consists of the random seed for PyTorch and Numpy
    - The sets of random seeds used in the experiments are `[1, 1], [9, 16], [3, 15], [21, 20], [51, 9], [324, 24]`

## Executing Evaluation

- Once the training is done, modify the file `model_checkpoints.json` accordingly by specifying the path of the best checkpoints. (Please check the log accordingly)
- Afterwards, run the following commands:
    ```commandline
    python calculate_metrics.py
    ```
  
    ```commandline
    python calculate_metrics_rak.py
    ```
- The results can be found in the directory `logs/metrics_calculation`


# Note
- The paper was written based on the master thesis of the same name. After the master thesis report had been completed, further experiments and hyper-paremeter optimization were done. 
Hence, some results reported on the paper are different from the ones reported in the master thesis report.
The master thesis report is available for download [here](https://drive.google.com/uc?export=view&id=1ebi9lRB-oy2W5zTkF-MNlVpTy8g5Jf8E)
