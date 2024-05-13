# Sliced-Wasserstein Fashion Compatibility Network (Slay-Net) 

# Note
This repo is still work-in-progress


# Abstract
In recent years, learning outfit compatibility patterns from human-generated fashion outfits has gained attention from 
both academia and industry due to its importance to the generation of recommendations on fashion e-Commerce platforms. 
The researches in this area mainly tackle three relevant tasks; Outfit Compatibility Prediction, Fashion FITB 
and Outfit Complementary Item Retrieval. 
This master thesis presents *Sliced-Wasserstein Fashion Compatibility Network* or *Slay-Net* to tackle these tasks. 
In the proposed approach, fashion outfits are modeled as set-structure data, so that the complex relationship 
between an item and the rest of the items within an outfit can be captured. Slay-Net includes a novel approach 
to learn to generate the set embedding of a fashion outfit by using an attention-based set encoding and 
Pooling by Sliced-Wasserstein Embedding (PSWE). Furthermore, the training of Slay-Net follows 
a curriculum learning framework that includes simultaneous training for Binary Classification and 
Contrastive Learning through multi-task learning. Among recent related works, Slay-Net is able to achieve 
the best performance in the Outfit Complementary Item Retrieval task, as measured by the Recall@top-k metric.


# Model Performance Comparison with Prior Works

| Work                                | Compatibility AUC | FITB Accuracy (%) | R@top-10(%)       | R@top-30(%)       | R@top-50(%)       |
|-------------------------------------|-------------------|-------------------|-------------------|-------------------|-------------------|
| [Vasileva et al., 2018]<sup>1</sup> | 0.86              | 57.83             | 3.5               | 8.56              | 12.66             |
| [Sarkar et al., 2022]<sup>1</sup> | 0.93              | 67.10             | 9.58              | 17.96             | 21.98             |
| [Wang and Zhong, 2023]<sup>1</sup> | **0.956**         | **70.33**         | 10.12             | 19.49             | 26.17             |
| Slay-Net (Ours) | 0.9023 &#177;  0.0098 | 67.78 &#177;  0.56 | **10.64** &#177; 0.51 | **20.44** &#177; 0.57 | **26.62** &#177;  0.67 |
<sup>1</sup> as reported in [Wang and Zhong, 2023]

Note:
- The data used is the nondisjoint Polyvore Outfits by [Vasileva et al., 2018]
- R@top-10, R@top-30 and R@top-50 stand for Recall@top-10, Recall@top-30 and Recall@top-50, respectively

# References 
* [Sarkar et al., 2022] Sarkar, R., Bodla, N., Vasileva, M., Lin, Y.-L., Beniwal, A., Lu, A., and Medioni, G. (2022). Outfittransformer: Outfit representations for fashion recommendation. In *Proceedings of the IEEE/CVF Conference on computer vision and pattern recognition*, pages 2263–2267.
* [Vasileva et al., 2018] Vasileva, M. I., Plummer, B. A., Dusad, K., Rajpal, S., Kumar, R., and Forsyth, D. (2018). Learning type-aware embeddings for fashion compatibility. In *Proceedings of the European conference on computer vision (ECCV)*, pages 390–405.
* [Wang and Zhong, 2023] Wang, X. and Zhong, Y. (2023). Text-conditioned outfit recommendation with hybrid attention layer. *IEEE Access*.
