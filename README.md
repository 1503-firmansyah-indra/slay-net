# Sliced-Wasserstein Fashion Compatibility Network (Slay-Net) 

# Note
This repo is still work-in-progress


# Model Performance Comparison with Prior Works

| Work                                | Compatibility AUC | FITB Accuracy (%) | R@top-10(%)       | R@top-30(%)       | R@top-50(%)       |
|-------------------------------------|-------------------|-------------------|-------------------|-------------------|-------------------|
| [Vasileva et al., 2018]<sup>1</sup> | 0.86              | 57.83             | 3.5               | 8.56              | 12.66             |
| [Sarkar et al., 2022]<sup>1</sup> | 0.93              | 67.10             | 9.58              | 17.96             | 21.98             |
| [Wang and Zhong, 2023]<sup>1</sup> | **0.956**         | **70.33**         | 10.12             | 19.49             | 26.17             |
| Slay-Net (Ours) | 0.9023 &#177;  0.0098 | 67.78 &#177;  0.56 | **10.64** &#177; 0.51 | **20.44** &#177; 0.57 | **26.62** &#177;  0.67 |
<sup>1</sup> as reported in [Wang and Zhong, 2023]


# References 
* [Sarkar et al., 2022] Sarkar, R., Bodla, N., Vasileva, M., Lin, Y.-L., Beniwal, A., Lu, A., and Medioni, G. (2022). Outfittransformer: Outfit representations for fashion recommendation. In *Proceedings of the IEEE/CVF Conference on computer vision and pattern recognition*, pages 2263–2267.
* [Vasileva et al., 2018] Vasileva, M. I., Plummer, B. A., Dusad, K., Rajpal, S., Kumar, R., and Forsyth, D. (2018). Learning type-aware embeddings for fashion compatibility. In *Proceedings of the European conference on computer vision (ECCV)*, pages 390–405.
* [Wang and Zhong, 2023] Wang, X. and Zhong, Y. (2023). Text-conditioned outfit recommendation with hybrid attention layer. *IEEE Access*.
