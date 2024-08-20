Code for Counterfactual Generative Models for Time-Varying Treatments, KDD 24' (_Shenghao Wu, Wenbin Zhou, Minshuo Chen, Shixiang Zhu_) [https://arxiv.org/abs/2305.15742]

This codepack includes demo program for the MSCVAE and MSDiffusion on 1d synthetic datasets.

- MSCVAE: The Jupyter notebook under `/mscvae` includes a demonstration of using the counterfactual generative model to generate 1-d counterfactual distributions. The notebook can be used to generate results for MSCVAE and CVAE on fully-synthetic datasets. The running time depends on the training size, and is around 3 mins for d=1, 10 mins for d=3, and >20 mins for d=5 using the recommended training size in the notebook.

- MSDiffusion: The python file under `/msdiffusion` includes training scripts for the msdiffusion model on 1-d synthetic dataset. Example training args are included in train.sh. We recommend using a GPU for accelerating the training procedure.

To install the required packages, run

` pip install -r requirements.txt `

