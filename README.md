# IMPORTANT !!!

## RGAE

Everyting regarding the default Robust Graph Autoencoder (RGAE) is taken from [Github Link](https://github.com/FGH00292/Hyperspectral-anomaly-detection-with-RGAE). 
This RGAE is based on the papers "Hyperspectral Anomaly Detection With Robust Graph Autoencoders" and "Hyperspectral Anomaly Detection With Robust Graph Autoencoders", both by G. Fan et al.

## KPCA

Everything regarding KPCA is taken from [Github link](https://github.com/xiangyusong19/SSIIFD_Hyperspectral-Anomaly-Detection/tree/main/Demos_full-pixels_detection?fbclid=IwAR16aahWpTO-_kgc1CuVpv9Y1mBGRn716N_U9lbiHi1m2ZSVMDOF14aAD9g).
The authors of this code are X. Song et al.

## Clustering

Everything regarding clustering is take nfrom [Github link](https://github.com/GatorSense/hsi_toolkit).
The code is written by A. Zare et al.

## My contributions

I have only modified certain aspects of the code. These are:
- PCA -> KPCA/Clustering
- SLIC -> Felzenszwalb/Qickshift/Watershed
- No optimizer -> RMSP/Momentum/ADAM

# Hyperspectral anomaly detection
This is the implementation of articles: ["Hyperspectral Anomaly Detection With Robust Graph Autoencoders"](https://ieeexplore.ieee.org/document/9494034) and ["Robust Graph Autoencoder for Hyperspectral Anomaly Detection"](https://ieeexplore.ieee.org/document/9414767).
# Usage
Run "**main.m**" after setting optimal parameters lambda, S and n_hid.
# Description
* **main.m** ---------- main file
  * **RGAE.m** ---------- implementation of the proposed algorithm;
    * **SuperGraph.m** ---------- construction of Laplacian matrix with SuperGraph;
      * **myPCA.m** ---------- PCA implementation;
    * **myRGAE.m** ---------- training of RGAE for hyperspectral anomaly detection;
  * **ROC.m**----------Calculate the AUC value with given detection map.
# Reference
If you find the code helpful, please kindly cite the following papers:
* Plain Text:<br>
  * G. Fan, Y. Ma, X. Mei, F. Fan, J. Huang and J. Ma, "Hyperspectral Anomaly Detection With Robust Graph Autoencoders," in IEEE Transactions on Geoscience and Remote Sensing, doi: 10.1109/TGRS.2021.3097097.<br>
  * G. Fan, Y. Ma, J. Huang, X. Mei and J. Ma, "Robust Graph Autoencoder for Hyperspectral Anomaly Detection," ICASSP 2021 - 2021 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP), 2021, pp. 1830-1834, doi: 10.1109/ICASSP39728.2021.9414767.<br>
* BibTeX:<br>
  * @ARTICLE{9494034,<br>
  author={G. {Fan} and Y. {Ma} and X. {Mei} and F. {Fan} and J. {Huang} and J. {Ma}},<br>
  journal={IEEE Transactions on Geoscience and Remote Sensing},<br>
  title={Hyperspectral Anomaly Detection With Robust Graph Autoencoders},<br>
  year={2021},<br>
  volume={},<br>
  number={},<br>
  pages={1-14}}<br>
  * @INPROCEEDINGS{9414767,<br>
  author={G. {Fan} and Y. {Ma} and J. {Huang} and X. {Mei} and J. {Ma}},<br>
  booktitle={ICASSP 2021 - 2021 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)},<br>
  title={Robust Graph Autoencoder for Hyperspectral Anomaly Detection},<br>
  year={2021},<br>
  volume={},<br>
  number={},<br>
  pages={1830-1834}}<br>
