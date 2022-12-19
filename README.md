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
    * **SuperGraph.m** ---------- construction of Laplacian matrix with SuperGraph (Here you choose to use SLIC/Felzeszwalb/Quickshift/Watershed as well as PCA/KPCA/Clustering);
      * **myKPCA.m** ---------- KPCA implementation;
      * **myClustering.m** ---------- Clustering implementation;
    * **myRGAE.m** ---------- training of default RGAE for hyperspectral anomaly detection;
    * **myRGAE_momentum.m** ---------- training of momentum RGAE for hyperspectral anomaly detection;
    * **myRGAE_RMSP.m** ---------- training of RMSP RGAE for hyperspectral anomaly detection;
    * **myRGAE_ADAM.m** ---------- training of ADAM RGAE for hyperspectral anomaly detection;
  * **ROC.m**----------Calculate the AUC value with given detection map.
