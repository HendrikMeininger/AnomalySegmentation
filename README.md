# AnomalySegmentation
 A collection of Methods for unsupervised Anomaly Segmentation

The python package "AnomalySegmentation", based on this code can be downloaded with "pip install anoseg".
Current version (Alpha): 0.0.4 

This project contains code for:

Deep Feature Correspondence (DFC):  
  code is based on https://github.com/YoungGod/DFC  
  paper: https://www.researchgate.net/publication/361590849_Learning_Deep_Feature_Correspondence_for_Unsupervised_Anomaly_Detection_and_Segmentation  

PaDiM: a Patch Distribution Modeling Framework for Anomaly Detection and Localization  
    Code is based on https://github.com/Pangoraw/PaDiM  
    Paper: https://arxiv.org/abs/2011.08785  

Implementation of Sub-Image Anomaly Detection with Deep Pyramid Correspondences (SPADE)  
    Code is based on https://github.com/byungjae89/SPADE-pytorch/tree/master  
    Paper: https://arxiv.org/abs/2005.02357  

Also code for two Anomaly Segmentation Enhancement stragtegies for high resolution images, Self-Ensembling, and  Patch-Divide&Conquer, are included.
A paper on those strategies, as well as an evaluation of the implementations of the three methods (DFC, PaDiM, and SPADE) with and without the Enhancement strategies, will be released in the future.
 
Some examples of how to train, test and evaluate the models can be found in "examples".
 

