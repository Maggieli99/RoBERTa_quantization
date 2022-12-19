# RoBERTa_quantization

> Final project of COMS 6998 Practical Deep Learning System Performance, Columbia University, 2022 Fall\
> By Yiquan Li (UNI: yl4886) and Zezhong Fan (UNI: zf2274)

## Project Description

Nowadays, neural network models have been improved considerably to reach very good performance. As a result, the model sizes have become larger and larger. The state-of-the-art models contain billions of parameters, which leads to much larger training time and memory usage. However, in real world scenarios, applications of deep learning need to be applied on edge devices in which the resources are constrained, and people also need to control the deployment costs of the model. That is when model compression comes into consideration. Model compression is used to shrink a model’s size while maintaining the model’s accuracy. We focus on one of the key model compression methods — model quantization.

In this project, we take RoBERTa as the model for experiments, explore a variety of integer-only quantization methods. Our goal is to compare the performance of the different quantization techniques and combinations, and find a feasible trade-off between the deployment costs and model accuracy in different NLP tasks.


