# RoBERTa_quantization

> Final project of COMS 6998 Practical Deep Learning System Performance, Columbia University, 2022 Fall\
> By Yiquan Li (UNI: yl4886) and Zezhong Fan (UNI: zf2274)

## Project Description

Nowadays, neural network models have been improved considerably to reach very good performance. As a result, the model sizes have become larger and larger. The state-of-the-art models contain billions of parameters, which leads to much larger training time and memory usage. However, in real world scenarios, applications of deep learning need to be applied on edge devices in which the resources are constrained, and people also need to control the deployment costs of the model. That is when model compression comes into consideration. Model compression is used to shrink a model’s size while maintaining the model’s accuracy. We focus on one of the key model compression methods — model quantization.

In this project, we take RoBERTa as the model for experiments, explore a variety of integer-only quantization methods. Our goal is to compare the performance of the different quantization techniques and combinations, and find a feasible trade-off between the deployment costs and model accuracy in different NLP tasks (CoLA, SST-2, RTE, MNLI).



## Repository Description

## Commands to execute the code

## Results and Observation

In the project, we use the uncompressed pre-trained RoBERTa-base model as our baseline, finetune the model based on 4 GLUE tasks. For MNLI, we run 2 experiments, one for unquantized model, the other for fully quantized model (end-to-end quantization for each parameter and layer). For CoLA, SST-2, and RTE, we run 1 baseline and 5 different quantization experiments. We finally try to evaluate the model performance based on test accuracy, training time, and inference time. The solution architecture is as below:

<img width="1028" alt="截屏2022-12-17 下午4 06 28" src="https://user-images.githubusercontent.com/63526495/208515674-937df826-d4f5-4787-9bf6-ec9f95719a89.png">

In this part, we will display our results and share our observations.

### Effect of Quantization on Accuracy 

Test accuracy for baseline Roberta and fully-quantized Roberta:

![image](https://user-images.githubusercontent.com/63526495/208516530-74465842-eee8-4fa5-96b0-393271e11b34.png)

From the plot, we can see that for MNLI and SST-2, the degradation of accuracy is almost negligible. However, for CoLA and RTE, the degradation of accuracy is obvious. Notice that the dataset size for these tasks are: 100 MB for MNLI, 7MB for SST-2, 2 MB for RTE, and 965 KB for CoLA. So, for a very intense task that takes large dataset, quantized RoBERTa can achieve very similar accuracy as compared to the full-precision baseline. For smaller tasks, the degradation in accuracy could be more significant.

### Effect of Quantization on Training time

Training time for baseline Roberta and fully-quantized Roberta:





