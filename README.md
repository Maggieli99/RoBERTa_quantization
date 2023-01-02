# RoBERTa_quantization

> By Yiquan Li (UNI: yl4886) and Zezhong Fan (UNI: zf2274)

## Project Description

Nowadays, neural network models have been improved considerably to reach very good performance. As a result, the model sizes have become larger and larger. The state-of-the-art models contain billions of parameters, which leads to much larger training time and memory usage. However, in real world scenarios, applications of deep learning need to be applied on edge devices in which the resources are constrained, and people also need to control the deployment costs of the model. That is when model compression comes into consideration. Model compression is used to shrink a model’s size while maintaining the model’s accuracy. We focus on one of the key model compression methods — model quantization.

In this project, we take RoBERTa as the model for experiments, explore a variety of integer-only quantization methods. Our goal is to compare the performance of the different quantization techniques and combinations, and find a feasible trade-off between the deployment costs and model accuracy in different NLP tasks (CoLA, SST-2, RTE, MNLI).



## Repository Description

The repo contains the source code of integer quantization of Roberta, how we fine-tuning Roberta base and quantized Roberta as well as how to conduct experiments and vislization of results. 

- `examples` cloned from Fairseq [1] developed by Meta. Include how to run different Models include Roberta and how to preprocess GLUE dataset on Roberta. 
- `fairseq` includes soure of Roberta Architecture. Quantization of different layers are included in `quantization/utils`. 
- `fairseq_cli` include source code of train the model in `train.py`. 
- `scripts` Helper script cloned from Fairseq repository [1].
- `download_glue_data.py` scripts to download glue datasets.
- `ibert_experiment.ipynb` notebook to conduct experiments on Roberta base and quantized models.
- `plot.ipynb` notebook to visualize the results. 
- `run.py` scripts to fine-tuning and train Roberta quantized model.
- `setup.py` install required libraries for experiments.
- `6998_final_presentation.pdf` is the slides used for presentation.


## Commands to execute the code

To download pretained Roberta model, please run 
```bash
wget https://dl.fbaipublicfiles.com/fairseq/models/roberta.base.tar.gz
tar -xvf roberta.base.tar.gz
```

To download GLUE datasets, please run 
```bash
git clone https://github.com/nyu-mll/GLUE-baselines.git
python GLUE-baselines/download_glue_data.py --data_dir glue_data --tasks all
```

To fine-tuning Roberta 
First we need to preprocess GLUE on specific tasks, please run
```bash
bash ./examples/roberta/preprocess_GLUE_tasks.sh glue_data CoLA
```
Then fine-tuning Roberta base model by running:

```bash
CUDA_VISIBLE_DEVICES=0 python run.py --arch roberta_base --task CoLA 
```   

Then fine-tuning quantized Roberta model by running:

``` bash
CUDA_VISIBLE_DEVICES=0 python run.py --arch roberta_base --task CoLA --restore-file checkpoint_best_CoLA.pt --lr 1e-6 --force-dequant none

```
Arguments could be modified. To change arguement, please check arguements in `run.py`
More details of examples of conducting experiments can be found in `ibert_experiment.ipynb`.
    



## Results & Observation

In the project, we use the uncompressed pre-trained RoBERTa-base model as our baseline, finetune the model based on 4 GLUE tasks. For MNLI, we run 2 experiments, one for unquantized model, the other for fully quantized model (end-to-end quantization for each parameter and layer). For CoLA, SST-2, and RTE, we run 1 baseline and 5 different quantization experiments. We finally try to evaluate the model performance based on test accuracy, training time, and inference time. The solution architecture is as below:

<img width="700" alt="截屏2022-12-17 下午4 06 28" src="https://user-images.githubusercontent.com/63526495/208515674-937df826-d4f5-4787-9bf6-ec9f95719a89.png">

In this part, we will display our results and share our observations.

### Effect of Quantization on Accuracy 

Test accuracy for baseline Roberta and fully-quantized Roberta:

![image](https://user-images.githubusercontent.com/63526495/208516530-74465842-eee8-4fa5-96b0-393271e11b34.png)

From the plot, we can see that for MNLI and SST-2, the degradation of accuracy is almost negligible. However, for CoLA and RTE, the degradation of accuracy is obvious. Notice that the dataset size for these tasks are: 100 MB for MNLI, 7MB for SST-2, 2 MB for RTE, and 965 KB for CoLA. So, for a very intense task that takes large dataset, quantized RoBERTa can achieve very similar accuracy as compared to the full-precision baseline. For smaller tasks, the degradation in accuracy could be more significant.

### Effect of Quantization on Training time

Training time for baseline Roberta and fully-quantized Roberta:

![image](https://user-images.githubusercontent.com/63526495/208522284-191c8721-c267-41b2-9c71-05e1a71b5911.png)

From the plot, we can see that the fine-tuning time of fully quantized model is around 5x more than baseline Roberta for MNLI, SST-2, and CoLA. For RTE, quantized model consumes 2.5x of the fine-tuning time than base model. This is because we apply quantization-aware training. It requires multiple forward and backward passes through the network with different quantization levels and thus leads to larger training time.

For specific values of the accuracy and time, please refer to the table below:

<img width="576" alt="Screen Shot 2022-12-17 at 18 09 18" src="https://user-images.githubusercontent.com/63526495/208526679-41f0ed2e-4489-432b-863c-adf78c3e8f8a.png">

### Comparsion of Integer-only Quantization Combinations

The way we perform different quantization combination is to first quantize all the layers and then dequantize the layer we do not want. So, the other than the baseline, the 5 combinations are as follows:
- End-to-end quantization
- Linear quantization
- Linear+GELU+Softmax
- Linear+GELU+LayerNorm 
- Linear+Softmax+LayerNorm

Training time for different quantization combinations and tasks:

![image](https://user-images.githubusercontent.com/63526495/208529800-f6603d18-995f-475b-907b-34b1d3894b52.png)

Inference time for different quantization combinations and tasks:

![image](https://user-images.githubusercontent.com/63526495/208527994-30500e9b-9299-44a2-9c69-6bd6fc6fc6b8.png)

From the training time plot, we can see that all quantized models takes significant more time to train. Among quantized models, the end-to-end quantization takes the longest time, only quantizing linear operations in the network takes the shortest time. For other combinations, Linear+Softmax+LayerNorm (Dequantizing GELU) gives the shortest time. It indicates that quantizing GELU activation function takes the most computing steps comparing to the two other non-linear operations.

As for inference time, we observe that when quantizing more layers, the inference time is larger. This goes against our ituition and rational. We expect the inference time to be smaller when quantizing more. One reason for this phenomenon is that PyTorch does not support integer operations. So, even if in the code we try to store the parameters as integers. When doing the computation, PyTorch still reads the parameters as Float32. Thus, the implementation can not achieve latency reduction. To achieve speedup, we may need to export the integer parameters and model architecture to other frameworks that supports integer computing.

### Training time-Accuracy Tradeoffs

![image](https://user-images.githubusercontent.com/63526495/208532181-4328dc8f-4367-4e5a-a628-ff7bb3947e88.png)

From the plot, we can discover that end-to-end Quant always needs more training time but have lower accuracy than base. Dequanting all nonlinear layer have less training time but relatively low accuracy. We find dequantizing GELU operation seems to be a good trade-off because it achieves a relatively high accuracy compared to other quantization combinations while having a moderate training time.

### Conclusion & Future Work

As a summary of the above observations:

- When training the tasks with very large dataset (e.g., MNLI, SST-2), quantized RoBERTa can achieve very similar accuracy as compared to the full-precision baseline. For smaller datasets (e.g., CoLA, RTE), the degradation in accuracy is more significant.
- Quantization-aware training leads to larger training time because it requires multiple forward and backward passes through the network with different quantization levels; in general, larger datasets leads to larger training time difference between quantized and base models
- Since Pytorch does not support integer operation, the inference times we get have significant bias. So inference on other computing units is needed.
- Only dequantizing GELU operation (Quantize linear+Softmax+LayerNorm) seems to be a good trade-off. In this case, it achieves a relatively high accuracy compared to other quantization combinations while having a moderate training time.

Future work:

- Pytorch does not support integer operation so in order to deploy integer-quantized model on GPU or CPU and achieve speedup during inference, we need export the integer parameters along with the model architecture to other frameworks that support deployment on integer processing units(TVM and TensorRT).
- Manually computing model size base is needed to further evaluate efficiency of quantization.

## References

[1] Fairseq Repo from Facebook Research: [Fairseq Repo](https://github.com/facebookresearch/fairseq)

[2] The project is inspired by the Paper:

```
@article{kim2021bert,
  title={I-BERT: Integer-only BERT Quantization},
  author={Kim, Sehoon and Gholami, Amir and Yao, Zhewei and Mahoney, Michael W and Keutzer, Kurt},
  journal={International Conference on Machine Learning (Accepted)},
  year={2021}
}
```


