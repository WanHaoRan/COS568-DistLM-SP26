# COS 568: Systems and Machine Learning
# Assignment 2: Distributed Training of Language Models


This assignment is designed to familiarize you with the basics of training language models with a toy example on a small scale. You will first get hands-on experience with fine-tuning pre-trained language models like BERT (implemented in PyTorch and Huggingface Transformers) on a single machine. Then, you will implement data parallelism, the most common paradigm of distributed training, using primitives from PyTorch Distributed and perform distributed training on multiple machines. You will understand the tradeoffs of different approaches of performing distributed training.


## Learning Outcomes

- Load a pretrained language model from Huggingface and fine-tune it on RTE datasets
- Write distributed training applications with PyTorch distributed
- Understand the trade-off between different methods of performing distributed training
- Describe how ML frameworks (PyTorch) interact with lower-level communication frameworks (e.g. OpenMPI, Gloo, and NCCL)


## Environment Setup

You may choose to either use GPUs or CPUs for this project (**Using CPUs via CloudLab is highly recommended and our codebase is only validated on CloudLab**). You are required to use up to 4 nodes in this assignment, and the nodes should be able to communicate with each other through a network interface. The nodes should have at least ~12 GB of RAM (in case of CPU training) or GPU memory (in case of GPU training) and at least ~10G of disk space.

**The following setup process caters to CPU-only environments.** Note that this assignment is best done if you have bare-metal access to your compute nodes (i.e. you can access the nodes using a terminal not through a Slurm scheduler). **We highly recommend you use CPU nodes on CloudLab** -- For instructions on accessing CloudLab, see [cloudlab.md](cloudlab.md).

- Create a fork of this repository and clone your own fork, please clone your repo on the path that is shared across all the nodes (e.g. each node's home directory). **Again, please read [cloudlab.md](cloudlab.md) carefully before you clone the repo and start the experiments!**
- Install software dependencies via apt and install pip dependencies **for each node**: 
  ```bash
  sudo apt-get update
  sudo apt-get install htop dstat python3-pip
  echo 'export PATH=$HOME/.local/bin:$PATH' >> ~/.bashrc
  source ~/.bashrc

  pip install torch==2.5.0 torchvision==0.20.0 torchaudio==2.5.0 --index-url https://download.pytorch.org/whl/cpu
  pip install numpy scipy scikit-learn tqdm pytorch_transformers apex
  ```
  - If you are using GPUs, install the appropriate PyTorch [here](https://pytorch.org/get-started/locally/).
-  You are only required to fine-tune RTE  (one of the datasets within GLUE) in this assignment, use the python script `download_glue_data.py` to download the dataset. If you are not running experiments via CloudLab, use this command to download datasets within GLUE: 
    ```bash
    mkdir glue_data
    python3 download_glue_data.py --data_dir glue_data
    ```


## Part 1: Fine-tuning BERT on GLUE Datasets on a Single Node

We have provided a base script ([`run_glue_skeleton.py`](run_glue_skeleton.py)) for you to get started, which provides some model/dataset setup and training setup to fine-tune BERT-base on the RTE dataset. The base script bakes in a handful of optimizations such as mixed-precision training and gradient accumulation -- we will not be needing them, and you can ignore the related code. We also provide `utils_glue.py` which includes some helper functions.

Here is a command you can use to run `run_glue.py` (remember to rename the `run_glue_skeleton.py` to `run_glue.py` after adding the necessary code):

```shell
export GLUE_DIR=$HOME/glue_data
export TASK_NAME=RTE

python3 run_glue.py \
  --model_type bert \
  --model_name_or_path bert-base-cased \
  --task_name $TASK_NAME \
  --do_train \
  --do_eval \
  --data_dir $GLUE_DIR/$TASK_NAME \
  --max_seq_length 128 \
  --per_device_train_batch_size 64 \
  --learning_rate 2e-5 \
  --num_train_epochs 3 \
  --output_dir /tmp/$TASK_NAME/ \
  --overwrite_output_dir
```

**Task 1:** Load the pretrained BERT-base model from Huggingface and fill in the standard training loop of a minibatch – forward pass (already implemented), backward pass, loss computation (already implemented), and parameter updates (via optimizer step). Record the loss values of the first five minibatches by printing the loss value after every iteration. Afterward, run training for 3 epochs (an epoch is a complete pass over a dataset -- when doing fine-tuning, we typically only need a small number of epochs) with batch size 64 and the default hyperparameters. All required code changes are marked with `TODO(cos568)` comments. 

There are several examples for training that describe these four steps. Some good resources include the [PyTorch examples repository](https://github.com/pytorch/examples) and the [Pytorch tutorials](https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html). This script is also a starting point for later parts of the assignment. Familiarize yourself with the script and run training on a single machine.


## Part 2: Distributed Data Parallel Training

Next, you will modify the script used in Part 1 to enable distributed data-parallel training. There are primarily two ways distributed training is performed: i) Data Parallel, ii) Model Parallel. In the case of Data parallel, each of the participating workers trains the same network, but on different data points from the dataset. After each iteration (forward and backward pass), the workers average their local gradients to come up with a single update. In model parallel training, the model is partitioned among a number of workers. Each worker performs training on part of the model and sends its output to the worker which has the next partition during the forward pass and vice-versa in the backward pass. Model parallel is usually used when the size of the network is very large and doesn’t fit on a single worker. In this assignment, we solely focus on Data Parallel Training. For more information about parallelism in ML, see the FAQs. For data parallel training, you will need to partition the data among other nodes. Look at the FAQs to find details on how to partition the data.

To understand more about the communication primitives in ML, please refer to these two articles: [PyTorch documentation](https://pytorch.org/tutorials/intermediate/dist_tuto.html#collective-communication), [NCCL documentation](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/usage/collectives.html). We will be using `gather`, `scatter`, and `all_reduce` in this assignment.

### Part 2(a): Gradient Synchronization with gather and scatter

PyTorch comes with the `gloo` and `nccl` communication backend built-in. On CPU nodes, we will use `gloo` to implement gradient aggregation using the `torch.distributed.gather` and `torch.distributed.scatter` calls. On GPU nodes, use the `nccl` backend.

We will do the following:

1. Set up PyTorch in distributed mode using [PyTorch Distributed](https://pytorch.org/docs/stable/distributed.html). Initialize the distributed environment using [init_process_group()](https://pytorch.org/docs/stable/distributed.html#torch.distributed.init_process_group), which only needs to be invoked once. You are encouraged to write a small test script to make sure you can initialize a process group on four nodes before proceeding. For details of the initialization process, see the FAQs.
2. Next, to perform gradient aggregation, you will need to read the gradients after the backward pass. PyTorch performs gradient computation using auto grad when you call `.backward` on a computation graph. The gradient is stored in the `.grad` attribute of the parameters. The parameters can be accessed using APIs like `model.parameters()` or `model.named_parameters()`.
3. Finally, to perform gradient aggregation, you will need to use `torch.distributed.gather` and `torch.distributed.scatter` communication collectives. Specifically, worker 0 (with rank 0) in the group will gather the gradients from all the participating workers, perform element-wise averaging, and then scatter the mean vector to all workers. The workers update the `grad` variable with the received vector and then continue training. 

**Task 2(a):** Implement gradient synchronization using `torch.distributed.gather` and `torch.distributed.scatter` in `gloo` or `nccl`. Verify that you are using the same total batch size, where total batch size = batch size on one worker * number of workers. With the same total batch size and the same seed, you should get similar loss values as in the single-node training case. Remember that you trained with a total batch size of 64 in Task 1.

[Hint] To start a distributed training job, you need to start the program on each node by running something like 
```shell
python run_glue.py [other input args] --master_ip $ip_address$ --master_port $port$ --world_size 4 --local_rank $rank$
```
Here `local_rank=0, 1, 2, 3`, which corresponds to your node ID. We recommend using `tmux` to keep the session persistent.


### Part 2(b): Gradient Synchronization with all_reduce

Ring AllReduce is an extremely scalable technique for performing gradient synchronization. Read [here](https://tech.preferred.jp/en/blog/technologies-behind-distributed-deep-learning-allreduce/) for more intuition and visualization of Ring Allreduce (and how it has been applied to distributed ML). Instead of using the `torch.distributed.gather` and `torch.distributed.scatter` collectives separately, you will next use the built-in `torch.distributed.all_reduce` collective to sync gradients among different nodes. Again, read the gradients after the backward pass and perform `torch.distributed.all_reduce` on the gradients. Note that the PyTorch `torch.distributed.all_reduce` call does not have an "average" mode, so you will need to use the `sum` operation and then get the average on each node by dividing the sum by the number of workers. After averaging, update the gradients of the model as in the previous part.

**Task 2(b):** Implement gradient synchronization using the `torch.distributed.all_reduce` collective in `gloo` or `nccl`. In this case, if you have set the same random seed while using the same total batch size, you should see the same loss values as in Task 2(a).


## Part 3: Distributed Data Parallel Training with PyTorch DistributedDataParallel

Now, instead of writing your own gradient synchronization, use the distributed functionality provided by PyTorch. 

**Task 3:** Register your model with [distributed data parallel](https://pytorch.org/docs/master/generated/torch.nn.parallel.DistributedDataParallel.html#torch.nn.parallel.DistributedDataParallel) and perform distributed training. Unlike in Part 2, you will not need to read the gradients for each layer as DistributedDataParallel performs these steps automatically. For more details, read [here](https://pytorch.org/tutorials/intermediate/ddp_tutorial.html).

## Part 4: Benchmarking Data Parallel Training

In this section, your task is to benchmark and profile the communication aspect of data parallel training and analyze the synchronization overhead for three communication methods (`gather/scatter`, `all_reduce`, and `DDP`). Similar to A1, you should use the `torch.profiler.profile` to record the profiling data during the training. In this part, you are required to profile **three training steps** for each communication method. Please use `torch.profiler.schedule()` to schedule the profiling, and skip the first training step. Here is an example of the profiling result on a CPU cluster:
![Profiler Result](figures/dist_training.svg)

**Task 4:** Add the profiling code to your training code in Task 2(a), 2(b) and 3, and save the profiling data of three training steps to a chrome trace file. Then, use the chrome browser to visualize the profiling data and analyze the communication overhead. 

## Deliverables

A *brief* report in PDF format (filename: `$NetID$_$firstname$_$lastname$.pdf`) and the code for each task. Please compress your source code into a .zip file and upload it to Canvas.

In the report, include the following content:
- Task 1: Run experiements for 3 epochs and report the evaluation metric after every epoch.
- Task 2, 3: Run each task for 1 epoch rather than 3 epochs. Discard the timings of the first iteration and report the average time per iteration for the remaining iterations for each task (2(a), 2(b), and 3). To time your code, see [this](https://realpython.com/python-timer/). Log the loss curve for each node (The loss curve should be *exactly* the same for Task 2(a) and 2(b))
- Task 4: Report the profiling data of three training steps for each communication method and analyze the communication overhead. Skip the first training step. Report the percentage of the communication time overhead out of the total time for each step. Compare the `all_reduce` and `DDP` methods, and discuss why `DDP` is more efficient, and in what percentage.
- Reason about the difference (or the lack of difference) among different setups. Feel free to refer to the [PyTorch Distributed [VLDB '18]](https://arxiv.org/pdf/2006.15704.pdf) paper for more context.
- Comment on the scalability of distributed ML based on your results.
- Provide any implementation details.
- Code for each task should be in different directories: `task1`, `task2a`, `task2b`, `task3` and `task4`.
- All your code that requires PyTorch distributed should be runnable using the following command:

```shell
python run_glue.py [other input args] --master_ip $ip_address$ --master_port $port$ --world_size 4 --local_rank $rank$
```


## Common FAQs and Resources

- **What is BERT?** BERT, which stands for Bidirectional Encoder Representations from Transformers, is an NLP model introduced by Google in 2018. It is an encoder-only model, and it is capable of various NLP tasks, such as question answering, sentiment analysis, and language translation. For more information, check this out: [A Visual Guide to Using BERT for the First Time
](https://jalammar.github.io/a-visual-guide-to-using-bert-for-the-first-time/). If you want more insights into which parameters we are doing gradient synchronization on, you can use a model profiler (e.g. [PyTorch Profiler](https://pytorch.org/tutorials/recipes/recipes/profiler_recipe.html), [DeepSpeed Profiler](https://www.deepspeed.ai/tutorials/flops-profiler/)) to inspect the model. Some profilers may require a GPU to run.

- **What is GLUE?** The General Language Understanding Evaluation benchmark (GLUE) is a collection of datasets used for training, evaluating, and analyzing NLP models relative to one another. The collection consists of nine difficult and diverse task datasets designed to test a model’s language understanding. For more information, check this out: [GLUE Explained: Understanding BERT Through Benchmarks](https://mccormickml.com/2019/11/05/GLUE/)

- **What is fine-tuning?** Fine-tuning in the context of NLP refers to the process of taking a pre-trained model (trained on a huge text corpus) and further training it on a specific dataset to tailor it for a particular task.

- **Testing programs:** We suggest you write small programs to test the functionality of communication collectives at first. For example, create a tensor and send it to another node. Next, try to perform all reduce on it. This will help you get comfortable with communication collectives.

- **Setting up distributed init:** Instead of a shared file system, we want to use TCP to init and communicate. You can use `ifconfig` to find the IP address of your nodes. On CloudLab, please use the experimental network, which listens on `10.10.1.*` interfaces (sending a huge amount of traffic through the other network interfaces may lead to your reservation getting terminated by the CloudLab admins). You can double-check network connectivity by `ping`ing a node from another. The port has to be a non-privileged port, i.e. greater than 1023.

  An example of `torch.distributed.init_prcess_group()` for CloudLab:
  ```python
  torch.distributed.init_process_group(
              backend='gloo', # for cpu
              init_method=f"tcp://10.10.1.2:12345", # "tcp://{master_ip}:{master_port}"
              world_size=args.world_size, # Number of nodes (4 in our experiments)
              rank=args.local_rank, # 0, 1, 2, 3
          )
  ```


- **Data Partitioning:** In the case of data parallel training, the workers train on non-overlapping data partitions. You will use the distributed sampler to distribute the data among workers. For more details, look at [torch.utils.data.distributed_sampler](https://pytorch.org/docs/stable/data.html#torch.utils.data.distributed.DistributedSampler)

- **Parallelism in ML:** HuggingFace has a nice [writeup](https://huggingface.co/docs/transformers/v4.15.0/parallelism) that summarizes the common parallelism paradigms in (large-scale) ML training. 

## Acknowledgements

- The skeleton code is modified from [run_glue.py](https://github.com/huggingface/transformers/blob/7e7fc53da5f230db379ece739457c81b2f50f13e/examples/run_glue.py) in an older version of the [`transformers/examples`](https://github.com/huggingface/transformers/tree/main/examples) repository. The source code supports DDP, but it will not work out of the box for Task 3.
- This assignment took a lot of inspiration from [this assignment](https://pages.cs.wisc.edu/~shivaram/cs744-fa22/assignment2.html) in CS 744 designed by [Prof. Shivaram Venkataraman](https://shivaram.org/). 
