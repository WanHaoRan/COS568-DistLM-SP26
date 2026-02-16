# Setting up Cloudlab

You can do this assignment using CloudLab. CloudLab is a research facility that provides bare-metal access and control over a substantial set of computing, storage, and networking resources. If you haven’t worked in CloudLab before, you need to register a CloudLab account.

This write-up walks you through the CloudLab registration process and shows you how to start an experiment in CloudLab.

Most importantly, it introduces our policies on using CloudLab for this project.

## Register a CloudLab Account

Use this [link](https://www.cloudlab.us/signup.php?pid=cos568proj2-sp26) to register and join project `cos568proj2-sp26`. Please use your Princeton email address for registration. Note that an SSH public key is required to access the nodes CloudLab assigns to you; if you are unfamiliar with creating and using ssh keypairs, we recommend taking a look at the first few steps in [GitHub’s guide to generating SSH keys](https://docs.github.com/en/authentication/connecting-to-github-with-ssh). (Obviously, the steps about how to upload the keypair into GitHub don’t apply to CloudLab.) 

## Start an Experiment

To start a new experiment, go to your CloudLab dashboard and click on the `Experiments` tab in the upper left corner, then select `Start Experiment`. This will lead to the profile selection panel. Click on `Change Profile`, and select a profile from the list. For example, if you choose the `cos568-proj2-sp26` profile in the `cos568proj2-sp26` project, you will be able to launch 4 CPU nodes. Select the profile and click on Next to move to the next panel. Here you should name your experiment with `NetID-ExperimentName`. The purpose of doing this is to prevent everyone from picking random names and ending up confusing each other since everyone in the `cos568proj2-sp26` project can see a full list of experiments created. You also need to specify from which cluster you want to start your experiment. Each cluster has different hardware. For more information on the hardware CloudLab provides, please refer to [this doc](https://docs.cloudlab.us/hardware.html). **Some students have observed network issues on Utah and Mass nodes so we highly recommend to run experiments on Wisc nodes**. Once you select the cluster you can instantiate the experiment. Once the experiment is ready you will see the ssh login command. Try to log in to the machine and check for the number of CPU cores available and memory available on the node using `htop` (You might need to install htop first by running `sudo apt-get update ; sudo apt-get install htop`).


## Storage

CloudLab has two storage systems, including:
1. A non-shared, ephemeral (which means the data will be removed after the experiment ends) local storage system that is only accessible to the node you logged in (i.e. your home directory `/home`). For example, if you are allocated 4 nodes in the experiments, each node has an independent `/home` directory that is not shared with other nodes.
2. A shared, persistent (which means the data will be kept after the experiment ends) NFS system that is accessible for **all nodes** inside the project (i.e. the directory starts with `/proj/cos568proj2-sp26-PG0`). This means **everyone** can access the file inside this directory. This directory has 100G disk space **in total**. **Please don't use this space for your assignment.**
3. You can check the disk hierarchy using `df -h`

[**!!!IMPORTANT!!!**]
To run experiments efficiently, we suggest to follow the workflow below:
1. Start an experiment (each experiment will last for 16 hours at maximum)
2. (Need to run every time when you start a new experiment) For each node, initialize the system using the script :
   ```bash
    sudo apt-get update
    sudo apt-get install htop dstat python3-pip
    echo 'export PATH=$HOME/.local/bin:$PATH' >> ~/.bashrc
    source ~/.bashrc
    
    pip install torch==2.5.0 torchvision==0.20.0 torchaudio==2.5.0 --index-url https://download.pytorch.org/whl/cpu
    pip install numpy scipy scikit-learn tqdm pytorch_transformers apex
   ```

3. (Need to run every time when you start a new experiment) For each node, clone your git repo to your home directory. **Please don't use the shared space for your assignment.**
4. (Need to run every time when you start a new experiment) Download GLUE/RTE data in your home directory: `python3 download_glue_data.py --data_dir ~/glue_data`.


## Policies on Using CloudLab Resources

The nodes you receive from CloudLab are real hardware machines sitting in different clusters. Therefore, we ask you not to hold the nodes for too long. CloudLab gives users 16 hours to start with, and users can extend it for a longer time. Manage your time efficiently and only hold onto those nodes when you are working on the assignment. You should use a private git repository to manage your code, and you must terminate the nodes when you are not using them. If you do have a need to extend the nodes, do not extend them by more than 1 day. We will terminate any cluster running for more than 48 hours.

As a member of the `cos568proj2-sp26` project, you have permission to access another member’s private user space. Stick to your own space and do not access others’ to peek at/copy/use their code, or intentionally/unintentionally overwrite files in others’ workspaces. For more information related to this, please refer to [https://odoc.princeton.edu/learning-curriculum/academic-integrity](https://odoc.princeton.edu/learning-curriculum/academic-integrity).
