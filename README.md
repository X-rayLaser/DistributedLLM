# Introduction

The repository provides a set of tools to distribute a given LLM among available machines/devices. This is useful when the model is too large to run on a single machine (for instance, it does not fit in RAM). Users simply need to provide their own LLM and configure the system using a small configuration file; the toolkit handles the rest.

This project builds on top of llama.cpp project. Therefore, it naturally supports scripts for converting the model to GGML format and quantizing it.

Here is the approach in the nutshell:
- split the model into pieces
- send each piece to a dedicated machine and load it in memory (RAM/VRAM)
- connect all machines in a network
- propagate tensors forward through the layers of the first piece, propagate resulting tensor through the layers of the second piece, and so on

Supported devices:
- CPU
- GPU (will come later)

Supported Models:
- LLaMA (version 1)
- OpenLLaMA (version 1)

Note that currently the toolkit does not suport LLaMA version 2. The same goes for OpenLLaMA. This support will be added in the future. 

# Status: early stage

This project is still under development and may have bugs or limitations. Use it on your own risk.


# 1. Main components

Compute node is an element of the network that stores a slice of a model (a subset of transformer layers). It takes a tensor, propagates it forward through it's subset of layers, and returns output tensor. Multiple compute nodes can be deployed on a single machine.

Server machine is a physical machine/device that runs a compute node.

Client node is an element of the network that establishes connection with compute nodes, queries their status, provisions them, and, finally, uses them for running inference on LLM.

Client machine is a machine that runs a client node.


# 2. The workflow
1. Decide how many machines should be used to distribte LLM between them

2. On each server machine, clone this repository and deploy a compute node on there

3. On the client machine:
- obtain a model and create a small and simple configuration file
- run provision command
- generate text using the LLM distributed on your network


# 3. Quick start

## Cloning the repository:

Cloning should be done with --recurse-submodule flag, since this repository contains a git submodule:
```
git clone --recurse-submodules https://github.com/X-rayLaser/DistributedLLM.git
```

## Deploying the compute node

To test how the system works, you can use a default docker-compose file docker-compose.yml. It will automatically deploy two compute nodes on the current machine.

To do so, build a docker image and run two containerized compute nodes:
```
sudo docker-compose build
sudo docker-compose up
```

To find out IP addresses of containers, identify names of running containers:
```
sudo docker-compose ps
```

For each name, execute the following command by replacing \<container name\> with an actual name:
```
sudo docker inspect <container_name> | grep "IPAddress"
```

You should now see the IP address in the output of the command.

### Deploying compute node on a particular machine

This subsection describes the steps required on the server machine.

Set the environment variable PORT by creating a .env file at the root directory of the repository. Inside the file, define the environment variable like so (replacing the value with your own):
```
PORT=9997
```

Build a Docker image using the following command:
```
sudo docker-compose -f docker-compose-prod.yml build
```

Finally, start the image in a container with this command:
```
sudo docker-compose -f docker-compose-prod.yml up 
```


## Provisioning

Provisioning will automatically prepare a chosen model, split it into pieces and send those pieces to their corresponding compute nodes.

This subsection describes the steps required on the client machine, which is the one that will be used to interact with the distributed LLM.

Create a directory "models" in the root directory of the repository:
```
mkdir models
```

This directory is a convenient place to store models you wish to use. Place a model of the supported type (e.g. LLama) in this folder.

To configure the nodes in the network, create a configuration file called my_config.json in the configs subdirectory of the root directory of the repository. Use the following template for the configuration file, replacing the values with your own settings:
```
{
    "model_id": "my_open_llama", # identifier given to the model
    "location": "models/open_llama_3b", # location of the Hugging Face model directory
    "nodes_map": { # assign model slices to compute nodes
        "127.0.0.1:9998": [0, 16], # assign a slice containing layers from 0 up to (and including) 16
        "127.0.0.1:9999": [17, 25] # assign a slice containing layers from 17 up to (and including) 25
    },
    "quantization": "q4_0", # sets quantization method implmented, no quantization by default
    "metadata": { # field storing meta information about the model
        "name": "open_llama_3b",
        "family": "llama_v1",
        "size": "3B",
        "quantization": "q4_0",
        "usage_class": "base"
    }
}

```

Build a docker image for a client container:
```
sudo docker-compose -f docker-compose-client.yml build
```

After creating the configuration file, execute the following Python script to provision the nodes inside the container:
```
sudo docker-compose -f docker-compose-client.yml run client python3 -u manager.py provision configs/my_config.json
```

## Running inference on distributed LLM

After successfully deploying the compute nodes and provisioning them, you can utilize the distributed LLM as if working with a regular LLM. At present, only basic text generation functionality is available, making it ideal for base models but unsuitable for chat models.

Assuming that you wish to generate text based on a prompt using a base model, execute the following command:
```
sudo docker-compose -f docker-compose-client.yml run client python3 manager.py generate_text <config_file> --prompt "Alan Turing" --num-tokens 100 --temp 0.0 --rp 1.11111
```

Replace <config_file> with the path to your configuration file. The --prompt flag specifies the input prompt, while the --num-tokens flag determines the number of generated tokens. Additionally, the --temp flag controls temperature scaling, and the --rp flag adjusts repetition penalty. Feel free to experiment with different values to achieve desired results.

# 4. Installation without docker

It is advisable to use a virtual environment when installing Python dependencies. On Ubuntu, you can create and activate a virtual environment using the following commands:
```
virtualenv --python=<path to python executable> venv
. venv/bin/activate
```

### Instruction for Ubuntu users

1. Install all Python dependencies:
```
pip install -r requirements.txt
```

2. Build C++ libraries of the vendor and copy them to the libs/ folder:
```
mkdir libs
cd vendor/llama.cpp
make libllama.so && make libembdinput.so
cd ../../
cp vendor/llama.cpp/libllama.so libs/libllama.so
cp vendor/llama.cpp/libembdinput.so libs/libembdinput.so
```

3. Build a Python extension implementing functionality for working with an LLM slice:
```
PYTHON_HEADERS_HOME=$(echo "from distutils.sysconfig import get_python_inc; print(get_python_inc())" | python)
g++ -fPIC -shared -I vendor/llama.cpp/examples -I vendor/llama.cpp -I $PYTHON_HEADERS_HOME -o libs/llm.so distllm/tensor_processor.cpp libs/libllama.so libs/libembdinput.so
```

4. Build a utility program that slices a given model:
```
g++ -fPIC -I vendor/llama.cpp/examples -I vendor/llama.cpp -o slice_model slice_model.cpp libs/libllama.so libs/libembdinput.so
```

5. Ensure that the libs/ directory is added to the Python path by setting the PYTHONPATH environment variable (this step must be performed prior to every usage):
```
export PYTHONPATH="${PYTHONPATH}:$(pwd)/libs"
```