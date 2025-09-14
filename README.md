# StarXAttention

The following is a modified implementation of Star Attention which utilises anti-diagonal scoring as proposed in XAttention for blockwise weighting.

## Setup Instructions

### Dependencies

Install all the project dependencies with
```
$ pip install -r requirements.txt
```

In a python shell, download the `punkt` tokenizer from the `nltk` library:
```
import nltk
nltk.download('punkt_tab')
```

### RULER Setup

To generate synthetic data for RULER, you need to download:
- Paul Graham Essays for NIAH from [NIAH Github](https://github.com/gkamradt/LLMTest_NeedleInAHaystack/tree/main/needlehaystack/PaulGrahamEssays) and [Paul Graham Blog](https://paulgraham.com/articles.html).
- QA datasets from [SQuAD](https://rajpurkar.github.io/SQuAD-explorer/) and [HotpotQA](https://hotpotqa.github.io/).

To download these data, run:
```
$ bash ruler/download_data.sh
```

### Downloading Models

To download a model from HuggingFace, use the script: [`scripts/download_hf_model.py`](scripts/download_hf_model.py).

*NOTE: For certain models, you might need to input the huggingface hub token from your account settings via the `--token` flag.*

## Launching Inference with StarX Attention

This repository contains code for launching inference with Star Attention on the benchmark RULER . The instructions to run benchmark are shared in the following subsections.

### RULER
To run inference on RULER, use the script: [`run_ruler.py`](run_ruler.py).

Usage:
```
$ python run_ruler.py \
    -n <experiment_name> \
    -p <path_to_model> \
    -pc <prompt_template_type> \
    -a star \
    -bs <context_block_size> \
    -l <list_of_sequence_lengths_to_run_inference> \
    -np <num_parallel_processes_per_node> \
    --output_dir <output_directory>
```

After running the evaluations, you can display all the results together using the script:
```
$ python ruler/gather_results_ruler.py \
    -e <path_to_results_directory>
```

#### Configuring the `-np` flag

The `-np` flag specifies the number of parallel processes (hosts) to use for running inference. For example, if your machine has 8 GPUs:
- `-np 8`: Launch 8 hosts, each host assigned a single GPU.
- `-np 4`: Launch 4 hosts, each host assigned 2 GPUs.

This is useful when you want to run star attention with bigger context block sizes or with bigger models where assigning a single GPU per host leads to out-of-memory errors.

To see an example of how to run this script with different configurations, check [`launch.sh`](launch.sh).

#### Configuring the `-nn` flag for Multi-Node Inference

If you have a multi-node setup (such as a slurm cluster), then you can add the `-nn <num_nodes>` for running multi-node inference. The script will launch a total of `nn * np` processes (hosts) for inference.

## References

- Llama Implementation: [huggingface/transformers](https://github.com/huggingface/transformers)
- Star Attention Implementation: [NVIDIA/star-attention](https://github.com/NVIDIA/Star-Attention/tree/main)
