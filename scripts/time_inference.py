import argparse
import json
import os
import time
from tqdm import tqdm
from typing import List, Optional

import torch.distributed as dist


def read_jsonl(filename, num_lines=-1):
    lines = []
    with open(filename) as f:
        for i, line in enumerate(f):
            lines.append(json.loads(line))
            if i == num_lines:
                break
    return lines


def init_distributed():
    """Initialize the distributed environment."""

    if 'RANK' in os.environ:
        dist.init_process_group('nccl')
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        print(f'[run_star_attn_inference.init_distributed] Rank: {rank}, World size: {world_size}')
    else:
        rank = 0
        world_size = 1

    return rank, world_size


def load_model(
    model_path,
    attn_type,
    tokens_to_generate,
    block_size=-1,
    anchor_block_size=-1,
    stop_words=None,
):
    if attn_type == 'dense':
        from model import DenseAttentionModel

        model = DenseAttentionModel(
            path=model_path,
            max_new_tokens=tokens_to_generate,
            stop_words=stop_words,
        )

    elif attn_type == 'ring':
        from model import RingAttentionModel

        model = RingAttentionModel(
            path=model_path,
            max_new_tokens=tokens_to_generate,
            stop_words=stop_words,
        )

    elif attn_type == 'star':
        from model import StarAttentionModel

        assert block_size > 0, 'block_size must be provided for star attention'
        model = StarAttentionModel(
            path=model_path,
            block_size=block_size,
            max_new_tokens=tokens_to_generate,
            stop_words=stop_words,
            anchor_block_size=anchor_block_size,
        )

    else:
        raise ValueError(f'Unsupported attention type: {attn_type}')

    return model


def main(
    model_path: str,
    attn_type: str,
    tokens_to_generate: int,
    input_file: str,
    num_samples: int = -1,
    stop_words: Optional[List[str]] = None,
    block_size: int = -1,
    anchor_block_size: int = -1,
):
    """Run inference using Star-Attention.

    Args:
        model_path: path to the model checkpoint
        attn_type: type of attention. One of ['dense', 'star', 'starkv']
        tokens_to_generate: number of tokens to generate during generation
        input_file: path to the input jsonl file
        output_file: path to the output jsonl file where the generated predictions will be saved
        stop_words: list of stop words for generation. Default: None
        block_size: block size for star attention. Default: -1 (should be provided for star attention)
    """
    rank, _ = init_distributed()

    if rank == 0:
        process_start_time = time.time()

    # Load data
    input_data = read_jsonl(input_file, num_lines=num_samples)
    num_samples = len(input_data)

    # Load model
    model = load_model(
        model_path,
        attn_type,
        tokens_to_generate,
        block_size,
        anchor_block_size,
        stop_words=stop_words,
    )

    # Warmup
    if rank == 0:
        print('Warmup...')
    for input_sample in input_data[:5]:
        _ = model(prompt_context=input_sample['input_context'], prompt_query=input_sample['input_query'])
        dist.barrier()

    # Generate predictions
    if rank == 0:
        print('Generating predictions...')
        inference_start_time = time.time()

    p2times = 0
    for input_sample in tqdm(input_data, total=len(input_data)):
        _ = model(prompt_context=input_sample['input_context'], prompt_query=input_sample['input_query'])
        dist.barrier()

    if rank == 0:
        print()
        end_time = time.time()
        print(f'Total time: {round((end_time - process_start_time) / 60, 1)} minutes')
        print(f'Inference time: {round((end_time - inference_start_time) / 60, 1)} minutes')
        print(f'Time per sample: {(end_time - inference_start_time) / len(input_data)} seconds')

    dist.barrier()
    dist.destroy_process_group()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', required=True, help='path to the model checkpoint')
    parser.add_argument('--attn_type', required=True, help='type of attention')
    parser.add_argument('--block_size', type=int, default=-1, help='block size for star attention')
    parser.add_argument('--anchor_block_size', type=int, default=-1, help='anchor block size for star attention')
    parser.add_argument('--tokens_to_generate', type=int, required=True, help='number of tokens to generate')
    parser.add_argument('--stop_words', default='', help='comma separated stop words for generation')
    parser.add_argument('--input_path', required=True, help='path to the input jsonl file')
    parser.add_argument('--num_samples', type=int, default=-1, help='number of samples to use from the input file')
    args = parser.parse_args()

    if not os.path.exists(args.model_path):
        raise FileNotFoundError(f'Invalid model path: {args.model_path}')

    if not os.path.exists(args.input_path):
        raise FileNotFoundError(f'Invalid input path: {args.input_path}')

    main(
        args.model_path,
        args.attn_type,
        args.tokens_to_generate,
        args.input_path,
        num_samples=args.num_samples,
        stop_words=list(filter(None, args.stop_words.split(','))),
        block_size=args.block_size,
        anchor_block_size=args.anchor_block_size,
    )