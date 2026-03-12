# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" Finetuning the library models for sequence classification on GLUE (Bert, XLM, XLNet, RoBERTa)."""

from __future__ import absolute_import, division, print_function

import argparse
import contextlib
import glob
import logging
import os
import random
import time  # Use time.perf_counter() for timing; see https://realpython.com/python-timer/

import numpy as np
import torch
from torch.profiler import profile, ProfilerActivity, schedule

try:
    import matplotlib
    matplotlib.use("Agg")  # Non-interactive backend for headless / server
    import matplotlib.pyplot as plt
    _HAS_MATPLOTLIB = True
except ImportError:
    _HAS_MATPLOTLIB = False

from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler, TensorDataset)
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm, trange

# import a previous version of the HuggingFace Transformers package
from pytorch_transformers import (WEIGHTS_NAME, BertConfig,
                                  BertForSequenceClassification, BertTokenizer,
                                  RobertaConfig,
                                  RobertaForSequenceClassification,
                                  RobertaTokenizer,
                                  XLMConfig, XLMForSequenceClassification,
                                  XLMTokenizer, XLNetConfig,
                                  XLNetForSequenceClassification,
                                  XLNetTokenizer)

from pytorch_transformers import AdamW, WarmupLinearSchedule

from utils_glue import (compute_metrics, convert_examples_to_features,
                        output_modes, processors)

logger = logging.getLogger(__name__)

ALL_MODELS = sum((tuple(conf.pretrained_config_archive_map.keys()) for conf in (BertConfig, XLNetConfig, XLMConfig, RobertaConfig)), ())

MODEL_CLASSES = {
    'bert': (BertConfig, BertForSequenceClassification, BertTokenizer),
    'xlnet': (XLNetConfig, XLNetForSequenceClassification, XLNetTokenizer),
    'xlm': (XLMConfig, XLMForSequenceClassification, XLMTokenizer),
    'roberta': (RobertaConfig, RobertaForSequenceClassification, RobertaTokenizer),
}


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.cuda.manual_seed_all(args.seed)


def sync_gradients(model, world_size, rank):
    """
    Gather gradients from all workers to rank 0, compute element-wise mean,
    then scatter the mean gradient back to all workers (Task 2a).
    Uses torch.distributed.gather and torch.distributed.scatter (gloo/nccl).
    """
    for p in model.parameters():
        if p.grad is None:
            continue
        # Gather: all ranks send p.grad to rank 0
        if rank == 0:
            gather_list = [torch.empty_like(p.grad) for _ in range(world_size)]
        else:
            gather_list = None
        torch.distributed.gather(p.grad, gather_list, dst=0)
        # Rank 0: average
        if rank == 0:
            mean_grad = torch.stack(gather_list).mean(dim=0)
            scatter_list = [mean_grad.clone() for _ in range(world_size)]
        else:
            scatter_list = None
        # Scatter: rank 0 sends mean_grad to every rank (into p.grad)
        torch.distributed.scatter(p.grad, scatter_list, src=0)


def train(args, train_dataset, model, tokenizer):
    """ Train the model """

    args.train_batch_size = args.per_device_train_batch_size
    if args.master_ip is not None:
        world_size = torch.distributed.get_world_size()
        rank = torch.distributed.get_rank()
        train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank)
    else:
        world_size = 1
        rank = 0
        train_sampler = RandomSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.train_batch_size)

    if args.max_steps > 0:
        t_total = args.max_steps
        args.num_train_epochs = args.max_steps // (len(train_dataloader) // args.gradient_accumulation_steps) + 1
    else:
        t_total = len(train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs

    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': args.weight_decay},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    scheduler = WarmupLinearSchedule(optimizer, warmup_steps=args.warmup_steps, t_total=t_total)
    if args.fp16:
        try:
            from apex import amp
        except ImportError:
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
        model, optimizer = amp.initialize(model, optimizer, opt_level=args.fp16_opt_level)

    # Train!
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_dataset))
    logger.info("  Num Epochs = %d", args.num_train_epochs)
    logger.info("  Instantaneous batch size per device = %d", args.per_device_train_batch_size)
    logger.info("  Total train batch size (w. parallel, distributed & accumulation) = %d",
                   args.train_batch_size * args.gradient_accumulation_steps * (torch.distributed.get_world_size() if args.local_rank != -1 else 1))
    logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
    logger.info("  Total optimization steps = %d", t_total)

    global_step = 0
    tr_loss, logging_loss = 0.0, 0.0
    model.zero_grad()

    # Timing via time.perf_counter() (high resolution); first iteration discarded, then average.
    # Ref: https://realpython.com/python-timer/
    iter_times = []
    loss_curve = []  # (step, loss) per node for plotting
    step_loss_accum = 0.0
    t_iter_start = None

    # Task 4: Profiling — skip 1st step, profile 3 steps, save Chrome trace.
    # FIX: default=False so profiling only runs when --profile is explicitly passed.
    trace_addr = None
    if args.profile:
        os.makedirs(args.output_dir, exist_ok=True)
        trace_addr = os.path.join(args.output_dir, "chrome_trace_part2a_rank{}.json".format(rank))
        sched = schedule(wait=1, warmup=0, active=3, repeat=1)
        activities = [ProfilerActivity.CPU, ProfilerActivity.CUDA] if (torch.cuda.is_available() and not args.no_cuda) else [ProfilerActivity.CPU]
        prof_context = profile(
            activities=activities,
            schedule=sched,
            on_trace_ready=lambda p: p.export_chrome_trace(trace_addr),
        )
        logger.info("[Task 4] Profiling enabled (skip 1 step, profile 3 steps). Trace will be saved to %s", trace_addr)
    else:
        prof_context = contextlib.nullcontext(enter_result=None)

    train_iterator = trange(int(args.num_train_epochs), desc="Epoch", disable=args.local_rank not in [-1, 0])
    set_seed(args)  # Added here for reproductibility (even between python 2 and 3)
    with prof_context as prof:
        for epoch in train_iterator:
            if args.master_ip is not None:
                train_sampler.set_epoch(epoch)
            epoch_iterator = tqdm(train_dataloader, desc="Iteration", disable=args.local_rank not in [-1, 0])
            for step, batch in enumerate(epoch_iterator):
                # Start timer at the beginning of each accumulation cycle (perf_counter: best for short durations)
                if step % args.gradient_accumulation_steps == 0:
                    t_iter_start = time.perf_counter()
                    step_loss_accum = 0.0
                model.train()
                batch = tuple(t.to(args.device) for t in batch)
                inputs = {'input_ids':      batch[0],
                          'attention_mask': batch[1],
                          'token_type_ids': batch[2] if args.model_type in ['bert', 'xlnet'] else None,  # XLM don't use segment_ids
                          'labels':         batch[3]}
                outputs = model(**inputs)
                loss = outputs[0]  # model outputs are always tuple in pytorch-transformers (see doc)

                if args.gradient_accumulation_steps > 1:
                    loss = loss / args.gradient_accumulation_steps

                if args.fp16:
                    with amp.scale_loss(loss, optimizer) as scaled_loss:
                        scaled_loss.backward()
                else:
                    loss.backward()

                tr_loss += loss.item()
                step_loss_accum += loss.item()
                if ((step + 1) % args.gradient_accumulation_steps == 0) or ((step + 1) == len(train_dataloader)):
                    # Task 2a: sync gradients (gather to rank 0, average, scatter) before clip/step
                    if args.master_ip is not None:
                        sync_gradients(model, world_size, rank)
                    if args.fp16:
                        torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), args.max_grad_norm)
                    else:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                    # Record iteration time (first iteration discarded when reporting average)
                    iter_times.append(time.perf_counter() - t_iter_start)
                    # Log loss curve: step_loss_accum is already average (we add loss/accum_steps each batch)
                    loss_curve.append((global_step + 1, step_loss_accum))
                    optimizer.step()
                    scheduler.step()  # Update learning rate schedule
                    model.zero_grad()
                    global_step += 1
                    # Task 4: profiler step
                    if prof is not None:
                        prof.step()
                        if torch.cuda.is_available() and not args.no_cuda:
                            torch.cuda.synchronize()

                if args.max_steps > 0 and global_step > args.max_steps:
                    epoch_iterator.close()
                    break
            if args.max_steps > 0 and global_step > args.max_steps:
                train_iterator.close()
                break

            # Barrier so all ranks finish the epoch before any rank enters evaluate().
            # Otherwise rank 0 can enter eval while others are still in sync_gradients → "Connection closed by peer".
            if args.do_eval:
                if args.master_ip is not None:
                    torch.distributed.barrier()
                if args.local_rank in [-1, 0]:
                    evaluate(args, model, tokenizer, prefix="epoch-{}".format(epoch))
                if args.master_ip is not None:
                    torch.distributed.barrier()

    if trace_addr is not None:
        logger.info("[Task 4] Trace file saved at %s (open in Chrome: chrome://tracing)", trace_addr)

    # Report average time per iteration (discard first iteration) and save to file
    os.makedirs(args.output_dir, exist_ok=True)
    avg_time_path = os.path.join(args.output_dir, "avg_time_part2a_rank{}.txt".format(rank))
    if len(iter_times) > 1:
        avg_time_per_iter = sum(iter_times[1:]) / (len(iter_times) - 1)
        logger.info("[Task 2(a)] Rank %s: average time per iteration (excluding first): %.4f sec (%d iterations)",
                    rank, avg_time_per_iter, len(iter_times) - 1)
        with open(avg_time_path, "w") as f:
            f.write("rank\tavg_time_per_iter_sec\tnum_iterations\n")
            f.write("%d\t%.6f\t%d\n" % (rank, avg_time_per_iter, len(iter_times) - 1))
        logger.info("[Task 2(a)] Rank %s: avg time written to %s", rank, avg_time_path)

    # Log loss curve for this node (each rank writes its own file and plot)
    loss_curve_path = os.path.join(args.output_dir, "loss_curve_part2a_rank{}.txt".format(rank))
    with open(loss_curve_path, "w") as f:
        f.write("step\tloss\n")
        for s, l in loss_curve:
            f.write("%d\t%.6f\n" % (s, l))
    logger.info("[Task 2(a)] Rank %s: loss curve written to %s", rank, loss_curve_path)

    # Plot loss curve and save as PNG
    if loss_curve and _HAS_MATPLOTLIB:
        steps = [s for s, _ in loss_curve]
        losses = [l for _, l in loss_curve]
        plt.figure(figsize=(8, 5))
        plt.plot(steps, losses, "b-", linewidth=1)
        plt.xlabel("Step")
        plt.ylabel("Loss")
        plt.title("Training loss (rank {})".format(rank))
        plt.grid(True, alpha=0.3)
        plot_path = os.path.join(args.output_dir, "loss_curve_part2a_rank{}.png".format(rank))
        plt.savefig(plot_path, dpi=150, bbox_inches="tight")
        plt.close()
        logger.info("[Task 2(a)] Rank %s: loss curve plot saved to %s", rank, plot_path)
    elif loss_curve and not _HAS_MATPLOTLIB:
        logger.warning("matplotlib not available; skipping loss curve plot. Install with: pip install matplotlib")

    return global_step, tr_loss / global_step


def evaluate(args, model, tokenizer, prefix=""):
    # Loop to handle MNLI double evaluation (matched, mis-matched)
    eval_task_names = ("mnli", "mnli-mm") if args.task_name == "mnli" else (args.task_name,)
    eval_outputs_dirs = (args.output_dir, args.output_dir + '-MM') if args.task_name == "mnli" else (args.output_dir,)

    results = {}
    for eval_task, eval_output_dir in zip(eval_task_names, eval_outputs_dirs):
        eval_dataset = load_and_cache_examples(args, eval_task, tokenizer, evaluate=True)

        if not os.path.exists(eval_output_dir) and args.local_rank in [-1, 0]:
            os.makedirs(eval_output_dir)

        args.eval_batch_size = args.per_device_eval_batch_size
        eval_sampler = SequentialSampler(eval_dataset)
        eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size)

        # Eval!
        logger.info("***** Running evaluation {} *****".format(prefix))
        logger.info("  Num examples = %d", len(eval_dataset))
        logger.info("  Batch size = %d", args.eval_batch_size)
        eval_loss = 0.0
        nb_eval_steps = 0
        preds = None
        out_label_ids = None
        for batch in tqdm(eval_dataloader, desc="Evaluating"):
            model.eval()
            batch = tuple(t.to(args.device) for t in batch)

            with torch.no_grad():
                inputs = {'input_ids':      batch[0],
                          'attention_mask': batch[1],
                          'token_type_ids': batch[2] if args.model_type in ['bert', 'xlnet'] else None,
                          'labels':         batch[3]}
                outputs = model(**inputs)
                tmp_eval_loss, logits = outputs[:2]

                eval_loss += tmp_eval_loss.mean().item()
            nb_eval_steps += 1
            if preds is None:
                preds = logits.detach().cpu().numpy()
                out_label_ids = inputs['labels'].detach().cpu().numpy()
            else:
                preds = np.append(preds, logits.detach().cpu().numpy(), axis=0)
                out_label_ids = np.append(out_label_ids, inputs['labels'].detach().cpu().numpy(), axis=0)

        eval_loss = eval_loss / nb_eval_steps
        if args.output_mode == "classification":
            preds = np.argmax(preds, axis=1)
        elif args.output_mode == "regression":
            preds = np.squeeze(preds)
        result = compute_metrics(eval_task, preds, out_label_ids)
        results.update(result)

        output_eval_file = os.path.join(eval_output_dir, "eval_results.txt")
        with open(output_eval_file, "w") as writer:
            logger.info("***** Eval results {} *****".format(prefix))
            for key in sorted(result.keys()):
                logger.info("  %s = %s", key, str(result[key]))
                writer.write("%s = %s\n" % (key, str(result[key])))

    return results


def load_and_cache_examples(args, task, tokenizer, evaluate=False):
    # FIX: barrier applies to ALL distributed calls (train and eval), not just train.
    # Without this, non-rank-0 nodes can race to read the eval cache before rank 0 writes it.
    if args.master_ip is not None and args.local_rank not in [-1, 0]:
        torch.distributed.barrier()

    processor = processors[task]()
    output_mode = output_modes[task]
    # Load data features from cache or dataset file
    cached_features_file = os.path.join(args.data_dir, 'cached_{}_{}_{}_{}'.format(
        'dev' if evaluate else 'train',
        list(filter(None, args.model_name_or_path.split('/'))).pop(),
        str(args.max_seq_length),
        str(task)))
    if os.path.exists(cached_features_file):
        logger.info("Loading features from cached file %s", cached_features_file)
        features = torch.load(cached_features_file)
    else:
        logger.info("Creating features from dataset file at %s", args.data_dir)
        label_list = processor.get_labels()
        if task in ['mnli', 'mnli-mm'] and args.model_type in ['roberta']:
            # HACK(label indices are swapped in RoBERTa pretrained model)
            label_list[1], label_list[2] = label_list[2], label_list[1]
        examples = processor.get_dev_examples(args.data_dir) if evaluate else processor.get_train_examples(args.data_dir)
        features = convert_examples_to_features(examples, label_list, args.max_seq_length, tokenizer, output_mode,
            cls_token_at_end=bool(args.model_type in ['xlnet']),
            cls_token=tokenizer.cls_token,
            cls_token_segment_id=2 if args.model_type in ['xlnet'] else 0,
            sep_token=tokenizer.sep_token,
            sep_token_extra=bool(args.model_type in ['roberta']),
            pad_on_left=bool(args.model_type in ['xlnet']),
            pad_token=tokenizer.convert_tokens_to_ids([tokenizer.pad_token])[0],
            pad_token_segment_id=4 if args.model_type in ['xlnet'] else 0,
        )
        if args.local_rank in [-1, 0]:
            logger.info("Saving features into cached file %s", cached_features_file)
            torch.save(features, cached_features_file)

    if args.master_ip is not None and args.local_rank == 0:
        torch.distributed.barrier()

    # Convert to Tensors and build dataset
    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    all_input_mask = torch.tensor([f.input_mask for f in features], dtype=torch.long)
    all_segment_ids = torch.tensor([f.segment_ids for f in features], dtype=torch.long)
    if output_mode == "classification":
        all_label_ids = torch.tensor([f.label_id for f in features], dtype=torch.long)
    elif output_mode == "regression":
        all_label_ids = torch.tensor([f.label_id for f in features], dtype=torch.float)

    dataset = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids)
    return dataset


def main():
    parser = argparse.ArgumentParser()

    ## Required parameters
    parser.add_argument("--data_dir", default=None, type=str, required=True,
                        help="The input data dir. Should contain the .tsv files (or other data files) for the task.")
    parser.add_argument("--model_type", default=None, type=str, required=True,
                        help="Model type selected in the list: " + ", ".join(MODEL_CLASSES.keys()))
    parser.add_argument("--model_name_or_path", default=None, type=str, required=True,
                        help="Path to pre-trained model or shortcut name selected in the list: " + ", ".join(ALL_MODELS))
    parser.add_argument("--task_name", default=None, type=str, required=True,
                        help="The name of the task to train selected in the list: " + ", ".join(processors.keys()))
    parser.add_argument("--output_dir", default=None, type=str, required=True,
                        help="The output directory where the model predictions and checkpoints will be written.")

    ## Other parameters
    parser.add_argument("--config_name", default="", type=str,
                        help="Pretrained config name or path if not the same as model_name")
    parser.add_argument("--tokenizer_name", default="", type=str,
                        help="Pretrained tokenizer name or path if not the same as model_name")
    parser.add_argument("--cache_dir", default="", type=str,
                        help="Where do you want to store the pre-trained models downloaded from s3")
    parser.add_argument("--max_seq_length", default=128, type=int,
                        help="The maximum total input sequence length after tokenization. Sequences longer "
                             "than this will be truncated, sequences shorter will be padded.")
    parser.add_argument("--do_train", action='store_true',
                        help="Whether to run training.")
    parser.add_argument("--do_eval", action='store_true',
                        help="Whether to run eval on the dev set.")
    parser.add_argument("--do_lower_case", action='store_true',
                        help="Set this flag if you are using an uncased model.")

    parser.add_argument("--per_device_train_batch_size", default=8, type=int,
                        help="Batch size per GPU/CPU for training.")
    parser.add_argument("--per_device_eval_batch_size", default=8, type=int,
                        help="Batch size per GPU/CPU for evaluation.")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument("--learning_rate", default=5e-5, type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--weight_decay", default=0.0, type=float,
                        help="Weight deay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float,
                        help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float,
                        help="Max gradient norm.")
    parser.add_argument("--num_train_epochs", default=3.0, type=float,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--max_steps", default=-1, type=int,
                        help="If > 0: set total number of training steps to perform. Override num_train_epochs.")
    parser.add_argument("--warmup_steps", default=0, type=int,
                        help="Linear warmup over warmup_steps.")

    parser.add_argument("--no_cuda", action='store_true',
                        help="Avoid using CUDA when available")
    parser.add_argument('--overwrite_output_dir', action='store_true',
                        help="Overwrite the content of the output directory")
    parser.add_argument('--overwrite_cache', action='store_true',
                        help="Overwrite the cached training and evaluation sets")
    parser.add_argument('--seed', type=int, default=42,
                        help="random seed for initialization")

    parser.add_argument('--fp16', action='store_true',
                        help="Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit")
    parser.add_argument('--fp16_opt_level', type=str, default='O1',
                        help="For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
                             "See details at https://nvidia.github.io/apex/amp.html")
    parser.add_argument("--local_rank", type=int, default=-1,
                        help="For distributed training: local_rank (global rank, e.g. 0,1,2,3 for 4 nodes). If single-node training, local_rank defaults to -1.")
    # Distributed training (Task 2a)
    parser.add_argument("--master_ip", default=None, type=str,
                        help="Master node IP for distributed training. If set, init_process_group is called.")
    parser.add_argument("--master_port", default=29500, type=int,
                        help="Master port for distributed training.")
    parser.add_argument("--world_size", default=1, type=int,
                        help="Total number of workers (nodes) in distributed training.")
    # Task 4: profiling — FIX: default=False so it only runs when --profile is explicitly passed.
    parser.add_argument("--profile", action="store_true", default=False,
                        help="Profile 3 training steps (skip first), save Chrome trace to output_dir.")
    args = parser.parse_args()

    # Initialize distributed process group (gloo for CPU, nccl for GPU)
    if args.master_ip is not None:
        backend = "nccl" if torch.cuda.is_available() and not args.no_cuda else "gloo"
        torch.distributed.init_process_group(
            backend=backend,
            init_method="tcp://{}:{}".format(args.master_ip, args.master_port),
            world_size=args.world_size,
            rank=args.local_rank,
        )

    if os.path.exists(args.output_dir) and os.listdir(args.output_dir) and args.do_train and not args.overwrite_output_dir:
        raise ValueError("Output directory ({}) already exists and is not empty. Use --overwrite_output_dir to overcome.".format(args.output_dir))

    # set up (distributed) training
    if torch.cuda.is_available() and not args.no_cuda:
        if args.master_ip is not None:
            torch.cuda.set_device(0)   # one GPU per node
            args.device = torch.device("cuda:0")
        else:
            args.device = torch.device("cuda")
    else:
        args.device = torch.device("cpu")

    args.n_gpu = torch.cuda.device_count() if (torch.cuda.is_available() and not args.no_cuda) else 0

    # Setup logging
    logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                        datefmt = '%m/%d/%Y %H:%M:%S',
                        level = logging.INFO if args.local_rank in [-1, 0] else logging.WARN)
    logger.warning("Process rank: %s, device: %s, distributed training: %s, 16-bits training: %s",
                    args.local_rank, args.device, bool(args.local_rank != -1), args.fp16)

    # Set seed
    set_seed(args)

    # Prepare GLUE task
    args.task_name = args.task_name.lower()
    if args.task_name not in processors:
        raise ValueError("Task not found: %s" % (args.task_name))
    processor = processors[args.task_name]()
    args.output_mode = output_modes[args.task_name]
    label_list = processor.get_labels()
    num_labels = len(label_list)

    # Load pretrained model and tokenizer
    if args.master_ip is not None and args.local_rank not in [-1, 0]:
        torch.distributed.barrier()  # Make sure only the first process downloads model & vocab

    args.model_type = args.model_type.lower()
    config_class, model_class, tokenizer_class = MODEL_CLASSES[args.model_type]
    config = config_class.from_pretrained(args.config_name if args.config_name else args.model_name_or_path, num_labels=num_labels, finetuning_task=args.task_name)
    tokenizer = tokenizer_class.from_pretrained(args.tokenizer_name if args.tokenizer_name else args.model_name_or_path, do_lower_case=args.do_lower_case)
    model = model_class.from_pretrained(args.model_name_or_path, config=config)

    if args.master_ip is not None and args.local_rank == 0:
        torch.distributed.barrier()  # Make sure only the first process downloads model & vocab

    model.to(args.device)

    logger.info("Training/evaluation parameters %s", args)

    # Training
    if args.do_train:
        train_dataset = load_and_cache_examples(args, args.task_name, tokenizer, evaluate=False)
        global_step, tr_loss = train(args, train_dataset, model, tokenizer)
        logger.info(" global_step = %s, average loss = %s", global_step, tr_loss)

    # Evaluation (only on rank 0 in distributed mode to avoid duplicate writes)
    if args.do_eval:
        if args.master_ip is not None:
            torch.distributed.barrier()
        if args.local_rank in [-1, 0]:
            evaluate(args, model, tokenizer, prefix="")
        if args.master_ip is not None:
            torch.distributed.barrier()

if __name__ == "__main__":
    main()
