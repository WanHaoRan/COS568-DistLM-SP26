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
"""
Part 2(a): Distributed Data Parallel Training with gather/scatter gradient synchronization.

Run with:
  python run_glue.py [other args] \
      --master_ip 10.10.1.1 --master_port 12345 \
      --world_size 4 --local_rank <0|1|2|3>

With world_size=4 and per_device_train_batch_size=16, total batch size = 64,
matching the single-node run from Task 1.
"""

from __future__ import absolute_import, division, print_function

import argparse
import contextlib
import logging
import os
import random
import time

import numpy as np
import torch
import torch.distributed as dist
from torch.profiler import profile, ProfilerActivity, schedule
from torch.utils.data import (DataLoader, SequentialSampler, TensorDataset)
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm, trange

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    _HAS_MATPLOTLIB = True
except ImportError:
    _HAS_MATPLOTLIB = False

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

ALL_MODELS = sum((tuple(conf.pretrained_config_archive_map.keys()) for conf in
                  (BertConfig, XLNetConfig, XLMConfig, RobertaConfig)), ())

MODEL_CLASSES = {
    'bert':    (BertConfig,   BertForSequenceClassification,   BertTokenizer),
    'xlnet':   (XLNetConfig,  XLNetForSequenceClassification,  XLNetTokenizer),
    'xlm':     (XLMConfig,    XLMForSequenceClassification,    XLMTokenizer),
    'roberta': (RobertaConfig, RobertaForSequenceClassification, RobertaTokenizer),
}


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.cuda.manual_seed_all(args.seed)


def sync_gradients_gather_scatter(model, world_size, rank):
    """
    Gradient synchronization using gather + scatter.
    Worker 0 gathers all gradients, averages them, scatters back to all workers.
    """
    for param in model.parameters():
        if param.grad is None:
            continue
        # Gather all gradients to rank 0
        if rank == 0:
            gather_list = [torch.empty_like(param.grad) for _ in range(world_size)]
        else:
            gather_list = None
        dist.gather(param.grad, gather_list=gather_list, dst=0)

        # Rank 0 averages and prepares scatter
        if rank == 0:
            mean_grad = torch.stack(gather_list).mean(dim=0)
            scatter_list = [mean_grad.clone() for _ in range(world_size)]
        else:
            scatter_list = None

        # Scatter averaged gradient back to all workers
        dist.scatter(param.grad, scatter_list=scatter_list, src=0)


def train(args, train_dataset, model, tokenizer):
    """Train the model with gather/scatter gradient synchronization."""

    args.train_batch_size = args.per_device_train_batch_size
    world_size = dist.get_world_size() if args.local_rank != -1 else 1
    rank = dist.get_rank() if args.local_rank != -1 else 0

    train_sampler = DistributedSampler(
        train_dataset, num_replicas=world_size, rank=rank, shuffle=True)
    train_dataloader = DataLoader(
        train_dataset, sampler=train_sampler, batch_size=args.train_batch_size)

    if args.max_steps > 0:
        t_total = args.max_steps
        args.num_train_epochs = (args.max_steps //
                                 (len(train_dataloader) // args.gradient_accumulation_steps) + 1)
    else:
        t_total = (len(train_dataloader) //
                   args.gradient_accumulation_steps * args.num_train_epochs)

    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters()
                    if not any(nd in n for nd in no_decay)], 'weight_decay': args.weight_decay},
        {'params': [p for n, p in model.named_parameters()
                    if any(nd in n for nd in no_decay)], 'weight_decay': 0.0},
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    scheduler = WarmupLinearSchedule(optimizer, warmup_steps=args.warmup_steps, t_total=t_total)

    logger.info("***** Running training (Task 2a: gather/scatter) *****")
    logger.info("  Num examples = %d", len(train_dataset))
    logger.info("  Num Epochs = %d", args.num_train_epochs)
    logger.info("  Per-device batch = %d", args.per_device_train_batch_size)
    logger.info("  World size = %d", world_size)
    logger.info("  Total batch size = %d", args.train_batch_size * world_size)
    logger.info("  Total optim steps = %d", t_total)

    global_step = 0
    tr_loss = 0.0
    model.zero_grad()

    # Timing (perf_counter for high resolution); discard first iteration
    iter_times = []
    # Loss curve: list of (step, loss)
    loss_curve = []

    # Task 4: profiling — skip 1 step, profile 3 steps, save chrome trace
    os.makedirs(args.output_dir, exist_ok=True)
    trace_path = os.path.join(args.output_dir, "chrome_trace_task2a_rank{}.json".format(rank))
    prof_schedule = schedule(wait=1, warmup=0, active=3, repeat=1)
    activities = ([ProfilerActivity.CPU, ProfilerActivity.CUDA]
                  if (torch.cuda.is_available() and not args.no_cuda)
                  else [ProfilerActivity.CPU])
    prof_context = profile(
        activities=activities,
        schedule=prof_schedule,
        on_trace_ready=lambda p: p.export_chrome_trace(trace_path),
    )
    logger.info("[Task 4] Chrome trace will be saved to %s", trace_path)

    train_iterator = trange(int(args.num_train_epochs), desc="Epoch",
                            disable=args.local_rank not in [-1, 0])
    set_seed(args)

    with prof_context as prof:
        for epoch_idx in train_iterator:
            train_sampler.set_epoch(epoch_idx)
            epoch_iterator = tqdm(train_dataloader, desc="Iteration",
                                  disable=args.local_rank not in [-1, 0])

            for step, batch in enumerate(epoch_iterator):
                t_start = time.perf_counter()

                model.train()
                batch = tuple(t.to(args.device) for t in batch)
                inputs = {
                    'input_ids':      batch[0],
                    'attention_mask': batch[1],
                    'token_type_ids': batch[2] if args.model_type in ['bert', 'xlnet'] else None,
                    'labels':         batch[3],
                }
                outputs = model(**inputs)
                loss = outputs[0]

                if args.gradient_accumulation_steps > 1:
                    loss = loss / args.gradient_accumulation_steps

                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                tr_loss += loss.item()

                if step < 5:
                    logger.info("  [rank %d] step %d loss = %.4f", rank, step, loss.item())

                if (step + 1) % args.gradient_accumulation_steps == 0:
                    # ---- Gradient synchronization via gather/scatter ----
                    if args.local_rank != -1:
                        sync_gradients_gather_scatter(model, world_size, rank)
                    # -----------------------------------------------------
                    optimizer.step()
                    scheduler.step()
                    model.zero_grad()
                    global_step += 1
                    loss_curve.append((global_step, loss.item()))
                    if prof is not None:
                        prof.step()

                elapsed = time.perf_counter() - t_start
                if step > 0:
                    iter_times.append(elapsed)

                if args.max_steps > 0 and global_step > args.max_steps:
                    epoch_iterator.close()
                    break

            if args.max_steps > 0 and global_step > args.max_steps:
                train_iterator.close()
                break

            # Barrier before/after eval to avoid race conditions
            if args.local_rank != -1:
                dist.barrier()
            if args.local_rank in [-1, 0]:
                evaluate(args, model, tokenizer, prefix="epoch_{}".format(epoch_idx + 1))
            if args.local_rank != -1:
                dist.barrier()

    # Save timing
    if iter_times:
        avg_iter_time = sum(iter_times) / len(iter_times)
        logger.info("  [rank %d] Avg iteration time (excl. first) = %.4f s", rank, avg_iter_time)
        timing_path = os.path.join(args.output_dir, "avg_time_task2a_rank{}.txt".format(rank))
        with open(timing_path, "w") as f:
            f.write("rank\tavg_time_per_iter_sec\tnum_iters\n")
            f.write("{}\t{:.6f}\t{}\n".format(rank, avg_iter_time, len(iter_times)))

    # Save loss curve
    loss_path = os.path.join(args.output_dir, "loss_curve_task2a_rank{}.txt".format(rank))
    with open(loss_path, "w") as f:
        f.write("step\tloss\n")
        for s, l in loss_curve:
            f.write("{}\t{:.6f}\n".format(s, l))
    logger.info("  [rank %d] Loss curve saved to %s", rank, loss_path)

    # Plot loss curve
    if loss_curve and _HAS_MATPLOTLIB:
        steps  = [s for s, _ in loss_curve]
        losses = [l for _, l in loss_curve]
        plt.figure(figsize=(8, 5))
        plt.plot(steps, losses, "b-", linewidth=1)
        plt.xlabel("Step")
        plt.ylabel("Loss")
        plt.title("Task 2a Training Loss (rank {})".format(rank))
        plt.grid(True, alpha=0.3)
        plot_path = os.path.join(args.output_dir, "loss_curve_task2a_rank{}.png".format(rank))
        plt.savefig(plot_path, dpi=150, bbox_inches="tight")
        plt.close()
        logger.info("  [rank %d] Loss curve plot saved to %s", rank, plot_path)

    return global_step, tr_loss / global_step


def evaluate(args, model, tokenizer, prefix=""):
    eval_task_names = (("mnli", "mnli-mm") if args.task_name == "mnli" else (args.task_name,))
    eval_outputs_dirs = ((args.output_dir, args.output_dir + '-MM')
                         if args.task_name == "mnli" else (args.output_dir,))

    results = {}
    for eval_task, eval_output_dir in zip(eval_task_names, eval_outputs_dirs):
        eval_dataset = load_and_cache_examples(args, eval_task, tokenizer, evaluate=True)
        if not os.path.exists(eval_output_dir) and args.local_rank in [-1, 0]:
            os.makedirs(eval_output_dir)

        args.eval_batch_size = args.per_device_eval_batch_size
        eval_sampler = SequentialSampler(eval_dataset)
        eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler,
                                     batch_size=args.eval_batch_size)

        logger.info("***** Running evaluation %s *****", prefix)
        eval_loss = 0.0
        nb_eval_steps = 0
        preds = None
        out_label_ids = None

        for batch in tqdm(eval_dataloader, desc="Evaluating"):
            model.eval()
            batch = tuple(t.to(args.device) for t in batch)
            with torch.no_grad():
                inputs = {
                    'input_ids':      batch[0],
                    'attention_mask': batch[1],
                    'token_type_ids': batch[2] if args.model_type in ['bert', 'xlnet'] else None,
                    'labels':         batch[3],
                }
                outputs = model(**inputs)
                tmp_eval_loss, logits = outputs[:2]
                eval_loss += tmp_eval_loss.mean().item()
            nb_eval_steps += 1
            if preds is None:
                preds = logits.detach().cpu().numpy()
                out_label_ids = inputs['labels'].detach().cpu().numpy()
            else:
                preds = np.append(preds, logits.detach().cpu().numpy(), axis=0)
                out_label_ids = np.append(out_label_ids,
                                          inputs['labels'].detach().cpu().numpy(), axis=0)

        eval_loss = eval_loss / nb_eval_steps
        if args.output_mode == "classification":
            preds = np.argmax(preds, axis=1)
        elif args.output_mode == "regression":
            preds = np.squeeze(preds)
        result = compute_metrics(eval_task, preds, out_label_ids)
        results.update(result)

        output_eval_file = os.path.join(eval_output_dir, "eval_results.txt")
        with open(output_eval_file, "w") as writer:
            logger.info("***** Eval results %s *****", prefix)
            for key in sorted(result.keys()):
                logger.info("  %s = %s", key, str(result[key]))
                writer.write("%s = %s\n" % (key, str(result[key])))

    return results


def load_and_cache_examples(args, task, tokenizer, evaluate=False):
    if args.local_rank not in [-1, 0]:
        dist.barrier()

    processor = processors[task]()
    output_mode = output_modes[task]
    cached_features_file = os.path.join(
        args.data_dir, 'cached_{}_{}_{}_{}'.format(
            'dev' if evaluate else 'train',
            list(filter(None, args.model_name_or_path.split('/'))).pop(),
            str(args.max_seq_length), str(task)))

    if os.path.exists(cached_features_file):
        logger.info("Loading features from cached file %s", cached_features_file)
        features = torch.load(cached_features_file)
    else:
        logger.info("Creating features from dataset file at %s", args.data_dir)
        label_list = processor.get_labels()
        if task in ['mnli', 'mnli-mm'] and args.model_type in ['roberta']:
            label_list[1], label_list[2] = label_list[2], label_list[1]
        examples = (processor.get_dev_examples(args.data_dir) if evaluate
                    else processor.get_train_examples(args.data_dir))
        features = convert_examples_to_features(
            examples, label_list, args.max_seq_length, tokenizer, output_mode,
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
            torch.save(features, cached_features_file)

    if args.local_rank == 0:
        dist.barrier()

    all_input_ids   = torch.tensor([f.input_ids   for f in features], dtype=torch.long)
    all_input_mask  = torch.tensor([f.input_mask  for f in features], dtype=torch.long)
    all_segment_ids = torch.tensor([f.segment_ids for f in features], dtype=torch.long)
    if output_mode == "classification":
        all_label_ids = torch.tensor([f.label_id for f in features], dtype=torch.long)
    elif output_mode == "regression":
        all_label_ids = torch.tensor([f.label_id for f in features], dtype=torch.float)

    return TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids)


def main():
    parser = argparse.ArgumentParser()

    # Required
    parser.add_argument("--data_dir",          required=True, type=str)
    parser.add_argument("--model_type",         required=True, type=str)
    parser.add_argument("--model_name_or_path", required=True, type=str)
    parser.add_argument("--task_name",          required=True, type=str)
    parser.add_argument("--output_dir",         required=True, type=str)

    # Optional
    parser.add_argument("--config_name",    default="",    type=str)
    parser.add_argument("--tokenizer_name", default="",    type=str)
    parser.add_argument("--cache_dir",      default="",    type=str)
    parser.add_argument("--max_seq_length", default=128,   type=int)
    parser.add_argument("--do_train",       action='store_true')
    parser.add_argument("--do_eval",        action='store_true')
    parser.add_argument("--do_lower_case",  action='store_true')
    parser.add_argument("--per_device_train_batch_size", default=8,   type=int)
    parser.add_argument("--per_device_eval_batch_size",  default=8,   type=int)
    parser.add_argument("--gradient_accumulation_steps", default=1,   type=int)
    parser.add_argument("--learning_rate",  default=5e-5,  type=float)
    parser.add_argument("--weight_decay",   default=0.0,   type=float)
    parser.add_argument("--adam_epsilon",   default=1e-8,  type=float)
    parser.add_argument("--max_grad_norm",  default=1.0,   type=float)
    parser.add_argument("--num_train_epochs", default=3.0, type=float)
    parser.add_argument("--max_steps",      default=-1,    type=int)
    parser.add_argument("--warmup_steps",   default=0,     type=int)
    parser.add_argument("--no_cuda",        action='store_true')
    parser.add_argument("--overwrite_output_dir", action='store_true')
    parser.add_argument("--overwrite_cache",      action='store_true')
    parser.add_argument("--seed",           default=42,    type=int)
    parser.add_argument("--fp16",           action='store_true')
    parser.add_argument("--fp16_opt_level", default='O1',  type=str)

    # Distributed
    parser.add_argument("--local_rank",  type=int, default=-1)
    parser.add_argument("--master_ip",   type=str, default="127.0.0.1")
    parser.add_argument("--master_port", type=str, default="12345")
    parser.add_argument("--world_size",  type=int, default=1)

    args = parser.parse_args()

    if (os.path.exists(args.output_dir) and os.listdir(args.output_dir)
            and args.do_train and not args.overwrite_output_dir):
        raise ValueError("Output directory ({}) already exists. Use --overwrite_output_dir.".format(
            args.output_dir))

    # Distributed init
    if args.local_rank != -1:
        backend = 'nccl' if (torch.cuda.is_available() and not args.no_cuda) else 'gloo'
        dist.init_process_group(
            backend=backend,
            init_method="tcp://{}:{}".format(args.master_ip, args.master_port),
            world_size=args.world_size,
            rank=args.local_rank,
        )

    args.device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    args.n_gpu = torch.cuda.device_count()

    logging.basicConfig(
        format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
        datefmt='%m/%d/%Y %H:%M:%S',
        level=logging.INFO if args.local_rank in [-1, 0] else logging.WARN,
    )

    set_seed(args)

    args.task_name = args.task_name.lower()
    if args.task_name not in processors:
        raise ValueError("Task not found: %s" % args.task_name)
    processor = processors[args.task_name]()
    args.output_mode = output_modes[args.task_name]
    label_list = processor.get_labels()
    num_labels = len(label_list)

    if args.local_rank not in [-1, 0]:
        dist.barrier()

    args.model_type = args.model_type.lower()
    config_class, model_class, tokenizer_class = MODEL_CLASSES[args.model_type]
    config = config_class.from_pretrained(
        args.config_name if args.config_name else args.model_name_or_path,
        num_labels=num_labels, finetuning_task=args.task_name)
    tokenizer = tokenizer_class.from_pretrained(
        args.tokenizer_name if args.tokenizer_name else args.model_name_or_path,
        do_lower_case=args.do_lower_case)
    model = model_class.from_pretrained(args.model_name_or_path, config=config)

    if args.local_rank == 0:
        dist.barrier()

    model.to(args.device)

    if args.do_train:
        train_dataset = load_and_cache_examples(args, args.task_name, tokenizer, evaluate=False)
        global_step, tr_loss = train(args, train_dataset, model, tokenizer)
        logger.info("global_step=%s, avg_loss=%s", global_step, tr_loss)

    if args.do_eval and args.local_rank in [-1, 0]:
        evaluate(args, model, tokenizer, prefix="final")


if __name__ == "__main__":
    main()
