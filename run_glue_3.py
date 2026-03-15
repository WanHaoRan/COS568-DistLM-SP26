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
Part 3: Distributed Data Parallel Training using torch.nn.parallel.DistributedDataParallel.

DDP handles gradient synchronization automatically as part of the backward pass
(via hooks that overlap communication with computation), so no manual gradient
handling is required here.

Run with:
  python run_glue_3.py [other args] \
      --master_ip <ip> --master_port <port> \
      --world_size 4 --local_rank <0|1|2|3>
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
from torch.profiler import profile, ProfilerActivity, schedule

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    _HAS_MATPLOTLIB = True
except ImportError:
    _HAS_MATPLOTLIB = False

from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler,
                               TensorDataset)
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from tqdm import tqdm, trange

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
    'bert':    (BertConfig,    BertForSequenceClassification,   BertTokenizer),
    'xlnet':   (XLNetConfig,   XLNetForSequenceClassification,  XLNetTokenizer),
    'xlm':     (XLMConfig,     XLMForSequenceClassification,    XLMTokenizer),
    'roberta': (RobertaConfig, RobertaForSequenceClassification, RobertaTokenizer),
}


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.cuda.manual_seed_all(args.seed)


def train(args, train_dataset, model, tokenizer):
    """
    Train with PyTorch DistributedDataParallel.

    The model is wrapped in DDP before train() is called, which automatically
    synchronizes gradients across all workers during the backward pass via
    efficient all-reduce (overlapping communication with computation layer-by-layer).
    No manual gradient aggregation is needed.
    """

    args.train_batch_size = args.per_device_train_batch_size
    if args.master_ip is not None:
        world_size = torch.distributed.get_world_size()
        rank = torch.distributed.get_rank()
        train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True)
    else:
        train_sampler = RandomSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.train_batch_size)

    if args.max_steps > 0:
        t_total = args.max_steps
        args.num_train_epochs = args.max_steps // (len(train_dataloader) // args.gradient_accumulation_steps) + 1
    else:
        t_total = len(train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs

    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': args.weight_decay},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0},
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    scheduler = WarmupLinearSchedule(optimizer, warmup_steps=args.warmup_steps, t_total=t_total)

    if args.fp16:
        try:
            from apex import amp
        except ImportError:
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
        model, optimizer = amp.initialize(model, optimizer, opt_level=args.fp16_opt_level)

    logger.info("***** Running training (Task 3: DDP) *****")
    logger.info("  Num examples        = %d", len(train_dataset))
    logger.info("  Num Epochs          = %d", args.num_train_epochs)
    logger.info("  Per-device batch    = %d", args.per_device_train_batch_size)
    logger.info("  World size          = %d", args.world_size)
    logger.info("  Total batch size    = %d", args.train_batch_size * args.world_size * args.gradient_accumulation_steps)
    logger.info("  Total optim steps   = %d", t_total)

    global_step = 0
    tr_loss = 0.0
    model.zero_grad()
    rank = torch.distributed.get_rank() if args.master_ip is not None else 0

    # Timing: first iteration discarded, then average reported
    iter_times = []
    loss_curve = []
    step_loss_accum = 0.0
    t_iter_start = None

    # Task 4: profiling
    trace_addr = None
    if args.profile:
        os.makedirs(args.output_dir, exist_ok=True)
        trace_addr = os.path.join(args.output_dir, "chrome_trace_part3_rank{}.json".format(rank))
        sched = schedule(wait=1, warmup=0, active=3, repeat=1)
        activities = [ProfilerActivity.CPU, ProfilerActivity.CUDA] if (torch.cuda.is_available() and not args.no_cuda) else [ProfilerActivity.CPU]
        prof_context = profile(
            activities=activities,
            schedule=sched,
            on_trace_ready=lambda p: p.export_chrome_trace(trace_addr),
        )
        logger.info("[Task 4] Profiling enabled. Trace will be saved to %s", trace_addr)
    else:
        prof_context = contextlib.nullcontext(enter_result=None)

    train_iterator = trange(int(args.num_train_epochs), desc="Epoch", disable=args.local_rank not in [-1, 0])
    set_seed(args)

    with prof_context as prof:
        for epoch in train_iterator:
            if args.master_ip is not None:
                train_sampler.set_epoch(epoch)
            epoch_iterator = tqdm(train_dataloader, desc="Iteration", disable=args.local_rank not in [-1, 0])
            for step, batch in enumerate(epoch_iterator):
                if step % args.gradient_accumulation_steps == 0:
                    t_iter_start = time.perf_counter()
                    step_loss_accum = 0.0

                model.train()
                batch = tuple(t.to(args.device) for t in batch)
                inputs = {
                    'input_ids':      batch[0],
                    'attention_mask': batch[1],
                    'token_type_ids': batch[2] if args.model_type in ['bert', 'xlnet'] else None,
                    'labels':         batch[3],
                }

                is_update_step = ((step + 1) % args.gradient_accumulation_steps == 0) or ((step + 1) == len(train_dataloader))
                use_no_sync = (args.master_ip is not None) and (not is_update_step)

                if use_no_sync:
                    with model.no_sync():
                        outputs = model(**inputs)
                        loss = outputs[0]
                        if args.gradient_accumulation_steps > 1:
                            loss = loss / args.gradient_accumulation_steps
                        if args.fp16:
                            with amp.scale_loss(loss, optimizer) as scaled_loss:
                                scaled_loss.backward()
                        else:
                            loss.backward()
                else:
                    outputs = model(**inputs)
                    loss = outputs[0]
                    if args.gradient_accumulation_steps > 1:
                        loss = loss / args.gradient_accumulation_steps
                    if args.fp16:
                        with amp.scale_loss(loss, optimizer) as scaled_loss:
                            scaled_loss.backward()
                    else:
                        # DDP hooks fire during backward and automatically all-reduce
                        # gradients across workers before backward() returns.
                        loss.backward()

                tr_loss += loss.item()
                step_loss_accum += loss.item()

                if step < 5:
                    logger.info("  [rank %d] step %d loss = %.4f", args.local_rank, step, loss.item())

                if is_update_step:
                    if args.fp16:
                        torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), args.max_grad_norm)
                    else:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                    iter_times.append(time.perf_counter() - t_iter_start)
                    loss_curve.append((global_step + 1, step_loss_accum))
                    optimizer.step()
                    scheduler.step()
                    model.zero_grad()
                    global_step += 1
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

            # Evaluate after each epoch on rank 0 only — DDP ensures all workers
            # have identical model weights, so rank 0 evaluation is representative.
            if args.do_eval and args.local_rank in [-1, 0]:
                evaluate(args, model, tokenizer, prefix="epoch_{}".format(epoch + 1))

    if trace_addr is not None:
        logger.info("[Task 4] Trace saved at %s (open in Chrome: chrome://tracing)", trace_addr)

    # Save timing and loss curve to files
    os.makedirs(args.output_dir, exist_ok=True)
    if len(iter_times) > 1:
        avg_time_per_iter = sum(iter_times[1:]) / (len(iter_times) - 1)
        logger.info("[Task 3] Rank %d: avg iter time (excl. first): %.4f sec (%d iters)",
                    rank, avg_time_per_iter, len(iter_times) - 1)
        with open(os.path.join(args.output_dir, "avg_time_part3_rank{}.txt".format(rank)), "w") as f:
            f.write("rank\tavg_time_per_iter_sec\tnum_iterations\n")
            f.write("%d\t%.6f\t%d\n" % (rank, avg_time_per_iter, len(iter_times) - 1))

    with open(os.path.join(args.output_dir, "loss_curve_part3_rank{}.txt".format(rank)), "w") as f:
        f.write("step\tloss\n")
        for s, l in loss_curve:
            f.write("%d\t%.6f\n" % (s, l))
    logger.info("[Task 3] Rank %d: loss curve written to %s", rank,
                os.path.join(args.output_dir, "loss_curve_part3_rank{}.txt".format(rank)))

    if loss_curve and _HAS_MATPLOTLIB:
        steps = [s for s, _ in loss_curve]
        losses = [l for _, l in loss_curve]
        plt.figure(figsize=(8, 5))
        plt.plot(steps, losses, "b-", linewidth=1)
        plt.xlabel("Step")
        plt.ylabel("Loss")
        plt.title("Training loss (rank {})".format(rank))
        plt.grid(True, alpha=0.3)
        plot_path = os.path.join(args.output_dir, "loss_curve_part3_rank{}.png".format(rank))
        plt.savefig(plot_path, dpi=150, bbox_inches="tight")
        plt.close()
    elif loss_curve and not _HAS_MATPLOTLIB:
        logger.warning("matplotlib not available; skipping loss curve plot.")

    return global_step, tr_loss / global_step


def evaluate(args, model, tokenizer, prefix=""):
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

        logger.info("***** Running evaluation %s *****", prefix)
        logger.info("  Num examples = %d", len(eval_dataset))
        logger.info("  Batch size   = %d", args.eval_batch_size)

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
            logger.info("***** Eval results %s *****", prefix)
            for key in sorted(result.keys()):
                logger.info("  %s = %s", key, str(result[key]))
                writer.write("%s = %s\n" % (key, str(result[key])))

    return results


def load_and_cache_examples(args, task, tokenizer, evaluate=False):
    # Only block non-rank-0 workers during training-time cache creation
    if (not evaluate) and args.master_ip is not None and args.local_rank not in [-1, 0]:
        torch.distributed.barrier()

    processor = processors[task]()
    output_mode = output_modes[task]
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
            label_list[1], label_list[2] = label_list[2], label_list[1]
        examples = processor.get_dev_examples(args.data_dir) if evaluate else processor.get_train_examples(args.data_dir)
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
            logger.info("Saving features into cached file %s", cached_features_file)
            torch.save(features, cached_features_file)

    # Release non-rank-0 workers after rank 0 finishes writing the cache
    if (not evaluate) and args.master_ip is not None and args.local_rank == 0:
        torch.distributed.barrier()

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
    parser.add_argument("--data_dir", default=None, type=str, required=True)
    parser.add_argument("--model_type", default=None, type=str, required=True)
    parser.add_argument("--model_name_or_path", default=None, type=str, required=True)
    parser.add_argument("--task_name", default=None, type=str, required=True)
    parser.add_argument("--output_dir", default=None, type=str, required=True)

    ## Other parameters
    parser.add_argument("--config_name", default="", type=str)
    parser.add_argument("--tokenizer_name", default="", type=str)
    parser.add_argument("--cache_dir", default="", type=str)
    parser.add_argument("--max_seq_length", default=128, type=int)
    parser.add_argument("--do_train", action='store_true')
    parser.add_argument("--do_eval", action='store_true')
    parser.add_argument("--do_lower_case", action='store_true')
    parser.add_argument("--per_device_train_batch_size", default=8, type=int)
    parser.add_argument("--per_device_eval_batch_size", default=8, type=int)
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1)
    parser.add_argument("--learning_rate", default=5e-5, type=float)
    parser.add_argument("--weight_decay", default=0.0, type=float)
    parser.add_argument("--adam_epsilon", default=1e-8, type=float)
    parser.add_argument("--max_grad_norm", default=1.0, type=float)
    parser.add_argument("--num_train_epochs", default=3.0, type=float)
    parser.add_argument("--max_steps", default=-1, type=int)
    parser.add_argument("--warmup_steps", default=0, type=int)
    parser.add_argument("--no_cuda", action='store_true')
    parser.add_argument('--overwrite_output_dir', action='store_true')
    parser.add_argument('--overwrite_cache', action='store_true')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--fp16', action='store_true')
    parser.add_argument('--fp16_opt_level', type=str, default='O1')
    parser.add_argument("--local_rank", type=int, default=-1)

    ## Distributed training
    parser.add_argument("--master_ip", default=None, type=str,
                        help="Master node IP. If set, init_process_group is called.")
    parser.add_argument("--master_port", default=29500, type=int)
    parser.add_argument("--world_size", default=1, type=int)

    ## Task 4: profiling
    parser.add_argument("--profile", action="store_true", default=False,
                        help="Profile 3 training steps (skip first), save Chrome trace.")

    args = parser.parse_args()

    # Initialize distributed process group
    if args.master_ip is not None:
        backend = "nccl" if torch.cuda.is_available() and not args.no_cuda else "gloo"
        torch.distributed.init_process_group(
            backend=backend,
            init_method="tcp://{}:{}".format(args.master_ip, args.master_port),
            world_size=args.world_size,
            rank=args.local_rank,
        )

    if os.path.exists(args.output_dir) and os.listdir(args.output_dir) and args.do_train and not args.overwrite_output_dir:
        raise ValueError("Output directory ({}) already exists. Use --overwrite_output_dir.".format(args.output_dir))

    # Device setup
    if torch.cuda.is_available() and not args.no_cuda:
        if args.master_ip is not None:
            torch.cuda.set_device(0)
            args.device = torch.device("cuda:0")
        else:
            args.device = torch.device("cuda")
    else:
        args.device = torch.device("cpu")
    args.n_gpu = torch.cuda.device_count() if (torch.cuda.is_available() and not args.no_cuda) else 0

    logging.basicConfig(
        format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
        datefmt='%m/%d/%Y %H:%M:%S',
        level=logging.INFO if args.local_rank in [-1, 0] else logging.WARN,
    )
    logger.warning("Process rank: %s, device: %s, distributed training: %s, 16-bits training: %s",
                   args.local_rank, args.device, bool(args.local_rank != -1), args.fp16)

    set_seed(args)

    args.task_name = args.task_name.lower()
    if args.task_name not in processors:
        raise ValueError("Task not found: %s" % args.task_name)
    processor = processors[args.task_name]()
    args.output_mode = output_modes[args.task_name]
    label_list = processor.get_labels()
    num_labels = len(label_list)

    # Only rank 0 downloads model; others wait
    if args.local_rank not in [-1, 0]:
        torch.distributed.barrier()

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
        torch.distributed.barrier()

    model.to(args.device)

    if args.master_ip is not None:
        if args.device.type == "cuda":
            model = DDP(model, device_ids=[0], output_device=0)
        else:
            model = DDP(model)

    logger.info("Training/evaluation parameters %s", args)

    if args.do_train:
        train_dataset = load_and_cache_examples(args, args.task_name, tokenizer, evaluate=False)
        global_step, tr_loss = train(args, train_dataset, model, tokenizer)
        logger.info(" global_step = %s, average loss = %s", global_step, tr_loss)

    # Final evaluation on rank 0 only — no barrier needed since train() already
    # handles per-epoch eval and all ranks have exited the training loop cleanly.
    if args.do_eval and args.local_rank in [-1, 0]:
        evaluate(args, model, tokenizer, prefix="final")


if __name__ == "__main__":
    main()
