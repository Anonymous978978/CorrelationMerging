import dataclasses
import json
import os
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

from transformers.file_utils import cached_property, is_torch_available, is_torch_tpu_available, torch_required
from transformers.trainer_utils import EvaluationStrategy
from transformers.utils import logging

if is_torch_available():
    import torch

if is_torch_tpu_available():
    import torch_xla.core.xla_model as xm

logger = logging.get_logger(__name__)


def default_logdir() -> str:
    """
    Same default as PyTorch
    """
    import socket
    from datetime import datetime

    current_time = datetime.now().strftime("%b%d_%H-%M-%S")
    return os.path.join("runs", current_time + "_" + socket.gethostname())


@dataclass
class TrainingArguments:
    output_dir: str = field(
        default="./model_output/",
        metadata={"help": "The output directory where the model predictions and checkpoints will be written."}
    )
    overwrite_output_dir: bool = field(
        default=False,
        metadata={
            "help": (
                "Overwrite the content of the output directory."
                "Use this to continue training if output_dir points to a checkpoint directory."
            )
        },
    )

    do_train: bool = field(default=not True, metadata={"help": "Whether to run training."})
    do_eval: bool = field(default=True, metadata={"help": "Whether to run eval on the dev set."})
    do_predict: bool = field(default=True, metadata={"help": "Whether to run predictions on the test set."})
    model_parallel: bool = field(
        default=False,
        metadata={
            "help": (
                "If there are more than one devices, whether to use model parallelism to distribute the "
                "model's modules across devices."
            )
        },
    )
    evaluation_strategy: EvaluationStrategy = field(
        default="no",
        metadata={"help": "Run evaluation during training at each logging step."},
    )
    prediction_loss_only: bool = field(
        default=False,
        metadata={"help": "When performing evaluation and predictions, only returns the loss."},
    )

    per_device_train_batch_size: int = field(
        default=8, metadata={"help": "Batch size per GPU/TPU core/CPU for training."}
    )
    per_device_eval_batch_size: int = field(
        default=8, metadata={"help": "Batch size per GPU/TPU core/CPU for evaluation."}
    )

    per_gpu_train_batch_size: Optional[int] = field(
        default=None,
        metadata={
            "help": "Deprecated, the use of `--per_device_train_batch_size` is preferred. "
                    "Batch size per GPU/TPU core/CPU for training."
        },
    )
    per_gpu_eval_batch_size: Optional[int] = field(
        default=32,
        metadata={
            "help": "Deprecated, the use of `--per_device_eval_batch_size` is preferred."
                    "Batch size per GPU/TPU core/CPU for evaluation."
        },
    )

    gradient_accumulation_steps: int = field(
        default=1,
        metadata={"help": "Number of updates steps to accumulate before performing a backward/update pass."},
    )
    eval_accumulation_steps: Optional[int] = field(
        default=None,
        metadata={"help": "Number of predictions steps to accumulate before moving the tensors to the CPU."},
    )

    learning_rate: float = field(default=5e-5, metadata={"help": "The initial learning rate for Adam."})
    weight_decay: float = field(default=0.0001, metadata={"help": "Weight decay if we apply some."})
    adam_beta1: float = field(default=0.9, metadata={"help": "Beta1 for Adam optimizer"})
    adam_beta2: float = field(default=0.98, metadata={"help": "Beta2 for Adam optimizer"})
    # adam_beta2: float = field(default=0.999, metadata={"help": "Beta2 for Adam optimizer"})
    adam_epsilon: float = field(default=1e-8, metadata={"help": "Epsilon for Adam optimizer."})
    max_grad_norm: float = field(default=1.0, metadata={"help": "Max gradient norm."})

    num_train_epochs: float = field(default=3.0, metadata={"help": "Total number of training epochs to perform."})
    max_steps: int = field(
        default=-1,
        metadata={"help": "If > 0: set total number of training steps to perform. Override num_train_epochs."},
    )
    warmup_steps: int = field(default=0, metadata={"help": "Linear warmup over warmup_steps."})

    logging_dir: Optional[str] = field(default_factory=default_logdir, metadata={"help": "Tensorboard log dir."})
    logging_first_step: bool = field(default=False, metadata={"help": "Log the first global_step"})
    logging_steps: int = field(default=500, metadata={"help": "Log every X updates steps."})
    save_steps: int = field(default=500, metadata={"help": "Save checkpoint every X updates steps."})
    save_total_limit: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "Limit the total amount of checkpoints."
                "Deletes the older checkpoints in the output_dir. Default is unlimited checkpoints"
            )
        },
    )
    no_cuda: bool = field(default=False, metadata={"help": "Do not use CUDA even when it is available"})
    seed: int = field(default=42, metadata={"help": "random seed for initialization"})

    fp16: bool = field(
        default=False,
        metadata={"help": "Whether to use 16-bit (mixed) precision (through NVIDIA Apex) instead of 32-bit"},
    )
    fp16_opt_level: str = field(
        default="O1",
        metadata={
            "help": (
                "For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
                "See details at https://nvidia.github.io/apex/amp.html"
            )
        },
    )
    local_rank: int = field(default=-1, metadata={"help": "For distributed training: local_rank"})

    tpu_num_cores: Optional[int] = field(
        default=None, metadata={"help": "TPU: Number of TPU cores (automatically passed by launcher script_v0)"}
    )
    tpu_metrics_debug: bool = field(
        default=False,
        metadata={"help": "Deprecated, the use of `--debug` is preferred. TPU: Whether to print debug metrics"},
    )
    debug: bool = field(default=False, metadata={"help": "Whether to print debug metrics on TPU"})

    dataloader_drop_last: bool = field(
        default=False, metadata={"help": "Drop the last incomplete batch if it is not divisible by the batch size."}
    )
    eval_steps: int = field(default=None, metadata={"help": "Run an evaluation every X steps."})
    dataloader_num_workers: int = field(
        default=0,
        metadata={
            "help": "Number of subprocesses to use for data loading (PyTorch only). 0 means that the data will be loaded in the main process."
        },
    )

    past_index: int = field(
        default=-1,
        metadata={"help": "If >=0, uses the corresponding part of the output as the past state for next step."},
    )

    run_name: Optional[str] = field(
        default=None, metadata={"help": "An optional descriptor for the run. Notably used for wandb logging."}
    )
    disable_tqdm: Optional[bool] = field(
        default=None, metadata={"help": "Whether or not to disable the tqdm progress bars."}
    )

    remove_unused_columns: Optional[bool] = field(
        default=True, metadata={"help": "Remove columns not required by the model when using an nlp.Dataset."}
    )
    label_names: Optional[List[str]] = field(
        default=None, metadata={"help": "The list of keys in your dictionary of inputs that correspond to the labels."}
    )

    load_best_model_at_end: Optional[bool] = field(
        default=False,
        metadata={"help": "Whether or not to load the best model found during training at the end of training."},
    )
    metric_for_best_model: Optional[str] = field(
        default=None, metadata={"help": "The metric to use to compare two different models."}
    )
    greater_is_better: Optional[bool] = field(
        default=None, metadata={"help": "Whether the `metric_for_best_model` should be maximized or not."}
    )
    ignore_skip: bool = field(
        default=False,
        metadata={
            "help": "When resuming training, whether or not to skip the first epochs and batches to get to the same training data."
        },
    )
    fp16_backend: str = field(
        default="auto",
        metadata={"help": "The backend to be used for mixed precision.", "choices": ["auto", "amp", "apex"]},
    )
    sharded_ddp: bool = field(
        default=False,
        metadata={"help": "Whether or not to use sharded DDP training (in distributed training only)."},
    )

    fitlog_dir: str = field(default="./logs")
    multi_head_num: int = field(default=32)
    sample_file: str = field(default="")
    max_length: int = field(default=0)
    sim_embed: str = field(default="/remote-home/train_data/adv_learn/counter-fitted-vectors.txt")
    cosine_npy: str = field(default="/remote-home/train_data/adv_learn/cos_sim_counter_fitting.npy")
    train_multi_head: bool = field(default=False)
    model_path: str = field(default="roberta-base-uncased")
    load_trained_model: bool = field(default=False)
    load_model_pattern: str = field(default="knn_bert")
    num_labels: int = field(default=2)
    known_ratio: float = field(default=1)
    data: str = field(default="clinc_full")

    lmcl: bool = field(default=False)
    cl_model: str = field(default='')
    ## number of layer size
    rnn_number_layers: int = field(default=1)

    sup_cont: bool = field(
        default=False, metadata={"help": "supervised contrastive learning!"}
    )
    supcont_pre_epoches: int = field(default=10)
    ind_pre_epoches: int = field(default=20)
    norm_coef: float = field(default=0.1)
    setting: str = field(default='lof')
    cl_mode: int = field(default=1)
    hidden_dim: int = field(default=128)
    clip: float = field(default=0.25)
    mode: str = field(default='test')
    temperature: float = field(default=0.3)
    auto_th: float = field(default=25)

    def __post_init__(self):
        if self.disable_tqdm is None:
            self.disable_tqdm = logger.getEffectiveLevel() > logging.WARN
        self.evaluation_strategy = EvaluationStrategy(self.evaluation_strategy)
        if self.do_eval is False and self.evaluation_strategy != EvaluationStrategy.NO:
            self.do_eval = True
        if self.eval_steps is None:
            self.eval_steps = self.logging_steps

        if self.load_best_model_at_end and self.metric_for_best_model is None:
            self.metric_for_best_model = "loss"
        if self.greater_is_better is None and self.metric_for_best_model is not None:
            self.greater_is_better = self.metric_for_best_model not in ["loss", "eval_loss"]
        if self.run_name is None:
            self.run_name = self.output_dir

        if is_torch_available() and self.device.type != "cuda" and self.fp16:
            raise ValueError("AMP (`--fp16`) can only be used on CUDA devices.")

    @property
    def train_batch_size(self) -> int:
        if self.per_gpu_train_batch_size:
            logger.warning(
                "Using deprecated `--per_gpu_train_batch_size` argument which will be removed in a future "
                "version. Using `--per_device_train_batch_size` is preferred."
            )
        per_device_batch_size = self.per_gpu_train_batch_size or self.per_device_train_batch_size
        if not self.model_parallel:
            train_batch_size = per_device_batch_size * max(1, self.n_gpu)
        else:
            train_batch_size = per_device_batch_size
        return train_batch_size

    @property
    def eval_batch_size(self) -> int:
        """
        The actual batch size for evaluation (may differ from :obj:`per_gpu_eval_batch_size` in distributed training).
        """
        if self.per_gpu_eval_batch_size:
            logger.warning(
                "Using deprecated `--per_gpu_eval_batch_size` argument which will be removed in a future "
                "version. Using `--per_device_eval_batch_size` is preferred."
            )
        per_device_batch_size = self.per_gpu_eval_batch_size or self.per_device_eval_batch_size
        if not self.model_parallel:
            eval_batch_size = per_device_batch_size * max(1, self.n_gpu)
        else:
            eval_batch_size = per_device_batch_size
        return eval_batch_size

    @cached_property
    @torch_required
    def _setup_devices(self) -> Tuple["torch.device", int]:
        logger.info("PyTorch: setting up devices")
        if self.no_cuda:
            device = torch.device("cpu")
            n_gpu = 0
        elif is_torch_tpu_available():
            device = xm.xla_device()
            n_gpu = 0
        elif self.local_rank == -1:
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            n_gpu = torch.cuda.device_count()
        else:
            torch.distributed.init_process_group(backend="nccl")
            device = torch.device("cuda", self.local_rank)
            n_gpu = 1

        if device.type == "cuda":
            torch.cuda.set_device(device)

        return device, n_gpu

    @property
    @torch_required
    def device(self) -> "torch.device":
        return self._setup_devices[0]

    @property
    @torch_required
    def n_gpu(self):
        return self._setup_devices[1]

    @property
    @torch_required
    def parallel_mode(self):
        if is_torch_tpu_available():
            return ParallelMode.TPU
        elif self.local_rank != -1:
            return ParallelMode.DISTRIBUTED
        elif self.n_gpu > 1:
            return ParallelMode.NOT_DISTRIBUTED
        else:
            return ParallelMode.NOT_PARALLEL

    def to_dict(self):
        """
        Serializes this instance while replace `Enum` by their values (for JSON serialization support).
        """
        d = dataclasses.asdict(self)
        for k, v in d.items():
            if isinstance(v, Enum):
                d[k] = v.value
        return d

    def to_json_string(self):
        """
        Serializes this instance to a JSON string.
        """
        return json.dumps(self.to_dict(), indent=2)

    def to_sanitized_dict(self) -> Dict[str, Any]:
        """
        Sanitized serialization to use with TensorBoard’s hparams
        """
        d = self.to_dict()
        d = {**d, **{"train_batch_size": self.train_batch_size, "eval_batch_size": self.eval_batch_size}}

        valid_types = [bool, int, float, str]
        if is_torch_available():
            valid_types.append(torch.Tensor)

        return {k: v if type(v) in valid_types else str(v) for k, v in d.items()}


class ParallelMode(Enum):
    NOT_PARALLEL = "not_parallel"
    NOT_DISTRIBUTED = "not_distributed"
    DISTRIBUTED = "distributed"
    TPU = "tpu"
