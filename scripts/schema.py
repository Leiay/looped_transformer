from quinine import (
    tstring,
    tinteger,
    tfloat,
    tboolean,
    stdict,
    tdict,
    tlist,
    default,
    required,
    allowed,
    nullable,
)
from funcy import merge


gpu_schema = {
    "cuda": merge(tboolean, default(True)),
    "n_gpu": merge(tinteger, required)  # which gpu device to use
}

model_schema = {
    "family": merge(tstring, allowed(["gpt2", "gpt2_loop", "gpt2_tying"])),
    "n_positions": merge(tinteger, required),  # maximum context length
    "n_dims": merge(tinteger, required),  # latent dimension
    "n_embd": merge(tinteger, required),
    "n_layer": merge(tinteger, required),
    "n_head": merge(tinteger, required),
    "pred_type": merge(tstring, default("regression")),
    "pretrained_path": merge(tstring, nullable, default(None)),
    "loop_func": merge(tstring, default("z=f(x+z)"), allowed(
        ["z=f(x+z)", "z=f(x*z)"])),
}

curriculum_base_schema = {
    "start": merge(tinteger, required),  # initial parameter
    "end": merge(tinteger, required),  # limit of final value
    "inc": merge(tinteger, required),  # how much to increment each time
    "interval": merge(tinteger, required),  # increment every how many steps
}

curriculum_schema = {
    "dims": stdict(curriculum_base_schema),
    "points": stdict(curriculum_base_schema),
    "loops": stdict(curriculum_base_schema),
}

training_schema = {
    "seed": merge(tinteger, default(42)),
    "task_name": merge(tstring, required),
    "use_fixed_dataset": merge(tboolean, default(False)),
    "train_size": merge(tinteger, default(0)),
    "test_size": merge(tinteger, default(0)),
    "use_ctx": merge(tboolean, default(False)),
    "batch_size": merge(tinteger, default(64)),
    "learning_rate": merge(tfloat, default(3e-4)),
    "weight_decay": merge(tfloat, default(0.)),
    "train_steps": merge(tinteger, default(1000)),
    "save_every_steps": merge(tinteger, default(1000)),  # how often to checkpoint
    "keep_every_steps": merge(tinteger, default(-1)),  # permanent checkpoints
    "resume_id": merge(tstring, nullable, default(None)),  # run uuid64
    "curriculum": stdict(curriculum_schema),
    "n_loop_window": merge(tinteger, default(1)),
    "sparsity": merge(tinteger, default(100)),
    "add_inputs_embeds": merge(tboolean, default(False)),
    "test_idx": merge(tinteger, default(-1)),  # openML dataset
}

wandb_schema = {
    "project": merge(tstring, default("ICL-breakdown")),
    "notes": merge(tstring, default("")),
    "name": merge(tstring, nullable, default(None)),
    "log_every_steps": merge(tinteger, default(10)),
    "timestamp": merge(tstring, nullable)
}

schema = {
    "out_dir": merge(tstring, required),
    "gpu": stdict(gpu_schema),
    "model": stdict(model_schema),
    "training": stdict(training_schema),
    "wandb": stdict(wandb_schema),
    "debug_mode": merge(tboolean, default(False)),
}
