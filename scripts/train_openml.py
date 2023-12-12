import os
import datetime

from quinine import QuinineArgumentParser
from tqdm import tqdm
import torch
import torch.nn.functional as F
import yaml
import copy
import random
import numpy as np

from curriculum import Curriculum
from schema import schema
from models import build_model
from tasks import get_task_sampler
from main_utils import init_device, get_run_id, load_pretrained_model
# from eval import get_run_metrics


import wandb

torch.backends.cudnn.benchmark = True
NUM_POINTS = 41


def train_step(args, curriculum, model, xs, ys, optimizer, ctx, scaler, eval=False):
    if args.model.family in ['gpt2', 'gpt2_tying']:
        B, n = ys.shape
        if ctx is not None:
            with ctx:
                y_pred = model(xs, ys, add_inputs_embeds=args.training.add_inputs_embeds)  # [B, n]
                if eval:
                    pred = y_pred.view(B * n, -1).data.max(1, keepdim=True)[1]  # get the index of the max log-probability
                    acc = pred.eq(ys.view(B * n).data.view_as(pred)).cpu().view(B, n)[:, -1].sum().item() / (B)
                    loss = 0
                else:
                    loss = F.cross_entropy(y_pred.view(B * n, -1), ys.view(B * n).long())
                    acc = 0
        else:
            y_pred = model(xs, ys, add_inputs_embeds=args.training.add_inputs_embeds)  # [B, n]
            if eval:
                pred = y_pred.view(B * n, -1).data.max(1, keepdim=True)[1]  # get the index of the max log-probability
                acc = pred.eq(ys.view(B * n).data.view_as(pred)).cpu().view(B, n)[:, -1].sum().item() / (B)
                loss = 0
            else:
                loss = F.cross_entropy(y_pred.view(B * n, -1), ys.view(B * n).long())
                acc = 0
    elif args.model.family in ['gpt2_loop']:
        n_loops = curriculum.n_loops  # K
        B = ys.shape[0]
        if ctx is not None:
            with ctx:
                horizon_start = max(0, n_loops - args.training.n_loop_window)
                y_pred_list = model(xs, ys, horizon_start, n_loops)
                # list of [B, n], length K
                y_pred_arr = torch.cat(y_pred_list, dim=0)  # [B * K, n]
                y_star_arr = torch.cat([ys] * len(y_pred_list), dim=0)  # [B * K, n]
                BK, n = y_star_arr.shape
                if eval:
                    pred = y_pred_arr.view(BK * n, -1).data.max(1, keepdim=True)[1]  # get the index of the max log-probability
                    acc = pred.eq(y_star_arr.view(BK * n).data.view_as(pred)).cpu().view(-1, B, n)[-1, :, -1].sum().item() / (B)
                    loss = 0
                else:
                    loss = F.cross_entropy(y_pred_arr.view(BK * n, -1), y_star_arr.view(BK * n).long(), reduction='none')
                    acc = 0
        else:
            horizon_start = max(0, n_loops - args.training.n_loop_window)
            y_pred_list = model(xs, ys, horizon_start, n_loops)
            # list of [B, n], length K
            y_pred_arr = torch.cat(y_pred_list, dim=0)  # [B * K, n]
            y_star_arr = torch.cat([ys] * len(y_pred_list), dim=0)  # [B * K, n]
            BK, n = y_star_arr.shape
            if eval:
                pred = y_pred_arr.view(BK * n, -1).data.max(1, keepdim=True)[1]  # get the index of the max log-probability
                acc = pred.eq(y_star_arr.view(BK * n).data.view_as(pred)).cpu().view(-1, B, n)[-1, :, -1].sum().item() / (B)
                loss = 0
            else:
                loss = F.cross_entropy(y_pred_arr.view(BK * n, -1), y_star_arr.view(BK * n).long())
                acc = 0

    if not eval:  # update
        if args.training.use_ctx:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()
        optimizer.zero_grad(set_to_none=True)
    return loss, acc


def get_batch(args, dataset_id, openml_datasets, device):

    X, y = openml_datasets[dataset_id]['X'], openml_datasets[dataset_id]['y']
    batch_size = min(args.training.batch_size, int(X.shape[0] // NUM_POINTS) - 1)
    batch_ids = random.sample(range(0, X.shape[0]), batch_size * NUM_POINTS)
    xs, ys = X[batch_ids], y[batch_ids]
    xs, ys = torch.tensor(xs).to(device), torch.tensor(ys).to(device)
    # make xs to have dimension d=20
    d_x = xs.shape[-1]
    xs = xs.reshape(-1, NUM_POINTS, d_x)
    B, n, d_x = xs.shape
    xs = torch.cat(
        [
            torch.zeros(B, n, args.model.n_dims - d_x, device=device),
            xs,
        ],
        axis=2,
    )  # xs.shape should be [B, n, d] now
    ys = ys.view(B, n)
    return xs.float(), ys.float()


def main(args, device):
    # TORCH 2.0 ZONE ###############################
    torch.set_float32_matmul_precision('highest')
    torch.backends.cuda.matmul.allow_tf32 = True  # allow tf32 on matmul
    torch.backends.cudnn.allow_tf32 = True  # allow tf32 on cudnn
    dtype = 'float16'  # 'bfloat16', 'float32'
    ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
    if args.training.use_ctx:
        ctx = torch.amp.autocast(device_type='cuda', dtype=ptdtype, cache_enabled=False)
    else:
        ctx = None
    ################################################
    torch.manual_seed(args.training.seed)
    model = build_model(args.model)
    # model = torch.compile(model)

    model.to(device)
    model.train()

    optimizer = torch.optim.Adam(
        model.parameters(), lr=args.training.learning_rate, weight_decay=args.training.weight_decay)
    scaler = torch.cuda.amp.GradScaler(enabled=(dtype == 'float16'))
    curriculum = Curriculum(args.training.curriculum)

    # Here the model load the pretrained model
    args, model, optimizer, curriculum, state_path, starting_step = load_pretrained_model(
        args, model, optimizer, curriculum, device)

    root = 'data/'
    import pickle
    openml_datasets_train = pickle.load(open(root + 'openml_train2.npy', 'rb'))
    openml_datasets_test = pickle.load(open(root + 'openml_test2.npy', 'rb'))
    dataset_id = list(openml_datasets_train.keys())

    # start iterate over datasets
    loss_bucket = []
    loss_dict = {}

    idx_id = args.training.test_idx  # Here we specify the test_id
    assert idx_id >= 0
    train_dataset_ids = copy.deepcopy(dataset_id)
    test_dataset_id = dataset_id[idx_id]
    train_dataset_ids.remove(test_dataset_id)

    wandb.init(
        dir=args.out_dir,
        project=args.wandb.project,
        config=args.__dict__,
        notes=args.wandb.notes,
        name=args.wandb.name,
        mode="disabled" if args.debug_mode else "online",
        resume=True,
    )

    pbar = tqdm(range(starting_step, args.training.train_steps))
    for i in pbar:
        # select a dataset at random
        train_dataset_id = random.choice(train_dataset_ids)
        xs, ys = get_batch(args, train_dataset_id, openml_datasets_train, device)

        train_loss, _ = train_step(args, curriculum, model, xs, ys, optimizer, ctx, scaler)

        # EVALUATION ======================================
        if i % args.wandb.log_every_steps == 0:
            with torch.no_grad():
                xs, ys = get_batch(args, test_dataset_id, openml_datasets_test, device)
                _, acc = train_step(args, curriculum, model, xs, ys, optimizer, ctx, scaler, eval=True)
            wandb.log(
                {
                    "train_loss": train_loss.detach(),
                    "test_acc": acc,
                    "loop_times": curriculum.n_loops,
                    "lr": optimizer.param_groups[0]['lr'],
                },
                step=i,
            )
        curriculum.update()
        pbar.set_description(f"loss {train_loss.detach()}")
        if i % args.training.save_every_steps == 0:
            training_state = {
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "train_step": i,
            }
            torch.save(training_state, state_path)
        if (
                args.training.keep_every_steps > 0
                and i % args.training.keep_every_steps == 0
                and i > 0
        ) or (i == args.training.train_steps - 1):
            torch.save({'model': model.state_dict()},
                       os.path.join(args.out_dir, f"model_{i}.pt"))

    wandb.finish()


if __name__ == "__main__":
    parser = QuinineArgumentParser(schema=schema)
    args = parser.parse_quinfig()
    print(f"Running with: {args}")

    device = init_device(args)

    if args.debug_mode:
        args.out_dir = "./results/debug"

    run_id = args.training.resume_id
    if run_id is None:
        run_id = get_run_id(args)  # str(uuid.uuid4())

    out_dir = os.path.join(args.out_dir, run_id)
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    args.out_dir = out_dir
    # add a timestamp here, if resumed, this will be the resumed time
    args.wandb['timestamp'] = datetime.datetime.now().strftime("%m/%d/%Y, %H:%M:%S")

    with open(os.path.join(out_dir, "config.yaml"), "w") as yaml_file:
        yaml.dump(args.__dict__, yaml_file, default_flow_style=False)

    main(args, device)

