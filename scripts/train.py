import os
import datetime

from quinine import QuinineArgumentParser
from tqdm import tqdm
import torch
import yaml

from curriculum import Curriculum
from schema import schema
from models import build_model
from tasks import get_task_sampler
from main_utils import init_device, get_run_id, load_pretrained_model
# from eval import get_run_metrics


import wandb

torch.backends.cudnn.benchmark = True


def calculate_gradient_norm(model):
    total_norm = 0.0
    norm_dict = {}
    for n, p in model.named_parameters():
        param_norm = p.grad.data.norm(2)
        total_norm += param_norm.item() ** 2
        norm_dict[n] = param_norm
    total_norm = total_norm ** (1. / 2)
    return norm_dict, total_norm


def train_step(args, curriculum, model, xs, ys, optimizer, ctx, scaler):
    if args.model.family in ['gpt2', 'gpt2_tying']:
        if ctx is not None:
            with ctx:
                y_pred = model(xs, ys, add_inputs_embeds=args.training.add_inputs_embeds)  # [B, n]
                # list of [B, n], length K + 1, get rid of the 0-th one
                loss = (ys - y_pred).square().mean()  # auto on both K and n (number of in context samples)
        else:
            y_pred = model(xs, ys, add_inputs_embeds=args.training.add_inputs_embeds)  # [B, n]
            # list of [B, n], length K + 1, get rid of the 0-th one
            loss = (ys - y_pred).square().mean()  # auto on both K and n (number of in context samples)
    elif args.model.family in ['gpt2_loop']:
        n_loops = curriculum.n_loops  # K
        if ctx is not None:
            with ctx:
                horizon_start = max(0, n_loops - args.training.n_loop_window)
                y_pred_list = model(xs, ys, horizon_start, n_loops)
                # list of [B, n], length K
                y_pred_arr = torch.cat(y_pred_list, dim=0)  # [B * K, n]
                y_star_arr = torch.cat([ys] * len(y_pred_list), dim=0)  # [B * K, n]
                loss = (y_star_arr - y_pred_arr).square().mean()  # auto on both K and n (number of in context samples)
                y_pred = y_pred_list[-1]  # [B, n]
        else:
            horizon_start = max(0, n_loops - args.training.n_loop_window)
            y_pred_list = model(xs, ys, horizon_start, n_loops)
            # list of [B, n], length K
            y_pred_arr = torch.cat(y_pred_list, dim=0)  # [B * K, n]
            y_star_arr = torch.cat([ys] * len(y_pred_list), dim=0)  # [B * K, n]
            loss = (y_star_arr - y_pred_arr).square().mean()  # auto on both K and n (number of in context samples)
            y_pred = y_pred_list[-1]  # [B, n]
    if args.training.use_ctx:
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
    else:
        loss.backward()
        optimizer.step()
    norm_dict, total_norm = calculate_gradient_norm(model)
    optimizer.zero_grad(set_to_none=True)
    return loss.detach(), y_pred.detach(), total_norm, norm_dict


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
    wandb.init(
        dir=args.out_dir,
        project=args.wandb.project,
        config=args.__dict__,
        notes=args.wandb.notes,
        name=args.wandb.name,
        mode="disabled" if args.debug_mode else "online",
        resume=True,
    )

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

    if args.training.use_fixed_dataset:
        from main_utils import gen_dataloader
        task_sampler = get_task_sampler(
            task_name=args.training.task_name,
            batch_size=args.training.batch_size,
            n_points=curriculum.n_points,
            n_dims=args.model.n_dims,
            n_dims_truncated=curriculum.n_dims_truncated,
            device=device,
            sparsity=args.training.sparsity,
        )
        train_loader = gen_dataloader(task_sampler, args.training.train_size,
                                      args.training.batch_size)
        train_iter = iter(train_loader)
        test_loader = gen_dataloader(task_sampler, args.training.test_size,
                                     args.training.batch_size)

    pbar = tqdm(range(starting_step, args.training.train_steps))
    for i in pbar:
        if args.training.use_fixed_dataset:
            try:
                batch = next(train_iter)
                xs, ys = batch['x'].to(device), batch['y'].to(device)
            except StopIteration:
                train_iter = iter(train_loader)
        else:
            task_sampler = get_task_sampler(
                task_name=args.training.task_name,
                batch_size=args.training.batch_size,
                n_points=curriculum.n_points,
                n_dims=args.model.n_dims,
                n_dims_truncated=curriculum.n_dims_truncated,
                device=device,
                sparsity=args.training.sparsity,
            )

            real_task = task_sampler()
            xs, ys = real_task.xs.float(), real_task.ys.float()

        loss, output, total_norm, grad_norm_dict = train_step(args, curriculum, model, xs, ys, optimizer, ctx, scaler)

        # EVALUATION ======================================
        point_wise_tags = list(range(curriculum.n_points))  # [0, 1, 2, ..., n-1]
        if i % args.wandb.log_every_steps == 0:
            point_wise_loss = (output - ys).square().mean(dim=0)  # [n,]
            if args.training.use_fixed_dataset:
                # eval
                with torch.no_grad():
                    for batch in test_loader:
                        xs, ys = batch['x'].to(device), batch['y'].to(device)
                        if args.model.family in ['gpt2']:
                            output = model(xs, ys)  # [B,]
                        elif args.model.family in ['gpt2_loop']:
                            n_loops = curriculum.n_loops  # K
                            y_pred_list = model(xs, ys, 0, n_loops)
                            output = y_pred_list[-1]  # [B, n]
                        else:
                            raise NotImplementedError
                        point_wise_loss = (output - ys).square().mean(dim=0)
                        loss = point_wise_loss.mean()
            wandb.log(
                {
                    "overall_loss": loss,
                    "loop_times": curriculum.n_loops,
                    "grad_norm/layerwise": grad_norm_dict,
                    "grad_norm": total_norm,
                    "pointwise/loss": dict(
                        zip(point_wise_tags, point_wise_loss.detach().cpu().numpy())
                    ),
                    "n_points": curriculum.n_points,
                    "n_dims": curriculum.n_dims_truncated,
                    "lr": optimizer.param_groups[0]['lr'],
                },
                step=i,
            )

        curriculum.update()

        pbar.set_description(f"loss {loss}")
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


if __name__ == "__main__":
    parser = QuinineArgumentParser(schema=schema)
    args = parser.parse_quinfig()
    print(f"Running with: {args}")

    device = init_device(args)

    if args.debug_mode:
        args.out_dir = "./results/debug"

    run_id = args.training.resume_id
    if run_id is None:
        run_id = get_run_id(args)

    out_dir = os.path.join(args.out_dir, run_id)
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    args.out_dir = out_dir
    # add a timestamp here, if resumed, this will be the resumed time
    args.wandb['timestamp'] = datetime.datetime.now().strftime("%m/%d/%Y, %H:%M:%S")

    with open(os.path.join(out_dir, "config.yaml"), "w") as yaml_file:
        yaml.dump(args.__dict__, yaml_file, default_flow_style=False)

    main(args, device)
