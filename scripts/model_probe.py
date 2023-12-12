import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import wandb
import datetime
import uuid
import argparse

from nano_gpt import GPT2Model, GPT2Config


def get_targets(xs, ys, target_mode):
    targets = []
    n_points = xs.shape[1]
    n_dims = xs.shape[2]
    if target_mode == 'grad':
        for i_n in range(1, n_points + 1):
            target = (ys[:, None, :i_n] @ xs[:, :i_n, :])[:, 0, :] / n_points  # [B, 1, d] -> [B, d]
            targets.append(target)
    elif target_mode == 'Wols':
        device = xs.device
        xs, ys = xs.cpu(), ys.cpu()
        for i_n in range(1, n_dims + 1):
            train_xs, train_ys = xs[:, :i_n], ys[:, :i_n]
            target, _, _, _ = torch.linalg.lstsq(train_xs, train_ys.unsqueeze(2))  # [B, d, 1]
            target = target[:, :, 0]  # -> [B, d]
            targets.append(target.to(device))
    else:
        raise NotImplementedError

    return targets


def get_run_name(lr, loop_mode, target_mode, control_mode, wandb_name):
    now = datetime.datetime.now().strftime('%m%d%H%M%S')
    run_uuid = str(uuid.uuid4())[:4]
    dir_path = f'./results2/model_probe/{now}_{loop_mode}_lr={lr}_target={target_mode}_control={control_mode}_{run_uuid}_{wandb_name}'
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
    state_path = dir_path + '/state_dict.pt'
    return state_path, dir_path


def trainer(args, model, p_model, optim, state_path, n_layer, sample_size, n_points, n_dims, n_dims_truncated, device, n_loops=0):
    pbar = tqdm(range(args.n_epochs))
    for i in pbar:
        real_task = LinearRegression(sample_size, n_points, n_dims, n_dims_truncated, device)
        xs, ys, w_b = real_task.xs, real_task.ys, real_task.w_b
        if args.control_exp:
            w_b = torch.ones_like(w_b)
            ys = (xs @ w_b).sum(-1)  # [B, n]
        # xs shape: B, n, d
        # ys shape: B, n
        with torch.no_grad():
            if n_loops > 0:
                _, embeds = model(xs, ys, 0, n_loops)
            else:
                _, embeds = model(xs, ys)
        targets = get_targets(xs, ys, target_mode=args.target_mode)
        p_loss = p_model(embeds, targets)
        p_loss.mean().backward()
        optim.step()
        optim.zero_grad(set_to_none=True)

        if args.control_exp:
            real_task = LinearRegression(sample_size, n_points, n_dims, n_dims_truncated, device)
            xs, ys, w_b = real_task.xs, real_task.ys, real_task.w_b
            with torch.no_grad():
                if n_loops > 0:
                    _, embeds = model(xs, ys, 0, n_loops)
                else:
                    _, embeds = model(xs, ys)
                targets = get_targets(xs, ys, target_mode=args.target_mode)
                p_loss = p_model(embeds, targets)
        # test
        if i % 100:
            point_wise_tags = list(range(n_layer + 1)) if n_loops == 0 else list(range(n_loops + 1))
            wandb.log(
                {
                    "overall_loss": p_loss.mean().detach().item(),
                    "pointwise/loss": dict(
                        zip(point_wise_tags, p_loss.mean(0).detach().data)
                    ),
                },
                step=i,
            )
        if i % 1000 == 0:
            training_state = {
                "model_state_dict": p_model.state_dict(),
                "p_loss": p_loss.mean(0).detach().data,
                "train_step": i,
            }
            torch.save(training_state, state_path)
    wandb.finish()


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--n-epochs', type=int, default=10000)  # 100000
    parser.add_argument('--n-gpus', type=int, default=1)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--target-mode', type=str, default='grad')
    parser.add_argument('--control-exp', type=bool, default=False)
    parser.add_argument('--wandb-name', type=str, default=None)
    args = parser.parse_args()

    sample_size = 1280
    n_points = 41
    n_dims_truncated = 20
    n_dims = 20
    device = torch.device('cuda:{}'.format(args.n_gpus))

    if args.target_mode == 'grad':
        n_targets = n_points
    elif args.target_mode == 'Wols':
        n_targets = n_dims
    else:
        raise NotImplementedError

    # for unlooped
    state_path, dir_path = get_run_name(args.lr, 'unloop', args.target_mode, args.control_exp, args.wandb_name)
    wandb.init(
        dir=dir_path,
        project='loop_probe',
        name='{}_unloop_lr={}_target={}_control={}'.format(args.wandb_name, args.lr, args.target_mode, args.control_exp),
    )

    result_dir = './results2/linear_regression_baseline'
    run_id = '0831113051-LR_baseline_L20-d195'
    n_positions = 101
    n_embd = 256
    n_layer = 20
    n_head = 8

    model = TransformerModel(n_dims, n_positions, n_embd, n_layer, n_head)
    step = -1
    model = get_model(model, result_dir, run_id, step)
    model = model.to(device)

    p_model = ProbeModel(
        n_layer=n_layer + 1,
        d_target=n_dims,
        D_embed=n_embd,
        n_seq=n_points * 2,
        n_targets=n_targets
    ).to(device)

    optim = torch.optim.Adam(p_model.parameters(), lr=args.lr, weight_decay=0)
    optim.zero_grad()
    trainer(args, model, p_model, optim, state_path, n_layer, sample_size, n_points, n_dims, n_dims_truncated, device)


    # for looped
    state_path, dir_path = get_run_name(args.lr, 'loop', args.target_mode, args.control_exp, args.wandb_name)
    wandb.init(
        dir=dir_path,
        project='loop_probe',
        name='{}_loop_lr={}_target={}_control={}'.format(args.wandb_name, args.lr, args.target_mode, args.control_exp),
        resume=True,
    )

    result_dir = './results2/linear_regression_loop'
    run_id = '0706234720-LR_loop_L1_ends{20}_T{15}_all-cbc4'
    n_positions = 101
    n_embd = 256
    n_head = 8
    n_layer = 1

    l_model = TransformerModelLooped(n_dims, n_positions, n_embd, n_layer, n_head)
    step = -1
    l_model = get_model(l_model, result_dir, run_id, step)
    l_model = l_model.to(device)

    n_loop = 20

    l_p_model = ProbeModel(
        n_layer=n_loop + 1,
        d_target=n_dims,
        D_embed=n_embd,
        n_seq=n_points * 2,
        n_targets=n_targets
    ).to(device)

    optim = torch.optim.Adam(
        l_p_model.parameters(), lr=args.lr, weight_decay=0)
    optim.zero_grad()

    trainer(args, l_model, l_p_model, optim, state_path, n_layer, sample_size, n_points, n_dims, n_dims_truncated, device, n_loops=n_loop)


def get_model(model, result_dir, run_id, step, best=False):
    if best:
        model_path = os.path.join(result_dir, run_id, 'model_best.pt')
        state_dict = torch.load(model_path, map_location='cpu')['state_dict']
        best_err = torch.load(model_path, map_location='cpu')['loss']
        print("saved model with loss:", best_err)
    if step == -1:
        model_path = os.path.join(result_dir, run_id, 'state.pt')
        state_dict = torch.load(model_path, map_location='cpu')['model_state_dict']
    else:
        model_path = os.path.join(result_dir, run_id, 'model_{}.pt'.format(step))
        state_dict = torch.load(model_path, map_location='cpu')['model']

    # return state_dict
    unwanted_prefix = '_orig_mod.'
    for k, v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
    model.load_state_dict(state_dict, strict=True)

    return model


class LinearRegression():
    def __init__(self, batch_size, n_points, n_dims, n_dims_truncated, device, w_star=None):
        super(LinearRegression, self).__init__()
        self.device = device
        self.xs = torch.randn(batch_size, n_points, n_dims).to(device)
        self.xs[..., n_dims_truncated:] = 0
        w_b = torch.randn(batch_size, n_dims, 1) if w_star is None else w_star.to(device)  # [B, d, 1]
        w_b[:, n_dims_truncated:] = 0
        self.w_b = w_b.to(device)
        self.ys = (self.xs @ self.w_b).sum(-1)  # [B, n]


# Define probing model
class ProbeModel(nn.Module):
    """Adds (optionally learned) positional embeddings to the inputs."""

    def __init__(self, n_layer, n_targets, d_target, D_embed, n_seq):
        super(ProbeModel, self).__init__()
        self.layer_probes = nn.ModuleList([])
        d_hidden = 64
        for _ in range(n_layer * n_targets):
            self.layer_probes.append(
                nn.Sequential(
                    nn.Linear(D_embed, d_hidden),
                    # nn.GELU(),
                    nn.ReLU(),
                    nn.Linear(d_hidden, d_target)
            ))
        self.layer_alphas = nn.ParameterList([
            nn.Parameter(torch.randn(n_seq)) for _ in range(n_layer * n_targets)])
        self.n_layer = n_layer

    def forward(self, seq_hiddens, targets):
        """
        :param seq_hiddens: list of hidden embeddings, each of shape [B, n, D]
        :param target: the target coefficients to decode, of shape [B, d]
        :return:
        """
        B = seq_hiddens[0].shape[0]
        n_layer = self.n_layer
        probe_erros_total = torch.zeros(B, n_layer, device=seq_hiddens[0].device)
        counter = 0
        for target in targets:
            probe_errors = []
            for i in range(self.n_layer):
                layer_probe = self.layer_probes[counter]
                layer_alpha = self.layer_alphas[counter]  # [n_seq,]
                counter += 1
                agg_hiddens = (seq_hiddens[i] * F.softmax(layer_alpha, dim=0)[None, :, None]).sum(1)
                # [B, n, D] * [1, n, 1] -> [B, n, D] -> [B, D]
                probe_pred = layer_probe(agg_hiddens)  # [B, D] -> [B, d]
                probe_err = (probe_pred - target) ** 2
                probe_errors.append(probe_err.mean(dim=1))  # [B]
            # probe_errors should be a list of n_layer tensors, each of shape [B]
            probe_erros_total += torch.stack(probe_errors, dim=1)  # [B, n_layer]
        return probe_erros_total / len(targets)


# Define the loop and unloop models
class TransformerModel(nn.Module):
    def __init__(self, n_dims, n_positions, n_embd=128, n_layer=12, n_head=4):

        super(TransformerModel, self).__init__()
        self.freq = 2
        self.ind = 0
        configuration = GPT2Config()
        configuration.block_size = self.freq * n_positions + 1
        configuration.n_layer = n_layer
        configuration.n_head = n_head
        configuration.n_embd = n_embd
        configuration.dropout = 0.0
        configuration.bias = True
        configuration.dropout = 0.

        self.n_positions = n_positions  # n = points in this setting
        self.n_dims = n_dims  # input dimension, d_in
        self.n_embd = n_embd  # d

        self._read_in = nn.Linear(n_dims, n_embd)
        self._backbone = GPT2Model(configuration)
        self._read_out = nn.Linear(n_embd, 1)

        self.print_flag = False

    def _combine(self, xs_b, ys_b):
        """
        :param xs_b: shape [B, n, d_in]
        :param ys_b: shape [B, n]
        :return: shape [B, 2n, d_in + 1]
        """
        B, n, d = xs_b.shape
        device = xs_b.device

        ys_b_wide = torch.cat(
            (
                ys_b.view(B, n, 1),
                torch.zeros(B, n, d-1, device=device),
            ),
            axis=2,
        )

        zs = torch.stack((xs_b, ys_b_wide), dim=2)
        zs = zs.view(B, self.freq * n, d)

        return zs

    def forward(self, xs, ys):
        """
        :param xs: [B, n, d]
        :param ys: [B, n]
        :return:
        """

        B, n, d_in = xs.shape
        zs = self._combine(xs, ys)  # [B, n, d_in], [B, n], [B, n] -> [B, 2n, d_in + 1]
        embeds = self._read_in(zs)  # [B, 2n, d_in + 1] -> [B, 2n, d]

        f_output, embeds_list = self._backbone(
            inputs_embeds=embeds, position_ids=None, rm_pos_embd=False, output_intermediate=True)  # [B, 2n, d]
        prediction = self._read_out(f_output)  # [B, 2n, d] -> [B, 2n, 1]
        w = prediction[:, self.ind::self.freq, 0]

        return w, embeds_list  # list of [B, n, d]


class TransformerModelLooped(TransformerModel):
    def __init__(self, n_dims, n_positions, n_embd=128, n_layer=12, n_head=4, loop_func='z=f(x+z)'):
        super(TransformerModelLooped, self).__init__(
            n_dims, n_positions, n_embd, n_layer, n_head)
        self.loop_func = loop_func

    def f(self, output, embeds):
        f_output, embeds_list = self._backbone(inputs_embeds=output + embeds,
                                               output_intermediate=True)  # [B, 2n + 1, d]
        return f_output, embeds_list

    def forward(self, xs, ys, n_loop_start, n_loops):
        """
        :param xs: [B, n, d]
        :param ys: [B, n]
        :param n_loop_start: int
        :param n_loops: int
        :return:
        """
        B, n, d_in = xs.shape
        zs = self._combine(xs, ys)  # [B, n, d_in], [B, n], [B, n] -> [B, 2n, d_in + 1]
        embeds = self._read_in(zs)  # [B, 2n, d_in + 1] -> [B, 2n, d]
        output = torch.zeros_like(embeds)  # also of shape [B, 2n, d]

        embeds_list_total = [output + embeds]
        pred_list = []
        for idx in range(n_loops):  # this will save memory when n_loops large.
            output, embeds_list = self.f(output, embeds)
            embeds_list_total.append(embeds_list[-1])
            prediction = self._read_out(output)  # [B, 2n, d] -> [B, 2n, 1]
            y = prediction[:, self.ind::self.freq, 0]  # [B, n]
            pred_list.append(y)

        return pred_list, embeds_list_total  # list of [B, n, n]


main()