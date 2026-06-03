import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from methods.isd import ImprovedSelfDistillation


class AugmentediSD(ImprovedSelfDistillation):
    def __init__(
        self,
        transport: str = "Linear",
        label_dropout: float = 0.1,
        use_jvp: bool = True,
        precfg: float | None = None,
        cfgrange: tuple[float, float] | None = None,
    ):
        self.transport = transport
        self.label_dropout = label_dropout
        self.jvp = use_jvp
        self.precfg = precfg
        self.cfgrange = cfgrange

    def training_step(self, model: nn.Module, x: torch.Tensor, c: torch.Tensor):
        bsize, _, _, _ = x.shape
        dist = torch.distributions.Beta(0.8, 1.0)
        t = dist.sample((bsize, 1, 1, 1)).to(x).clamp(0.0, 1.0)
        s = dist.sample((bsize, 1, 1, 1)).to(x).clamp(0.0, 1.0)
        t, s = torch.maximum(t, s), torch.minimum(t, s)

        z = torch.randn_like(x)
        nullc = self._destruct_module(model).num_classes
        ndrop = round(self.label_dropout * len(c))
        c[-ndrop:] = nullc

        ft = t.flatten()
        omega = (torch.rand_like(ft) * np.log(8.0)).exp()
        t_min = torch.rand_like(omega) * 0.5
        t_max = torch.rand_like(omega) * 0.5 + 0.5
        omega[torch.logical_or(ft < t_min, ft >= t_max)] = 1.0
        icl = dict(omega=omega, t_min=t_min, t_max=t_max, y=c)

        if self.transport == "Linear":
            x_t = (1 - t) * x + t * z
            v_t = z - x
        elif self.transport == "TrigFlow":
            pt = t * 1.57
            x_t = pt.cos() * x + pt.sin() * z
            v_t = pt.cos() * z - pt.sin() * x

        F_th = model(x_t, t, t, **icl)
        with torch.no_grad():
            uncond = model(x_t, t, t, omega, t_min, t_max, y=torch.full_like(c, nullc))
            v_t = uncond + omega[:, None, None, None] * (v_t - uncond)

        mse = (F_th - v_t).square().mean(dim=(1, 2, 3))
        cossim = (1.0 - F.cosine_similarity(F_th, v_t, dim=1)).mean(dim=(1, 2))
        flow_matching_loss = mse + cossim

        v_t = F_th.detach()
        if self.jvp:
            jvp_fn = torch.compiler.disable(torch.func.jvp, recursive=False)
            F_th_t_s, dF_dt = jvp_fn(
                lambda x_t, t, s: model(x_t, t, s, **icl),
                (x_t, t, s),
                (v_t, torch.ones_like(t), torch.zeros_like(s)),
            )
            dF_dt = dF_dt.detach()
        else:
            F_th_t_s = model(x_t, t, s, **icl)
            with torch.no_grad():
                eps = 0.005
                dt = 1 / (2 * eps)
                dF_dt = (
                    model(x_t + eps * v_t, t + eps, s, **icl) * dt
                    - model(x_t - eps * v_t, t - eps, s, **icl) * dt
                )

        with torch.no_grad():
            if self.transport == "Linear":
                target = v_t - (t - s) * dF_dt
            elif self.transport == "TrigFlow":
                d = (t - s) * 1.57
                target = F_th_t_s + d.cos() * (v_t - F_th_t_s) - d.sin() * (x_t + dF_dt)
            else:
                assert False
        flow_map_distill_loss = (F_th_t_s - target).square().mean(dim=(1, 2, 3))
        flow_map_distill_loss = (flow_map_distill_loss + 25).sqrt() - 5

        loss = (t.flatten() * 1.57).cos() * (flow_matching_loss + flow_map_distill_loss)
        return loss.mean()

    @torch.no_grad()
    def sample(
        self,
        z: torch.Tensor,
        model: nn.Module,
        y: torch.Tensor,
        sampling_steps: int = 35,
    ):
        device = z.device
        precfg = torch.tensor([self.precfg], device=device)
        cfg_start, cfg_end = self.cfgrange
        cfg_start = torch.tensor([cfg_start], device=device)
        cfg_end = torch.tensor([cfg_end], device=device)
        t_steps = torch.linspace(1.0, 0.0, sampling_steps + 1, device=device)

        x_cur = z
        samples = [x_cur.clone()]
        for t_cur, t_next in zip(t_steps, t_steps[1:]):
            omega = precfg.masked_fill(
                torch.logical_or(t_cur < cfg_start, t_cur >= cfg_end), 1.0
            )
            F_th = model(x_cur, t_cur, t_next, omega, cfg_start, cfg_end, y)
            if self.transport == "Linear":
                x_cur = x_cur + (t_next - t_cur) * F_th
            elif self.transport == "TrigFlow":
                x_cur = (
                    torch.cos((t_next - t_cur) * 1.57) * x_cur
                    + torch.sin((t_next - t_cur) * 1.57) * F_th
                )
            else:
                assert False
            samples.append(x_cur)
        return torch.stack(samples, dim=0)
