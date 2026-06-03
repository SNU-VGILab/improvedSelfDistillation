import torch
import torch.nn as nn


class ImprovedSelfDistillation:
    def __init__(
        self,
        transport: str = "Linear",
        label_dropout: float = 0.1,
        p: float = 1.0,
        use_jvp: bool = True,
        precfg: float | None = None,
    ):
        self.transport = transport
        self.label_dropout = label_dropout
        self.p = p
        self.jvp = use_jvp
        self.precfg = precfg

    def _destruct_module(self, model):
        while hasattr(model, "module"):
            model = model.module
        return model

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

        if self.transport == "Linear":
            x_t = (1 - t) * x + t * z
            v_t = z - x
        elif self.transport == "TrigFlow":
            pt = t * 1.57
            x_t = pt.cos() * x + pt.sin() * z
            v_t = pt.cos() * z - pt.sin() * x

        F_th = model(x_t, t, t, **dict(y=c))
        with torch.no_grad():
            uncond = model(x_t, t, t, y=torch.full_like(c, nullc))
            v_t = uncond + self.precfg * (v_t - uncond)

        flow_matching_loss = (F_th - v_t).square().mean(dim=(1, 2, 3))

        v_t = F_th.detach()
        if self.jvp:
            jvp_fn = torch.compiler.disable(torch.func.jvp, recursive=False)
            F_th, dF_dt = jvp_fn(
                lambda x_t, t, s: model(x_t, t, s, y=c),
                (x_t, t, s),
                (v_t, torch.ones_like(t), torch.zeros_like(s)),
            )
            dF_dt = dF_dt.detach()
        else:
            F_th = model(x_t, t, s, y=c)
            with torch.no_grad():
                eps = 0.005
                dt = 1 / (2 * eps)
                dF_dt = (
                    model(x_t + eps * v_t, t + eps, s, y=c) * dt
                    - model(x_t - eps * v_t, t - eps, s, y=c) * dt
                )

        with torch.no_grad():
            if self.transport == "Linear":
                target = v_t - (t - s) * dF_dt
            elif self.transport == "TrigFlow":
                d = (t - s) * 1.57
                target = F_th + d.cos() * (v_t - F_th) - d.sin() * (x_t + dF_dt)
            else:
                assert False
        flow_map_distill_loss = (F_th - target).square().mean(dim=(1, 2, 3))

        loss = flow_matching_loss + flow_map_distill_loss
        weight = (loss + 0.01).detach() ** self.p
        return (loss / weight).mean()

    @torch.no_grad()
    def sample(
        self,
        z: torch.Tensor,
        model: nn.Module,
        y: torch.Tensor,
        sampling_steps: int = 35,
    ):
        device = z.device
        t_steps = torch.linspace(1.0, 0.0, sampling_steps + 1, device=device)

        x_cur = z
        samples = [x_cur.clone()]
        for t_cur, t_next in zip(t_steps, t_steps[1:]):
            F_th = model(x_cur, t_cur, t_next, y)
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
