import argparse
import json
from pathlib import Path

import torch
import torch_fidelity
from accelerate import Accelerator
from tqdm import tqdm

from methods import METHODS
from networks import NETWORKS
from utils import FIDDataset
from vaes import VAES

accelerator = Accelerator()

_PREDEFINED_TARGETS = []
_PREDEFINED_CHECKPOINTS = []


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mother", type=Path, default=Path("./outputs"))
    parser.add_argument("--targets", nargs="*", default=_PREDEFINED_TARGETS)
    parser.add_argument("--checkpoints", nargs="*", default=_PREDEFINED_CHECKPOINTS)

    parser.add_argument("--sampling-steps", type=int, default=None)

    parser.add_argument("--precfg", type=float, default=None)
    parser.add_argument("--cfg-start", type=float, default=None)
    parser.add_argument("--cfg-end", type=float, default=None)

    parser.add_argument("--dump", type=Path, default=Path("fid50k.json"))
    parser.add_argument("--total-samples", type=int, default=50000)
    parser.add_argument("--batch-size", type=int, default=128)

    parser.add_argument("--fid-reference-file", default="./buffers/refs/in1k256_fid_ref.npz")  # fmt: skip
    return parser.parse_args()


def main():
    args = parse_args()
    targets = [args.mother / target for target in args.targets]
    assert len(targets) == len(set(targets))
    assert args.batch_size < args.total_samples

    # representative
    r, *_ = targets
    assert r.exists()

    c = json.loads((r / "config.json").read_text())
    image_size = c["data"]["image_size"]
    num_classes = c["data"]["num_classes"]
    latent_norm = c["data"]["latent_norm"]
    latent_multiplier = c["data"]["latent_multiplier"]
    method_config = dict(c["method"])
    model_type = c["model"]["type"]
    in_channels = c["model"]["in_chans"]
    vae_type = c["vae"]["type"]
    downsample_ratio = c["vae"]["downsample_ratio"]
    sampling_steps = args.sampling_steps or c["sample"]["sampling_steps"]

    accelerator.print(f"Total: {len(targets)}")

    cfg_range = None
    if args.cfg_start is not None and args.cfg_end is not None:
        cfg_range = [args.cfg_start, args.cfg_end]

    accelerator.print(f"Total: {len(targets)}")
    accelerator.print(f"[*DEBUG] PRECFG={args.precfg}, CFG_RANGE={cfg_range}")
    accelerator.print(f"[*DEBUG] Dump to {args.dump}")

    latent_size = image_size // downsample_ratio

    if args.precfg is not None:
        method_config["precfg"] = args.precfg
    if cfg_range is not None:
        method_config["cfgrange"] = cfg_range
    method_type = method_config.pop("type")
    method = METHODS[method_type](**method_config)
    model = NETWORKS[model_type](
        input_size=latent_size,
        num_classes=num_classes,
        in_channels=in_channels,
    )
    vae = VAES[vae_type](vae_type)

    torch.backends.cuda.matmul.allow_tf32 = True
    assert torch.cuda.is_available()
    torch.set_grad_enabled(False)

    device = accelerator.device
    n_proc = accelerator.num_processes
    accelerator.print(f"[*DEBUG] Total process: {n_proc}")

    n_samples_per_proc = -(-args.total_samples // n_proc)
    n_iters = -(-n_samples_per_proc // args.batch_size)
    offset = n_samples_per_proc * accelerator.process_index

    labels = torch.arange(num_classes)
    labels = labels.repeat(
        -(-(args.total_samples + args.batch_size + n_proc) // len(labels))
    )

    if latent_norm:
        mean_std = torch.load(
            f"./buffers/vaes/stat/{vae_type}_{image_size}.pt",
            weights_only=True,
        )
        latent_mean, latent_std = mean_std["mean"], mean_std["std"]
    else:
        latent_mean, latent_std = torch.tensor(0), torch.tensor(1)

    latent_mean = latent_mean.clone().detach().to(device)
    latent_std = latent_std.clone().detach().to(device)

    for mother in targets:
        mother = Path(mother)
        assert mother.exists()

        config = json.loads((mother / "config.json").read_text())
        assert config["data"]["image_size"] == image_size, mother
        assert config["data"]["num_classes"] == num_classes, mother
        assert config["data"]["latent_norm"] == latent_norm, mother
        assert config["data"]["latent_multiplier"] == latent_multiplier, mother
        assert config["model"]["type"] == model_type, mother
        assert config["model"]["in_chans"] == in_channels, mother
        assert config["vae"]["type"] == vae_type, mother
        assert config["vae"]["downsample_ratio"] == downsample_ratio, mother

        for ckpt in args.checkpoints:
            if not (mother / "checkpoints" / ckpt).exists():
                accelerator.print(
                    f"[*DEBUG] WARNING: {ckpt} not found for {mother.name}"
                )

    metrics = {}
    if args.dump.exists():
        metrics.update(json.loads(args.dump.read_text()))

    for mother in (pbar := tqdm(targets, disable=not accelerator.is_main_process)):
        mother = Path(mother)
        config = json.loads((mother / "config.json").read_text())

        pbar.set_postfix_str(mother.name)
        if mother.name not in metrics:
            metrics[mother.name] = {}

        for ckpt in tqdm(
            args.checkpoints, leave=False, disable=not accelerator.is_main_process
        ):
            if not (mother / "checkpoints" / ckpt).exists():
                continue

            metric_key = ckpt
            if args.precfg is not None or cfg_range is not None:
                metric_key = f"{ckpt}:{args.precfg}:{cfg_range}"
            if metric_key in metrics[mother.name]:
                continue

            checkpoint = torch.load(
                mother / "checkpoints" / ckpt,
                map_location=lambda storage, loc: storage,
                weights_only=True,
            )
            if "ema" in checkpoint:
                checkpoint = checkpoint["ema"]
            model.load_state_dict(checkpoint)
            model.eval()
            model.to(device)

            wrapped = accelerator.prepare(model)

            all_samples = []
            torch.manual_seed(accelerator.process_index)
            for i in tqdm(
                range(n_iters), leave=False, disable=not accelerator.is_main_process
            ):
                _offset = offset + i * args.batch_size
                z = torch.randn(
                    args.batch_size,
                    in_channels,
                    latent_size,
                    latent_size,
                    device=device,
                )
                y = labels[_offset : _offset + args.batch_size].to(device)

                x_cur = method.sample(z, wrapped, y, sampling_steps)[-1]
                samples = (x_cur * latent_std) / latent_multiplier + latent_mean
                samples = vae.decode_to_images(samples)
                samples = accelerator.gather(samples).cpu()
                if accelerator.is_main_process:
                    all_samples.append(samples)

            if accelerator.is_main_process:
                all_samples = torch.cat(all_samples)[: args.total_samples]
                metrics_dict = torch_fidelity.calculate_metrics(
                    input1=FIDDataset(all_samples),
                    input2=None,
                    fid_statistics_file=args.fid_reference_file,
                    cuda=True,
                    isc=True,
                    fid=True,
                    kid=False,
                    prc=False,
                    verbose=True,
                )
                metrics[mother.name][metric_key] = metrics_dict
                with open(args.dump, "w") as f:
                    json.dump(metrics, f, indent=2)

            accelerator.wait_for_everyone()


if __name__ == "__main__":
    main()


"""
CUDA_VISIBLE_DEVICES=2 accelerate launch \
    --dynamo_backend=no \
    --num_processes=1 \
    --num_machines=1 \
    --mixed_precision=bf16 \
    eval.py
"""
