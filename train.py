import warnings

warnings.filterwarnings("ignore")

import argparse
import json
import gc
import os
import yaml
from copy import deepcopy
from datetime import datetime, timezone, timedelta
from pathlib import Path
from time import time
from tqdm import tqdm

import wandb
import torch
import torch.distributed as dist
from accelerate import Accelerator
from accelerate.utils import DistributedDataParallelKwargs, InitProcessGroupKwargs
from torch.utils.data import DataLoader
from torchvision.utils import save_image

from methods import METHODS
from networks import NETWORKS
from optimizers import AdamW, NorMuon
from utils import (
    ImgLatentDataset,
    create_logger,
    remove_module_prefix,
    set_seed,
    torch_fidelity_metrics,
    update_ema,
)
from vaes import VAES


def train(config, accelerator):
    device = accelerator.device
    rank = accelerator.process_index
    seed = config["train"]["global_seed"] * accelerator.num_processes + rank
    set_seed(seed)

    if accelerator.is_main_process:
        stamp = (
            config.get("stamp", None)
            or datetime.now(timezone(timedelta(hours=9))).strftime("%Y.%m.%dKST%H.%M.%S")  # fmt: skip
        )
        if _name := config.get("name"):
            stamp = f"{stamp}-{_name}"
        experiment_dir = f"{config['output_dir']}/{stamp}"
        os.makedirs(experiment_dir, exist_ok=True)
        with open(f"{experiment_dir}/config.json", "w") as f:
            json.dump(config, f, indent=2, ensure_ascii=False)
        run = wandb.init(
            project="iSD",
            id=stamp,
            name=stamp,
            config=config,
            dir=experiment_dir,
            resume="allow",
        )

        checkpoint_dir = f"{experiment_dir}/checkpoints"

        os.makedirs(config["output_dir"], exist_ok=True)
        os.makedirs(checkpoint_dir, exist_ok=True)
        logger = create_logger(experiment_dir, "train")
        logger.info(f"Experiment directory created at {experiment_dir}")
        logger.info(f"Gradient accumulation: {accelerator.gradient_accumulation_steps}")

    downsample_ratio = config["vae"]["downsample_ratio"]
    assert config["data"]["image_size"] % downsample_ratio == 0

    latent_size = config["data"]["image_size"] // downsample_ratio
    model = NETWORKS[config["model"]["type"]](
        input_size=latent_size,
        num_classes=config["data"]["num_classes"],
        in_channels=config["model"]["in_chans"],
    ).to(device)
    ema = deepcopy(model).requires_grad_(False).to(device)

    vae = VAES[config["vae"]["type"]](config["vae"]["type"])
    if accelerator.is_main_process:
        logger.info("Loaded VAE model")

        demoimages_dir = f"{experiment_dir}/demoimages"
        os.makedirs(demoimages_dir, exist_ok=True)

        demo_y = torch.tensor(config["sample"]["demos"], device=device)
        demo_z = torch.randn(
            len(demo_y), model.in_channels, latent_size, latent_size, device=device
        )

    method_config = dict(config["method"])
    method_type = method_config.pop("type")
    method = METHODS[method_type](**method_config)
    if accelerator.is_main_process:
        logger.info(
            f"{config['model']['type']} Parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M"
        )
        logger.info(
            f"Optimizer: {config['optimizer']['type']}, lr={config['optimizer']['lr']}, beta1={config['optimizer']['beta1']}, beta2={config['optimizer']['beta2']}"
        )

    muon, muon_keys = None, {}
    if "muon" in config:
        muon_keys = {
            n: p
            for n, p in model.named_parameters()
            if p.dim() == 2
            and all(k not in n for k in ["embedding_table", "embeddings", "logvar"])
        }
        assert all(isinstance(getattr(p, "label", None), str) for p in muon_keys.values())  # fmt: skip

        assert config["muon"]["type"] == "NorMuon"
        muon = NorMuon(
            muon_keys.values(),
            lr=config["muon"]["lr"],
            weight_decay=config["muon"]["weight_decay"],
            momentum=config["muon"]["momentum"],
            beta2=config["muon"]["beta2"],
            use_polar_express=config["muon"]["use_polar_express"],
        )

    assert config["optimizer"]["type"] == "AdamW"
    opt = AdamW(
        (p for n, p in model.named_parameters() if n not in muon_keys),
        lr=config["optimizer"]["lr"],
        weight_decay=config["optimizer"]["weight_decay"],
        betas=(config["optimizer"]["beta1"], config["optimizer"]["beta2"]),
    )

    dataset = ImgLatentDataset(
        data_dir=config["data"]["data_path"],
        latent_norm=config["data"]["latent_norm"],
        latent_multiplier=config["data"]["latent_multiplier"],
    )
    batch_size_per_gpu = (
        config["train"]["global_batch_size"]
        // accelerator.num_processes
        // accelerator.gradient_accumulation_steps
    )

    global_batch_size = (
        batch_size_per_gpu
        * accelerator.num_processes
        * accelerator.gradient_accumulation_steps
    )
    loader = DataLoader(
        dataset,
        batch_size=batch_size_per_gpu,
        shuffle=True,
        num_workers=config["data"]["num_workers"],
        pin_memory=True,
        drop_last=True,
    )

    latent_mean, latent_std, latent_multiplier = (
        dataset._latent_mean.cuda(),
        dataset._latent_std.cuda(),
        dataset.latent_multiplier,
    )

    if accelerator.is_main_process:
        logger.info(
            f"Dataset contains {len(dataset):,} images {config['data']['data_path']}"
        )
        logger.info(
            f"Batch size {batch_size_per_gpu} per gpu, with {global_batch_size} global batch size"
        )

    train_steps = 0
    if "ckpt" in config["train"]:
        checkpoint_path = config["train"]["ckpt"]
        checkpoint = torch.load(
            checkpoint_path, map_location=lambda storage, loc: storage
        )
        model.load_state_dict(checkpoint["model"])
        ema.load_state_dict(checkpoint["ema"])
        if config["train"]["load_opt"]:
            opt.load_state_dict(checkpoint["opt"])
            # muon.load_state_dict(checkpoint["muon"])  # TODO: fix muon checkpoints
        if not config["train"].get("reinit_train_steps"):
            train_steps = checkpoint["steps"]
        del checkpoint
        if accelerator.is_main_process:
            logger.info(f"Loaded checkpoint at: {checkpoint_path}.")
    else:
        if accelerator.is_main_process:
            logger.info("Starting training from scratch.")

    top1, top1_fid = None, None
    if accelerator.is_main_process:
        if existing := list(Path(checkpoint_dir).glob("top.*.pt")):
            (top1,) = existing
            top1_fid = float(top1.name[4:-3])
            steps = torch.load(top1, map_location="cpu")["steps"]
            logger.info(f"Previous top-1 ({steps}-steps): FID={top1_fid}")
        else:
            logger.info(f"Cannot found previous top-1")

    model.train()
    ema.eval()
    if muon is not None:
        model, opt, muon, loader = accelerator.prepare(model, opt, muon, loader)
    else:
        model, opt, loader = accelerator.prepare(model, opt, loader)

    sampling_steps = config["sample"]["sampling_steps"]

    # training states
    log_steps = 0
    accum_steps = 0
    running_loss = 0
    start_time = time()

    while True:
        for x, y in loader:
            with accelerator.accumulate(model):
                if accelerator.mixed_precision == "no":
                    x = x.to(device, dtype=torch.float32)
                else:
                    x = x.to(device)
                y = y.to(device)

                loss = method.training_step(model, x, y)

                opt.zero_grad()
                if muon is not None:
                    muon.zero_grad()
                accelerator.backward(loss)
                if "max_grad_norm" in config["optimizer"]:
                    if accelerator.sync_gradients:
                        accelerator.clip_grad_norm_(
                            model.parameters(), config["optimizer"]["max_grad_norm"]
                        )
                for param in model.parameters():
                    if param.grad is not None:
                        torch.nan_to_num_(param.grad, nan=0.0, posinf=0.0, neginf=0.0)
                opt.step()
                if muon is not None:
                    muon.step()
                update_ema(ema, model, config["train"]["ema_decay"])

            if accelerator.is_main_process:
                run.log({"train/loss/every": loss.item()}, step=train_steps)

            running_loss += loss.item()
            log_steps += 1
            if accum_steps == 0 and train_steps % config["train"]["log_every"] == 0:
                torch.cuda.synchronize()
                end_time = time()
                steps_per_sec = log_steps / (end_time - start_time)
                avg_loss = torch.tensor(running_loss / log_steps, device=device)
                if accelerator.num_processes > 1:
                    dist.all_reduce(avg_loss, op=dist.ReduceOp.SUM)
                    avg_loss = avg_loss.item() / dist.get_world_size()
                if accelerator.is_main_process:
                    logger.info(
                        f"(step={train_steps:07d}) Train Loss: {avg_loss:.4f}, Train Steps/Sec: {steps_per_sec:.2f}"
                    )
                    run.log(
                        {"train/loss/avg": avg_loss, "train/steps/sec": steps_per_sec},
                        step=train_steps,
                    )

                running_loss = 0
                log_steps = 0
                start_time = time()

            if accum_steps == 0 and train_steps % config["train"]["ckpt_every"] == 0:
                if accelerator.is_main_process:
                    checkpoint = {
                        "model": remove_module_prefix(model.state_dict()),
                        "ema": remove_module_prefix(ema.state_dict()),
                        "opt": opt.state_dict(),
                        "config": config,
                        "steps": train_steps,
                    }
                    if muon is not None:
                        checkpoint["muon"] = muon.state_dict()

                    checkpoint_path = f"{checkpoint_dir}/latest.pt"
                    torch.save(checkpoint, checkpoint_path)
                    if (
                        "ckpt_legacy" not in config["train"]
                        or train_steps % config["train"]["ckpt_legacy"] == 0
                    ):
                        checkpoint_path = f"{checkpoint_dir}/{train_steps:07d}.pt"
                        torch.save(checkpoint, checkpoint_path)
                    logger.info(f"Saved checkpoint to {checkpoint_path}")
                    sampled = method.sample(
                        demo_z,
                        ema,
                        demo_y,
                        sampling_steps,
                    )
                    # if sampling step is too large
                    timesteps, batch_size, *_ = sampled.shape
                    if timesteps > 10:
                        _indices = list(range(timesteps))
                        _indices = _indices[-1 :: (-timesteps // 10)]
                        timesteps = len(_indices)
                        sampled = sampled[_indices[::-1]]
                    # vae decode
                    demox = sampled.reshape(-1, *sampled.shape[2:])
                    demox = (demox * latent_std) / latent_multiplier + latent_mean
                    demox = vae.decode_to_images(demox).cpu()
                    demoimages_path = f"{demoimages_dir}/{train_steps:07d}.png"
                    save_image(demox, os.path.join(demoimages_path), nrow=len(demo_y))
                    # write to wandb
                    ndarr = demox.mul(255).add_(0.5).clamp_(0, 255)
                    ndarr = ndarr.to("cpu", torch.uint8).numpy()
                    _, channels, height, width = ndarr.shape
                    ndarr = ndarr.reshape(timesteps, batch_size, channels, height, width)  # fmt: skip
                    ndarr = ndarr.transpose(0, 1, 3, 4, 2)
                    run.log(
                        {
                            f"test/sample/{i}/{t}": wandb.Image(img, "RGB")
                            for t, batch in enumerate(ndarr)
                            for i, img in enumerate(batch)
                        },
                        step=train_steps,
                    )

                    logger.info(f"Saved demoimages to {demoimages_path}")

                    del demox, sampled
                    gc.collect()
                    torch.cuda.empty_cache()

                if accelerator.num_processes > 1:
                    accelerator.wait_for_everyone()

                with torch.no_grad():
                    n = config["sample"]["per_batch_size"]
                    total_samples = config["sample"]["fid_sample_num"]

                    n_proc = accelerator.num_processes
                    n_samples_per_proc = -(-total_samples // n_proc)
                    n_iters = -(-n_samples_per_proc // n)
                    offset = n_samples_per_proc * accelerator.process_index

                    labels = torch.arange(config["data"]["num_classes"], device=device)
                    labels = labels.repeat(
                        -(-(total_samples + n + n_proc) // len(labels))
                    )
                    # collaborative sampling
                    all_samples = []
                    for i in tqdm(
                        range(n_iters),
                        desc="Sampling for FID",
                        disable=not accelerator.is_main_process,
                    ):
                        _offset = offset + i * n
                        y = labels[_offset : _offset + n]
                        z = torch.randn(
                            len(y),
                            ema.in_channels,
                            latent_size,
                            latent_size,
                            device=device,
                        )

                        samples = method.sample(z, ema, y, sampling_steps)[-1]
                        samples = (samples * latent_std) / latent_multiplier + latent_mean  # fmt: skip
                        samples = vae.decode_to_images(samples)
                        samples = accelerator.gather(samples).cpu()
                        if accelerator.is_main_process:
                            all_samples.append(samples)

                        del samples, z, y
                        gc.collect()
                        torch.cuda.empty_cache()
                    # measure metrics
                    if accelerator.is_main_process:
                        all_samples = torch.cat(all_samples)[:total_samples]
                        if fid_ref := config["data"]["fid_reference_file"]:
                            metrics_dict = torch_fidelity_metrics(all_samples, fid_ref)
                            _fid = metrics_dict["frechet_inception_distance"]
                            _is = metrics_dict["inception_score_mean"]
                            logger.info(f"FID: {_fid:.3f}")
                            logger.info(f"Inception Score: {_is:.3f}")
                            # Log metrics to TensorBoard
                            run.log(
                                {
                                    "test/metrics/fid": _fid,
                                    "test/metrics/inception_score": _is,
                                },
                                step=train_steps,
                            )
                            # Top-1 checkpoint
                            if top1_fid is None or _fid < top1_fid:
                                torch.save(
                                    checkpoint,
                                    new_top1 := f"{checkpoint_dir}/top.{_fid}.pt",
                                )
                                if top1 is not None:
                                    top1.unlink(missing_ok=True)
                                top1 = Path(new_top1)
                                top1_fid = _fid
                                run.log(
                                    {"test/metrics/fid-min": top1_fid}, step=train_steps
                                )

                        del checkpoint

                    del labels, all_samples
                    gc.collect()
                    torch.cuda.empty_cache()

                if accelerator.num_processes > 1:
                    accelerator.wait_for_everyone()

            accum_steps += 1
            if accum_steps % accelerator.gradient_accumulation_steps == 0:
                train_steps += 1
                accum_steps = 0
            if train_steps >= config["train"]["max_steps"]:
                break
        if train_steps >= config["train"]["max_steps"]:
            break
    if accelerator.is_main_process:
        logger.info("Done!")
        wandb.finish()

    return accelerator


if __name__ == "__main__":
    # read config
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="./configs/base4.yaml")
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    ddp_kwargs = DistributedDataParallelKwargs(broadcast_buffers=False)
    init_kwargs = InitProcessGroupKwargs(timeout=timedelta(minutes=60))
    grad_accum = config["train"].get("gradient_accumulation_steps", 1)
    accelerator = Accelerator(
        gradient_accumulation_steps=grad_accum,
        kwargs_handlers=[ddp_kwargs, init_kwargs],
    )
    train(config, accelerator)
