import argparse
import csv
import json
import os
from pathlib import Path
from typing import Dict, List

import numpy as np
import SimpleITK as sitk
import torch


def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)


def load_json(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_nii(path: Path):
    img = sitk.ReadImage(str(path))
    arr = sitk.GetArrayFromImage(img).astype(np.float32)  # [D, H, W]
    return img, arr


def save_nii(arr: np.ndarray, ref_img: sitk.Image, out_path: Path):
    out = sitk.GetImageFromArray(arr.astype(np.float32))
    out.SetSpacing(ref_img.GetSpacing())
    out.SetOrigin(ref_img.GetOrigin())
    out.SetDirection(ref_img.GetDirection())
    sitk.WriteImage(out, str(out_path))


def normalize_for_network(arr: np.ndarray):
    mean = arr.mean()
    std = arr.std()
    arr_n = (arr - mean) / (std + 1e-8)
    return arr_n, mean, std


def denormalize_from_network(arr: np.ndarray, mean: float, std: float):
    return arr * (std + 1e-8) + mean


def sobel_like_gradient(arr: np.ndarray) -> np.ndarray:
    grads = np.gradient(arr.astype(np.float32))
    sq = sum(g * g for g in grads)
    return np.sqrt(sq + 1e-8)


def avg_gradient(arr: np.ndarray) -> float:
    return float(np.mean(sobel_like_gradient(arr)))


def high_freq_energy(arr: np.ndarray) -> float:
    return float(np.mean(np.abs(sobel_like_gradient(arr))))


def second_order_energy(arr: np.ndarray) -> float:
    arr = arr.astype(np.float32)
    grads = np.gradient(arr)
    div = 0.0
    for g in grads:
        for gg in np.gradient(g):
            div = div + gg
    return float(np.mean(np.abs(div)))


def entropy_metric(arr: np.ndarray, bins: int = 256) -> float:
    a = arr.astype(np.float32)
    a = (a - a.min()) / (a.max() - a.min() + 1e-8)
    hist, _ = np.histogram(a, bins=bins, range=(0.0, 1.0), density=False)
    hist = hist.astype(np.float64)
    hist = hist / (hist.sum() + 1e-12)
    hist = hist[hist > 0]
    return float(-np.sum(hist * np.log2(hist + 1e-12)))


def extract_state_dict(checkpoint):
    if isinstance(checkpoint, dict):
        for k in ["network_weights", "state_dict", "network", "model", "model_state_dict"]:
            if k in checkpoint and isinstance(checkpoint[k], dict):
                return checkpoint[k]
    return checkpoint


def build_network_and_load(ckpt_path: str, device: str, plans_json: str, dataset_json: str, configuration: str):
    """
    关键修复：
    不再直接 EnhancementPretrainNet(in_channels=1)
    而是按 trainer 的 build_network_architecture 方式构建
    """
    from nnunetv2.utilities.plans_handling.plans_handler import PlansManager
    from nnunetv2.utilities.label_handling.label_handling import determine_num_input_channels
    from SenNet.trainer.enhancement_pretrain_trainer import EnhancementPretrainTrainer

    plans = load_json(plans_json)
    dataset = load_json(dataset_json)

    plans_manager = PlansManager(plans)
    configuration_manager = plans_manager.get_configuration(configuration)
    label_manager = plans_manager.get_label_manager(dataset)
    num_input_channels = determine_num_input_channels(plans_manager, configuration_manager, dataset)

    net = EnhancementPretrainTrainer.build_network_architecture(
        configuration_manager.network_arch_class_name,
        configuration_manager.network_arch_init_kwargs,
        configuration_manager.network_arch_init_kwargs_req_import,
        num_input_channels,
        label_manager.num_segmentation_heads,
        enable_deep_supervision=False,
    )

    checkpoint = torch.load(ckpt_path, map_location=device)
    state_dict = extract_state_dict(checkpoint)

    cleaned = {}
    for k, v in state_dict.items():
        nk = k
        if nk.startswith("module."):
            nk = nk[7:]
        if nk.startswith("network."):
            nk = nk[8:]
        cleaned[nk] = v

    incompatible = net.load_state_dict(cleaned, strict=False)
    print(
        f"[INFO] loaded checkpoint: "
        f"missing={len(incompatible.missing_keys)}, "
        f"unexpected={len(incompatible.unexpected_keys)}"
    )
    if len(incompatible.missing_keys) > 0:
        print("[INFO] missing keys:", incompatible.missing_keys[:20])
    if len(incompatible.unexpected_keys) > 0:
        print("[INFO] unexpected keys:", incompatible.unexpected_keys[:20])

    net.to(device)
    net.eval()
    return net


def list_nii_files(input_dir: Path):
    files = sorted(input_dir.glob("*.nii.gz"))
    if not files:
        raise RuntimeError(f"No .nii.gz files found in {input_dir}")
    return files


def run_inference(args):
    device = args.device if torch.cuda.is_available() and "cuda" in args.device else "cpu"
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    enhanced_dir = output_dir / "enhanced"
    res_dir = output_dir / "raw"
    differ_dir = output_dir / "difference"

    ensure_dir(enhanced_dir)
    if args.copy_raw:
        ensure_dir(res_dir)
    os.makedirs(differ_dir, exist_ok=True)
    net = build_network_and_load(
        args.checkpoint,
        device,
        args.plans_json,
        args.dataset_json,
        args.configuration
    )

    files = list_nii_files(input_dir)

    with torch.no_grad():
        for f in files:
            ref_img, raw = load_nii(f)
            raw_norm, mean, std = normalize_for_network(raw)

            data = torch.from_numpy(raw_norm[None, None]).to(device)  # [B,C,D,H,W]
            out = net(data, return_dict=True)

            if not isinstance(out, dict):
                raise RuntimeError("Network must return dict when return_dict=True")

            enhanced_norm = out["enhanced"].detach().cpu().numpy()[0, 0]
            res_norm = out["residual"].detach().cpu().numpy()[0, 0]
            mae = np.mean(np.abs(raw_norm - enhanced_norm))
            # norm_mae = np.mean(np.abs(res_norm - enhanced_norm)) / (res_norm.max() - res_norm.min() + 1e-8)
            norm_mae_range = mae / (raw_norm.max() - raw_norm.min() + 1e-8)

            body_mask = raw_norm > -900
            if np.any(body_mask):
                mae_body = np.mean(np.abs(raw_norm[body_mask] - enhanced_norm[body_mask]))
                norm_mae_body = mae_body / (raw_norm[body_mask].max() - raw_norm[body_mask].min() + 1e-8)
            else:
                mae_body = 0.0
                norm_mae_body = 0.0
            print(f"每张图的差距: MAE={mae:.4f}")
            print(f"每张图的差距: Range Norm MAE={norm_mae_range:.6f}")
            print(f"每张图的差距: Body Norm MAE={norm_mae_body:.6f}")
            enhanced = denormalize_from_network(enhanced_norm, mean, std)
            res = denormalize_from_network(res_norm, mean, std)

            save_nii(enhanced, ref_img, enhanced_dir / f.name)
            save_nii(np.abs(enhanced_norm - raw_norm), ref_img, differ_dir / f.name)
            if args.copy_raw:
                save_nii(res, ref_img, res_dir / f.name)

            print(f"[SAVE] {f.name}")


def compute_metrics(args):
    raw_dir = Path(args.raw_dir)
    enhanced_dir = Path(args.enhanced_dir)
    out_csv = Path(args.output_csv)
    out_json = Path(args.output_json)

    ensure_dir(out_csv.parent)
    ensure_dir(out_json.parent)

    enh_files = list_nii_files(enhanced_dir)
    rows: List[Dict] = []

    for ef in enh_files:
        rf = raw_dir / ef.name
        if not rf.exists():
            print(f"[WARN] raw missing: {rf.name}")
            continue

        _, raw = load_nii(rf)
        _, enh = load_nii(ef)

        row = {
            "case": ef.name.replace(".nii.gz", ""),
            "raw_std": float(raw.std()),
            "enh_std": float(enh.std()),
            "raw_entropy": entropy_metric(raw),
            "enh_entropy": entropy_metric(enh),
            "raw_avg_gradient": avg_gradient(raw),
            "enh_avg_gradient": avg_gradient(enh),
            "raw_edge_energy": second_order_energy(raw),
            "enh_edge_energy": second_order_energy(enh),
            "raw_hf_energy": high_freq_energy(raw),
            "enh_hf_energy": high_freq_energy(enh),
            "mae_raw_enh": float(np.mean(np.abs(raw - enh))),
        }
        rows.append(row)

    if not rows:
        raise RuntimeError("No valid cases were processed.")

    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    summary = {}
    for k in rows[0].keys():
        if k == "case":
            continue
        summary[k] = float(np.mean([r[k] for r in rows]))

    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print("[DONE] metrics saved")


def build_parser():
    parser = argparse.ArgumentParser("Enhancement validation")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p1 = sub.add_parser("infer", help="Input CBCT .nii.gz and save enhanced .nii.gz")
    p1.add_argument("--checkpoint", required=True)
    p1.add_argument("--plans_json", required=True)
    p1.add_argument("--dataset_json", required=True)
    p1.add_argument("--configuration", required=True)
    p1.add_argument("--input_dir", required=True)
    p1.add_argument("--output_dir", required=True)
    p1.add_argument("--device", default="cuda:1")
    p1.add_argument("--copy_raw", action="store_true")
    p1.set_defaults(func=run_inference)

    p2 = sub.add_parser("metrics", help="Compute metrics from raw .nii.gz and enhanced .nii.gz")
    p2.add_argument("--raw_dir", required=True)
    p2.add_argument("--enhanced_dir", required=True)
    p2.add_argument("--output_csv", required=True)
    p2.add_argument("--output_json", required=True)
    p2.set_defaults(func=compute_metrics)

    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    # python /home/zr/nnUNet/SenNet/validate/enhancement_validation.py infer \
    #   --checkpoint /path/to/checkpoint_best.pth \
    #   --plans_json /path/to/nnUNetPlans.json \
    #   --dataset_json /path/to/dataset.json \
    #   --configuration 3d_fullres \
    #   --input_dir /path/to/raw_cbct_nii \
    #   --output_dir /path/to/enhancement_results \
    #   --copy_raw

    # python /home/zr/nnUNet/SenNet/validate/enhancement_validation.py metrics \
    #   --raw_dir /path/to/enhancement_results/raw \
    #   --enhanced_dir /path/to/enhancement_results/enhanced \
    #   --output_csv /path/to/enhancement_results/metrics/per_case.csv \
    #   --output_json /path/to/enhancement_results/metrics/summary.json
    main()