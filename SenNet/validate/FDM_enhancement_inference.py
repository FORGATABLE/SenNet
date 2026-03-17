from __future__ import annotations

import argparse
import json
from itertools import product
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import SimpleITK as sitk
import torch
from acvl_utils.cropping_and_padding.padding import pad_nd_image
from tqdm import tqdm
from nnunetv2.inference.sliding_window_prediction import compute_gaussian, compute_steps_for_sliding_window


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def load_json(path: Path) -> dict:
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def load_nii(path: Path) -> Tuple[sitk.Image, np.ndarray]:
    image = sitk.ReadImage(str(path))
    array = sitk.GetArrayFromImage(image).astype(np.float32)
    return image, array


def save_nii(array: np.ndarray, ref_image: sitk.Image, out_path: Path) -> None:
    out = sitk.GetImageFromArray(array.astype(np.float32, copy=False))
    out.SetSpacing(ref_image.GetSpacing())
    out.SetOrigin(ref_image.GetOrigin())
    out.SetDirection(ref_image.GetDirection())
    sitk.WriteImage(out, str(out_path), True)


def normalize_volume(array: np.ndarray) -> Tuple[np.ndarray, float, float]:
    mean = float(array.mean())
    std = float(array.std())
    if std < 1e-8:
        std = 1.0
    normalized = (array - mean) / std
    return normalized.astype(np.float32, copy=False), mean, std


def denormalize_volume(array: np.ndarray, mean: float, std: float) -> np.ndarray:
    return array * std + mean


def denormalize_residual(array: np.ndarray, std: float) -> np.ndarray:
    return array * std


def extract_state_dict(checkpoint):
    if isinstance(checkpoint, dict):
        for key in ('network_weights', 'state_dict', 'network', 'model', 'model_state_dict'):
            if key in checkpoint and isinstance(checkpoint[key], dict):
                return checkpoint[key]
    return checkpoint


def clean_state_dict_keys(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    cleaned = {}
    for key, value in state_dict.items():
        new_key = key
        if new_key.startswith('module.'):
            new_key = new_key[7:]
        if new_key.startswith('network.'):
            new_key = new_key[8:]
        cleaned[new_key] = value
    return cleaned


def build_network_and_patch_size(
    checkpoint_path: Path,
    device: torch.device,
    plans_json: Path,
    dataset_json: Path,
    configuration: str,
):
    from nnunetv2.utilities.label_handling.label_handling import determine_num_input_channels
    from nnunetv2.utilities.plans_handling.plans_handler import PlansManager
    from SenNet.trainer.FDM_enhancedPreTrainer import FDMEnhancedPreTrainer

    plans = load_json(plans_json)
    dataset = load_json(dataset_json)

    plans_manager = PlansManager(plans)
    configuration_manager = plans_manager.get_configuration(configuration)
    label_manager = plans_manager.get_label_manager(dataset)
    num_input_channels = determine_num_input_channels(plans_manager, configuration_manager, dataset)

    network = FDMEnhancedPreTrainer.build_network_architecture(
        configuration_manager.network_arch_class_name,
        configuration_manager.network_arch_init_kwargs,
        configuration_manager.network_arch_init_kwargs_req_import,
        num_input_channels,
        label_manager.num_segmentation_heads,
        enable_deep_supervision=False,
    )

    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    state_dict = clean_state_dict_keys(extract_state_dict(checkpoint))
    incompatible = network.load_state_dict(state_dict, strict=False)
    print(
        f"[INFO] Loaded checkpoint {checkpoint_path.name}: "
        f"missing={len(incompatible.missing_keys)}, unexpected={len(incompatible.unexpected_keys)}"
    )
    if incompatible.missing_keys:
        print(f"[INFO] Missing keys (first 20): {incompatible.missing_keys[:20]}")
    if incompatible.unexpected_keys:
        print(f"[INFO] Unexpected keys (first 20): {incompatible.unexpected_keys[:20]}")

    network.to(device)
    network.eval()
    patch_size = tuple(int(i) for i in configuration_manager.patch_size)
    return network, patch_size


def collect_input_files(input_path: Path) -> List[Path]:
    if input_path.is_file():
        if input_path.name.endswith('.nii.gz'):
            return [input_path]
        raise RuntimeError(f'Input file is not a .nii.gz image: {input_path}')

    if not input_path.is_dir():
        raise RuntimeError(f'Input path does not exist: {input_path}')

    files = sorted(input_path.glob('*.nii.gz'))
    if not files:
        raise RuntimeError(f'No .nii.gz files found in {input_path}')
    return files


def build_slicers(image_size: Sequence[int], patch_size: Sequence[int], step_size: float) -> List[Tuple[slice, ...]]:
    steps = compute_steps_for_sliding_window(tuple(image_size), tuple(patch_size), step_size)
    slicers: List[Tuple[slice, ...]] = []
    for coords in product(*steps):
        spatial_slices = tuple(slice(start, start + size) for start, size in zip(coords, patch_size))
        slicers.append((slice(None), *spatial_slices))
    return slicers


def predict_sliding_window(
    network,
    input_tensor: torch.Tensor,
    patch_size: Sequence[int],
    device: torch.device,
    step_size: float = 0.5,
    use_gaussian: bool = True,
    use_amp: bool = True,
) -> Dict[str, torch.Tensor]:
    assert input_tensor.ndim == 4, 'Expected input tensor with shape [C, D, H, W].'

    padded, revert_padding = pad_nd_image(
        input_tensor,
        new_shape=tuple(patch_size),
        mode='constant',
        kwargs={'value': 0},
        return_slicer=True,
        shape_must_be_divisible_by=None,
    )

    slicers = build_slicers(padded.shape[1:], patch_size, step_size)
    gaussian = None
    if use_gaussian:
        gaussian = compute_gaussian(
            tuple(patch_size),
            sigma_scale=1.0 / 8.0,
            value_scaling_factor=1.0,
            dtype=torch.float32,
            device=torch.device('cpu'),
        )

    prediction_enhanced = torch.zeros_like(padded, dtype=torch.float32)
    prediction_residual = torch.zeros_like(padded, dtype=torch.float32)
    prediction_count = torch.zeros(padded.shape[1:], dtype=torch.float32)

    with torch.no_grad():
        autocast_enabled = use_amp and device.type == 'cuda'
        warned_legacy_output = False
        for slicer in tqdm(slicers):
            patch = padded[slicer].unsqueeze(0).to(device)
            with torch.autocast(device_type=device.type, enabled=autocast_enabled):
                output = network(patch, return_dict=True)

            patch_cpu = patch[0].detach().to('cpu', dtype=torch.float32)
            enhanced_patch = output['enhanced'][0].detach().to('cpu', dtype=torch.float32)
            residual_patch = output['residual'][0].detach().to('cpu', dtype=torch.float32)

            if residual_patch.shape != patch_cpu.shape:
                raise RuntimeError(
                    f"Residual patch shape mismatch: got {tuple(residual_patch.shape)} but expected {tuple(patch_cpu.shape)}"
                )
            if enhanced_patch.shape != patch_cpu.shape:
                if enhanced_patch.shape[1:] == patch_cpu.shape[1:]:
                    if not warned_legacy_output:
                        print(
                            "[WARN] The checkpoint returned a feature-shaped enhanced output. "
                            "Falling back to raw_patch + residual for compatibility."
                        )
                        warned_legacy_output = True
                    enhanced_patch = patch_cpu + residual_patch
                else:
                    raise RuntimeError(
                        f"Enhanced patch shape mismatch: got {tuple(enhanced_patch.shape)} but expected {tuple(patch_cpu.shape)}"
                    )

            if gaussian is not None:
                weight = gaussian.unsqueeze(0)
                enhanced_patch = enhanced_patch * weight
                residual_patch = residual_patch * weight
                prediction_count[slicer[1:]] += gaussian
            else:
                prediction_count[slicer[1:]] += 1.0

            prediction_enhanced[slicer] += enhanced_patch
            prediction_residual[slicer] += residual_patch

    prediction_count = torch.clamp(prediction_count, min=1e-8)
    prediction_enhanced = prediction_enhanced / prediction_count.unsqueeze(0)
    prediction_residual = prediction_residual / prediction_count.unsqueeze(0)

    revert = (slice(None), *revert_padding[1:])
    return {
        'enhanced': prediction_enhanced[revert],
        'residual': prediction_residual[revert],
    }


def run_inference(args) -> None:
    device = torch.device(args.device if torch.cuda.is_available() and 'cuda' in args.device else 'cpu')
    input_path = Path(args.input_path)
    output_dir = Path(args.output_dir)
    enhanced_dir = output_dir / 'enhanced'
    residual_dir = output_dir / 'residual'

    ensure_dir(enhanced_dir)
    ensure_dir(residual_dir)

    network, patch_size = build_network_and_patch_size(
        checkpoint_path=Path(args.checkpoint),
        device=device,
        plans_json=Path(args.plans_json),
        dataset_json=Path(args.dataset_json),
        configuration=args.configuration,
    )
    input_files = collect_input_files(input_path)

    print(f'[INFO] Using device: {device}')
    print(f'[INFO] Patch size: {patch_size}')
    print(f'[INFO] Number of cases: {len(input_files)}')

    # progress_bar = tqdm(input_files, desc='FDM enhancement inference', unit='case')
    for case_path in input_files:
        # progress_bar.set_postfix_str(case_path.name)
        ref_image, raw = load_nii(case_path)
        raw_norm, mean, std = normalize_volume(raw)
        raw_tensor = torch.from_numpy(raw_norm[None])

        prediction = predict_sliding_window(
            network=network,
            input_tensor=raw_tensor,
            patch_size=patch_size,
            device=device,
            step_size=args.step_size,
            use_gaussian=not args.disable_gaussian,
            use_amp=not args.disable_amp,
        )

        enhanced_norm = prediction['enhanced'][0].numpy()
        residual_norm = prediction['residual'][0].numpy()

        enhanced = denormalize_volume(enhanced_norm, mean, std)
        residual = denormalize_residual(residual_norm, std)

        save_nii(enhanced, ref_image, enhanced_dir / case_path.name)
        save_nii(residual, ref_image, residual_dir / case_path.name)
        tqdm.write(f'[SAVE] {case_path.name}')


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser('FDM enhancement inference')
    parser.add_argument('--checkpoint', required=True, help='Path to the trained FDM enhancer checkpoint (.pth).')
    parser.add_argument('--plans_json', required=True, help='Path to nnUNet plans.json used during training.')
    parser.add_argument('--dataset_json', required=True, help='Path to dataset.json used during training.')
    parser.add_argument('--configuration', required=True, help='nnUNet configuration name, for example 3d_fullres.')
    parser.add_argument('--input_path', required=True, help='A single .nii.gz file or a directory containing .nii.gz files.')
    parser.add_argument('--output_dir', required=True, help='Output directory. Results are saved in enhanced/ and residual/.')
    parser.add_argument('--device', default='cuda:0', help='Inference device, for example cuda:0 or cpu.')
    parser.add_argument('--step_size', type=float, default=0.5, help='Sliding-window step size in (0, 1].')
    parser.add_argument('--disable_gaussian', action='store_true', help='Disable Gaussian weighting during patch fusion.')
    parser.add_argument('--disable_amp', action='store_true', help='Disable mixed-precision inference on CUDA.')
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    run_inference(args)


if __name__ == '__main__':
    """
    python SenNet/validate/FDM_enhancement_inference.py \
  --checkpoint /path/to/checkpoint_best.pth \
  --plans_json /path/to/nnUNetPlans.json \
  --dataset_json /path/to/dataset.json \
  --configuration 3d_fullres \
  --input_path /path/to/nii_or_dir \
  --output_dir /path/to/fdm_enhance_out \
  --device cuda:0"""
    main()
