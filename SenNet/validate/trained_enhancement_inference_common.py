from __future__ import annotations

import argparse
import json
import os
from itertools import product
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Type

import nnunetv2
import numpy as np
import SimpleITK as sitk
import torch
from acvl_utils.cropping_and_padding.bounding_boxes import bounding_box_to_slice
from acvl_utils.cropping_and_padding.padding import pad_nd_image
from nnunetv2.inference.sliding_window_prediction import compute_gaussian, compute_steps_for_sliding_window
from nnunetv2.preprocessing.cropping.cropping import crop_to_nonzero
from nnunetv2.preprocessing.normalization.default_normalization_schemes import (
    CTNormalization,
    NoNormalization,
    RGBTo01Normalization,
    RescaleTo01Normalization,
    ZScoreNormalization,
)
from nnunetv2.preprocessing.resampling.default_resampling import compute_new_shape
from nnunetv2.utilities.find_class_by_name import recursive_find_python_class
from tqdm import tqdm


NormalizationState = Dict[str, object]


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def load_json(path: Path) -> dict:
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def save_nii(array: np.ndarray, ref_image: sitk.Image, out_path: Path) -> None:
    out = sitk.GetImageFromArray(array.astype(np.float32, copy=False))
    out.SetSpacing(ref_image.GetSpacing())
    out.SetOrigin(ref_image.GetOrigin())
    out.SetDirection(ref_image.GetDirection())
    sitk.WriteImage(out, str(out_path), True)


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


def _get_normalizer_class(scheme: str):
    normalizer_class = recursive_find_python_class(
        os.path.join(nnunetv2.__path__[0], 'preprocessing', 'normalization'),
        scheme,
        'nnunetv2.preprocessing.normalization',
    )
    if normalizer_class is None:
        raise RuntimeError(f'Unable to locate normalization class: {scheme}')
    return normalizer_class


def _normalize_channel_like_nnunet(
    image: np.ndarray,
    seg: np.ndarray,
    normalizer_class,
    use_mask_for_norm: bool,
    intensityproperties: dict,
) -> Tuple[np.ndarray, NormalizationState]:
    image = image.astype(np.float32, copy=True)

    if issubclass(normalizer_class, CTNormalization):
        mean_intensity = float(intensityproperties['mean'])
        std_intensity = float(max(intensityproperties['std'], 1e-8))
        lower_bound = float(intensityproperties['percentile_00_5'])
        upper_bound = float(intensityproperties['percentile_99_5'])
        np.clip(image, lower_bound, upper_bound, out=image)
        image -= mean_intensity
        image /= std_intensity
        state: NormalizationState = {
            'type': 'ct',
            'mean': mean_intensity,
            'std': std_intensity,
        }
        return image, state

    if issubclass(normalizer_class, ZScoreNormalization):
        if use_mask_for_norm:
            mask = seg >= 0
            if np.any(mask):
                mean = float(image[mask].mean())
                std = float(max(image[mask].std(), 1e-8))
                image[mask] = (image[mask] - mean) / std
            else:
                mean = float(image.mean())
                std = float(max(image.std(), 1e-8))
                image -= mean
                image /= std
                mask = None
        else:
            mean = float(image.mean())
            std = float(max(image.std(), 1e-8))
            image -= mean
            image /= std
            mask = None
        state = {
            'type': 'zscore',
            'mean': mean,
            'std': std,
            'mask': mask,
        }
        return image, state

    if issubclass(normalizer_class, NoNormalization):
        return image, {'type': 'identity'}

    if issubclass(normalizer_class, RescaleTo01Normalization):
        min_value = float(image.min())
        max_value = float(image.max())
        scale = float(max(max_value - min_value, 1e-8))
        image -= min_value
        image /= scale
        return image, {'type': 'rescale01', 'min': min_value, 'scale': scale}

    if issubclass(normalizer_class, RGBTo01Normalization):
        image /= 255.0
        return image, {'type': 'rgb01', 'scale': 255.0}

    raise NotImplementedError(
        f'Enhancement inference does not support inverse normalization for custom scheme {normalizer_class.__name__}.'
    )


def _invert_normalization(
    array: np.ndarray,
    states: List[NormalizationState],
    output_kind: str,
) -> np.ndarray:
    restored = array.astype(np.float32, copy=True)
    for channel_idx, state in enumerate(states):
        state_type = state['type']
        if state_type == 'identity':
            continue

        if state_type == 'ct':
            std = float(state['std'])
            if output_kind == 'enhanced':
                restored[channel_idx] = restored[channel_idx] * std + float(state['mean'])
            else:
                restored[channel_idx] = restored[channel_idx] * std
            continue

        if state_type == 'zscore':
            std = float(state['std'])
            mean = float(state['mean'])
            mask = state.get('mask', None)
            if mask is None:
                if output_kind == 'enhanced':
                    restored[channel_idx] = restored[channel_idx] * std + mean
                else:
                    restored[channel_idx] = restored[channel_idx] * std
            else:
                mask = np.asarray(mask, dtype=bool)
                if output_kind == 'enhanced':
                    restored[channel_idx][mask] = restored[channel_idx][mask] * std + mean
                else:
                    restored[channel_idx][mask] = restored[channel_idx][mask] * std
            continue

        if state_type == 'rescale01':
            scale = float(state['scale'])
            if output_kind == 'enhanced':
                restored[channel_idx] = restored[channel_idx] * scale + float(state['min'])
            else:
                restored[channel_idx] = restored[channel_idx] * scale
            continue

        if state_type == 'rgb01':
            restored[channel_idx] = restored[channel_idx] * float(state['scale'])
            continue

        raise RuntimeError(f'Unsupported normalization state type: {state_type}')
    return restored


def preprocess_case_like_nnunet(
    case_path: Path,
    plans_manager,
    configuration_manager,
    dataset_json: dict,
) -> Dict[str, object]:
    rw = plans_manager.image_reader_writer_class()
    data, properties = rw.read_images([str(case_path)])
    properties = dict(properties)

    data = data.astype(np.float32, copy=True)
    transpose_forward = [0, *[i + 1 for i in plans_manager.transpose_forward]]
    data = data.transpose(transpose_forward)

    original_spacing = [properties['spacing'][i] for i in plans_manager.transpose_forward]
    properties['shape_before_cropping'] = data.shape[1:]

    data, pseudo_seg, bbox = crop_to_nonzero(data, seg=None)
    properties['bbox_used_for_cropping'] = bbox
    properties['shape_after_cropping_and_before_resampling'] = data.shape[1:]

    normalization_states: List[NormalizationState] = []
    for channel_idx in range(data.shape[0]):
        normalizer_class = _get_normalizer_class(configuration_manager.normalization_schemes[channel_idx])
        data[channel_idx], state = _normalize_channel_like_nnunet(
            data[channel_idx],
            pseudo_seg[0],
            normalizer_class,
            configuration_manager.use_mask_for_norm[channel_idx],
            plans_manager.foreground_intensity_properties_per_channel[str(channel_idx)],
        )
        normalization_states.append(state)

    target_spacing = list(configuration_manager.spacing)
    if len(target_spacing) < len(data.shape[1:]):
        target_spacing = [original_spacing[0], *target_spacing]
    new_shape = compute_new_shape(data.shape[1:], original_spacing, target_spacing)
    data = configuration_manager.resampling_fn_data(data, new_shape, original_spacing, target_spacing)
    if isinstance(data, torch.Tensor):
        data = data.cpu().numpy()

    reference_image = sitk.ReadImage(str(case_path))
    return {
        'data': data.astype(np.float32, copy=False),
        'properties': properties,
        'reference_image': reference_image,
        'normalization_states': normalization_states,
    }


def revert_preprocessing_like_nnunet(
    array: np.ndarray,
    properties: dict,
    plans_manager,
    configuration_manager,
    normalization_states: List[NormalizationState],
    output_kind: str,
) -> np.ndarray:
    spacing_transposed = [properties['spacing'][i] for i in plans_manager.transpose_forward]
    current_spacing = list(configuration_manager.spacing)
    if len(current_spacing) < len(properties['shape_after_cropping_and_before_resampling']):
        current_spacing = [spacing_transposed[0], *current_spacing]

    restored = configuration_manager.resampling_fn_data(
        array,
        properties['shape_after_cropping_and_before_resampling'],
        current_spacing,
        spacing_transposed,
    )
    if isinstance(restored, torch.Tensor):
        restored = restored.cpu().numpy()
    restored = restored.astype(np.float32, copy=False)

    restored = _invert_normalization(restored, normalization_states, output_kind=output_kind)

    uncropped = np.zeros((restored.shape[0], *properties['shape_before_cropping']), dtype=np.float32)
    slicer = bounding_box_to_slice(properties['bbox_used_for_cropping'])
    uncropped[(slice(None), *slicer)] = restored

    transpose_backward = [0, *[i + 1 for i in plans_manager.transpose_backward]]
    uncropped = uncropped.transpose(transpose_backward)
    return uncropped


def build_enhancer_and_patch_size(
    checkpoint_path: Path,
    device: torch.device,
    plans_json: Path,
    dataset_json: Path,
    configuration: str,
    trainer_cls: Type,
    model_name: str,
):
    from nnunetv2.utilities.label_handling.label_handling import determine_num_input_channels
    from nnunetv2.utilities.plans_handling.plans_handler import PlansManager

    plans = load_json(plans_json)
    dataset = load_json(dataset_json)

    plans_manager = PlansManager(plans)
    configuration_manager = plans_manager.get_configuration(configuration)
    label_manager = plans_manager.get_label_manager(dataset)
    num_input_channels = determine_num_input_channels(plans_manager, configuration_manager, dataset)

    network = trainer_cls.build_network_architecture(
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
        f"[INFO] Loaded trained {model_name} checkpoint {checkpoint_path.name}: "
        f"missing={len(incompatible.missing_keys)}, unexpected={len(incompatible.unexpected_keys)}"
    )
    if incompatible.missing_keys:
        print(f"[INFO] Missing keys (first 20): {incompatible.missing_keys[:20]}")
    if incompatible.unexpected_keys:
        print(f"[INFO] Unexpected keys (first 20): {incompatible.unexpected_keys[:20]}")

    network.to(device)
    network.eval()
    enhancer = network.enhancer
    enhancer.eval()
    patch_size = tuple(int(i) for i in configuration_manager.patch_size)
    return enhancer, patch_size, plans_manager, configuration_manager, dataset


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
    enhancer,
    input_tensor: torch.Tensor,
    patch_size: Sequence[int],
    device: torch.device,
    case_name: Optional[str] = None,
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

    patch_desc = f'Patches<{case_name}>' if case_name is not None else 'Patches'

    with torch.no_grad():
        autocast_enabled = use_amp and device.type == 'cuda'
        for slicer in tqdm(slicers, desc=patch_desc, unit='patch', leave=False):
            patch = padded[slicer].unsqueeze(0).to(device)
            with torch.autocast(device_type=device.type, enabled=autocast_enabled):
                enhanced_patch, residual_patch = enhancer(patch)

            patch_cpu = patch[0].detach().to('cpu', dtype=torch.float32)
            enhanced_patch = enhanced_patch[0].detach().to('cpu', dtype=torch.float32)
            residual_patch = residual_patch[0].detach().to('cpu', dtype=torch.float32)

            if enhanced_patch.shape != patch_cpu.shape:
                raise RuntimeError(
                    f"Enhanced patch shape mismatch: got {tuple(enhanced_patch.shape)} but expected {tuple(patch_cpu.shape)}"
                )
            if residual_patch.shape != patch_cpu.shape:
                raise RuntimeError(
                    f"Residual patch shape mismatch: got {tuple(residual_patch.shape)} but expected {tuple(patch_cpu.shape)}"
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


def run_inference(args, trainer_cls: Type, model_name: str) -> None:
    device = torch.device(args.device if torch.cuda.is_available() and 'cuda' in args.device else 'cpu')
    input_path = Path(args.input_path)
    output_dir = Path(args.output_dir)
    enhanced_dir = output_dir / 'enhanced'
    residual_dir = output_dir / 'residual'

    ensure_dir(enhanced_dir)
    ensure_dir(residual_dir)

    enhancer, patch_size, plans_manager, configuration_manager, dataset_json = build_enhancer_and_patch_size(
        checkpoint_path=Path(args.checkpoint),
        device=device,
        plans_json=Path(args.plans_json),
        dataset_json=Path(args.dataset_json),
        configuration=args.configuration,
        trainer_cls=trainer_cls,
        model_name=model_name,
    )
    input_files = collect_input_files(input_path)

    print(f'[INFO] Using device: {device}')
    print(f'[INFO] Patch size: {patch_size}')
    print(f'[INFO] Number of cases: {len(input_files)}')
    print('[INFO] Preprocessing: nnUNet-consistent transpose -> crop -> normalize -> resample')

    # case_progress = tqdm(input_files, desc=f'{model_name} trained enhancement inference', unit='case')
    for case_path in input_files:
        # case_progress.set_postfix_str(case_path.name)
        print(f'[INFO] Processing case: {case_path.name}')
        preprocessed = preprocess_case_like_nnunet(
            case_path=case_path,
            plans_manager=plans_manager,
            configuration_manager=configuration_manager,
            dataset_json=dataset_json,
        )
        input_tensor = torch.from_numpy(preprocessed['data'])

        prediction = predict_sliding_window(
            enhancer=enhancer,
            input_tensor=input_tensor,
            patch_size=patch_size,
            device=device,
            case_name=case_path.name,
            step_size=args.step_size,
            use_gaussian=not args.disable_gaussian,
            use_amp=not args.disable_amp,
        )

        enhanced = revert_preprocessing_like_nnunet(
            prediction['enhanced'].numpy(),
            preprocessed['properties'],
            plans_manager,
            configuration_manager,
            preprocessed['normalization_states'],
            output_kind='enhanced',
        )[0]
        residual = revert_preprocessing_like_nnunet(
            prediction['residual'].numpy(),
            preprocessed['properties'],
            plans_manager,
            configuration_manager,
            preprocessed['normalization_states'],
            output_kind='residual',
        )[0]

        save_nii(enhanced, preprocessed['reference_image'], enhanced_dir / case_path.name)
        save_nii(residual, preprocessed['reference_image'], residual_dir / case_path.name)
        tqdm.write(f'[SAVE] {case_path.name}')



def build_parser(model_name: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(f'{model_name} trained-model enhancement inference')
    parser.add_argument('--checkpoint', required=True, help=f'Path to the trained {model_name} model checkpoint (.pth).')
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