from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Type

import SenNet
import nnunetv2
from nnunetv2.paths import nnUNet_results
from nnunetv2.utilities.file_path_utilities import get_output_folder
from nnunetv2.utilities.find_class_by_name import recursive_find_python_class

from SenNet.validate.trained_enhancement_inference_common import run_inference


def resolve_trainer_class(trainer_name: str) -> Type:
    trainer_class = recursive_find_python_class(
        os.path.join(SenNet.__path__[0], 'trainer'),
        trainer_name,
        'SenNet.trainer',
    )
    if trainer_class is None:
        trainer_class = recursive_find_python_class(
            os.path.join(nnunetv2.__path__[0], 'training', 'nnUNetTrainer'),
            trainer_name,
            'nnunetv2.training.nnUNetTrainer',
        )
    if trainer_class is None:
        raise RuntimeError(
            f'Unable to locate trainer class {trainer_name}. '
            f'Expected it under SenNet/trainer or nnunetv2/training/nnUNetTrainer.'
        )
    return trainer_class


def resolve_model_paths(dataset_name_or_id: str, trainer_name: str, plans_identifier: str, configuration: str, fold: str, checkpoint_name: str):
    if nnUNet_results is None:
        raise RuntimeError(
            'nnUNet_results is not defined. Please set the nnUNet_results environment variable before running inference.'
        )

    model_folder = Path(get_output_folder(dataset_name_or_id, trainer_name, plans_identifier, configuration))
    fold_name = f'fold_{fold}'
    checkpoint_path = model_folder / fold_name / checkpoint_name
    plans_json = model_folder / 'plans.json'
    dataset_json = model_folder / 'dataset.json'

    if not model_folder.is_dir():
        raise RuntimeError(f'Model folder does not exist: {model_folder}')
    if not plans_json.is_file():
        raise RuntimeError(f'plans.json not found: {plans_json}')
    if not dataset_json.is_file():
        raise RuntimeError(f'dataset.json not found: {dataset_json}')
    if not checkpoint_path.is_file():
        raise RuntimeError(f'Checkpoint not found: {checkpoint_path}')

    return model_folder, checkpoint_path, plans_json, dataset_json


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description='Enhancement-only inference for SenNet trained models. This follows nnUNet-style arguments and automatically resolves the trained model folder from dataset, trainer, plans and configuration.'
    )
    parser.add_argument('-i', type=str, required=True,
                        help='Input .nii.gz file or folder containing .nii.gz files.')
    parser.add_argument('-o', type=str, required=True,
                        help='Output folder. Results are saved into enhanced/ and residual/.')
    parser.add_argument('-d', type=str, required=True,
                        help='Dataset name or id, for example 224 or Dataset224_XXX.')
    parser.add_argument('-p', type=str, required=False, default='nnUNetPlans',
                        help='Plans identifier. Default: nnUNetPlans')
    parser.add_argument('-tr', type=str, required=True,
                        help='Trainer class name, for example FDMTrainer, FDTMTrainer or SFFDTMTrainer.')
    parser.add_argument('-c', type=str, required=True,
                        help='Configuration name, for example 3d_fullres.')
    parser.add_argument('-f', type=str, required=False, default='0',
                        help='Fold to use. Default: 0')
    parser.add_argument('-chk', type=str, required=False, default='checkpoint_final.pth',
                        help='Checkpoint name. Default: checkpoint_final.pth')
    parser.add_argument('-device', type=str, default='cuda', required=False,
                        help="Inference device. Use 'cuda', 'cuda:0' or 'cpu'. Default: cuda")
    parser.add_argument('-step_size', type=float, required=False, default=0.5,
                        help='Sliding-window step size. Default: 0.5')
    parser.add_argument('--disable_gaussian', action='store_true', required=False, default=False,
                        help='Disable Gaussian weighting during patch fusion.')
    parser.add_argument('--disable_amp', action='store_true', required=False, default=False,
                        help='Disable mixed-precision inference on CUDA.')
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    trainer_class = resolve_trainer_class(args.tr)
    model_folder, checkpoint_path, plans_json, dataset_json = resolve_model_paths(
        dataset_name_or_id=args.d,
        trainer_name=args.tr,
        plans_identifier=args.p,
        configuration=args.c,
        fold=args.f,
        checkpoint_name=args.chk,
    )

    print(f'[INFO] Resolved model folder: {model_folder}')
    print(f'[INFO] Resolved checkpoint: {checkpoint_path}')

    inference_args = argparse.Namespace(
        checkpoint=str(checkpoint_path),
        plans_json=str(plans_json),
        dataset_json=str(dataset_json),
        configuration=args.c,
        input_path=args.i,
        output_dir=args.o,
        device=args.device,
        step_size=args.step_size,
        disable_gaussian=args.disable_gaussian,
        disable_amp=args.disable_amp,
    )
    run_inference(inference_args, trainer_cls=trainer_class, model_name=args.tr)


if __name__ == '__main__':
    """
    python SenNet/validate/enhancement_predict.py \
      -d 224 \
      -i /path/to/nii_or_dir \
      -o /path/to/output_dir \
      -tr FDTMTrainer \
      -c 3d_fullres \
      -p nnUNetPlans \
      -f 0 \
      -chk checkpoint_final.pth \
      -device cuda:0
    """
    main()