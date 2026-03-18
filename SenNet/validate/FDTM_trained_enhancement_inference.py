from __future__ import annotations

from SenNet.trainer.FDTM_Trainer import FDTMTrainer
from SenNet.validate.trained_enhancement_inference_common import build_parser, run_inference


MODEL_NAME = 'FDTM'


def main() -> None:
    parser = build_parser(MODEL_NAME)
    args = parser.parse_args()
    run_inference(args, trainer_cls=FDTMTrainer, model_name=MODEL_NAME)


if __name__ == '__main__':
    """
    python SenNet/validate/FDTM_trained_enhancement_inference.py \
      --checkpoint /path/to/checkpoint_final.pth \
      --plans_json /path/to/nnUNetPlans.json \
      --dataset_json /path/to/dataset.json \
      --configuration 3d_fullres \
      --input_path /path/to/nii_or_dir \
      --output_dir /path/to/fdtm_trained_enhance_out \
      --device cuda:0
    """
    main()