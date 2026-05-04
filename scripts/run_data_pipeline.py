import argparse
import os
import sys
from pathlib import Path


PROJ_ROOT = Path(__file__).resolve().parent.parent  # traditional/
REPO_ROOT = PROJ_ROOT.parent  # 3D_Reconstruction/
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from traditional.pipeline import run_pipeline


def main():
    parser = argparse.ArgumentParser(description="Run the traditional SfM pipeline on a folder of images")
    parser.add_argument("--image-dir", default=str(PROJ_ROOT / "data_image/Dog_RGB"), help="Directory that contains input images")
    parser.add_argument("--config", default=str(PROJ_ROOT / "config" / "camera.json"), help="Camera config JSON path")
    parser.add_argument("--output-dir", default=str(PROJ_ROOT / "output"), help="Output directory for reconstructed files")
    parser.add_argument("--skip-undistort", action="store_true", help="Skip image undistortion")
    parser.add_argument("--max-dim", type=int, default=2272, help="Maximum image dimension (resize larger images)")
    args = parser.parse_args()

    if not os.path.isdir(args.image_dir):
        raise FileNotFoundError(f"Image directory not found: {args.image_dir}")

    if not os.path.isfile(args.config):
        raise FileNotFoundError(f"Camera config not found: {args.config}")

    run_pipeline(
        image_dir=args.image_dir,
        config_path=args.config,
        output_dir=args.output_dir,
        skip_undistort=args.skip_undistort,
        max_image_dim=args.max_dim,
    )


if __name__ == "__main__":
    main()