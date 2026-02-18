#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path

import cv2
import mrcfile
import numpy as np
import tifffile


@dataclass
class AlignmentResult:
    tif_path: Path
    z_offset: int
    y_offset: int
    x_offset: int
    score: float


def read_volume_any(path: Path) -> np.ndarray:
    if path.suffix.lower() in {".tif", ".tiff"}:
        volume = tifffile.imread(path)
    elif path.suffix.lower() == ".mrc":
        with mrcfile.open(path, permissive=True) as mrc_handle:
            if mrc_handle.data is None:
                raise ValueError(f"MRC file has no readable data block: {path}")
            volume = np.asarray(mrc_handle.data)
    else:
        raise ValueError(f"Unsupported file format: {path}")

    if volume.ndim == 2:
        volume = volume[np.newaxis, :, :]
    if volume.ndim != 3:
        raise ValueError(f"Expected 3D volume for {path}, got shape={volume.shape}")
    return volume.astype(np.float32, copy=False)


def normalize_for_matching(image: np.ndarray) -> np.ndarray:
    image = image.astype(np.float32, copy=False)
    image_min = float(image.min())
    image_max = float(image.max())
    if image_max <= image_min:
        return np.zeros_like(image, dtype=np.float32)
    return (image - image_min) / (image_max - image_min)


def compute_depth_profile(volume: np.ndarray) -> np.ndarray:
    return volume.mean(axis=(1, 2))


def normalized_cross_correlation(first: np.ndarray, second: np.ndarray) -> float:
    first_centered = first - first.mean()
    second_centered = second - second.mean()
    denominator = np.linalg.norm(first_centered) * np.linalg.norm(second_centered)
    if denominator == 0:
        return -1.0
    return float((first_centered * second_centered).sum() / denominator)


def candidate_z_positions(
    mrc_volume: np.ndarray,
    tif_volume: np.ndarray,
    max_candidates: int,
) -> list[int]:
    mrc_depth, _, _ = mrc_volume.shape
    tif_depth, _, _ = tif_volume.shape
    if tif_depth > mrc_depth:
        raise ValueError(
            f"TIFF depth {tif_depth} is larger than MRC depth {mrc_depth}; cannot align."
        )

    mrc_profile = compute_depth_profile(mrc_volume)
    tif_profile = compute_depth_profile(tif_volume)

    all_scores: list[tuple[float, int]] = []
    max_start = mrc_depth - tif_depth
    for z_start in range(max_start + 1):
        profile_window = mrc_profile[z_start : z_start + tif_depth]
        score = normalized_cross_correlation(profile_window, tif_profile)
        all_scores.append((score, z_start))

    all_scores.sort(reverse=True, key=lambda item: item[0])
    return [z_start for _, z_start in all_scores[:max_candidates]]


def best_xy_from_template(
    mrc_subvolume: np.ndarray,
    tif_volume: np.ndarray,
) -> tuple[int, int, float]:
    mrc_mip = mrc_subvolume.max(axis=0)
    tif_mip = tif_volume.max(axis=0)

    mrc_mip = normalize_for_matching(mrc_mip)
    tif_mip = normalize_for_matching(tif_mip)

    mrc_height, mrc_width = mrc_mip.shape
    tif_height, tif_width = tif_mip.shape
    if tif_height > mrc_height or tif_width > mrc_width:
        raise ValueError(
            "TIFF XY size is larger than candidate MRC XY size; cannot use template matching."
        )

    template_scores = cv2.matchTemplate(mrc_mip, tif_mip, cv2.TM_CCOEFF_NORMED)
    _, score_max, _, max_location = cv2.minMaxLoc(template_scores)
    x_offset, y_offset = int(max_location[0]), int(max_location[1])
    return y_offset, x_offset, float(score_max)


def refine_with_local_3d_ncc(
    mrc_volume: np.ndarray,
    tif_volume: np.ndarray,
    z_guess: int,
    y_guess: int,
    x_guess: int,
    radius_z: int,
    radius_xy: int,
    ncc_stride: int,
) -> tuple[int, int, int, float]:
    mrc_depth, mrc_height, mrc_width = mrc_volume.shape
    tif_depth, tif_height, tif_width = tif_volume.shape

    best_score = -np.inf
    best_offsets = (z_guess, y_guess, x_guess)

    tif_for_ncc = tif_volume[::ncc_stride, ::ncc_stride, ::ncc_stride]

    for z_offset in range(max(0, z_guess - radius_z), min(mrc_depth - tif_depth, z_guess + radius_z) + 1):
        for y_offset in range(max(0, y_guess - radius_xy), min(mrc_height - tif_height, y_guess + radius_xy) + 1):
            for x_offset in range(max(0, x_guess - radius_xy), min(mrc_width - tif_width, x_guess + radius_xy) + 1):
                mrc_patch = mrc_volume[
                    z_offset : z_offset + tif_depth,
                    y_offset : y_offset + tif_height,
                    x_offset : x_offset + tif_width,
                ]
                mrc_for_ncc = mrc_patch[::ncc_stride, ::ncc_stride, ::ncc_stride]
                ncc_score = normalized_cross_correlation(mrc_for_ncc, tif_for_ncc)
                if ncc_score > best_score:
                    best_score = ncc_score
                    best_offsets = (z_offset, y_offset, x_offset)

    return (*best_offsets, float(best_score))


def align_single_tif(
    mrc_volume: np.ndarray,
    tif_path: Path,
    max_z_candidates: int,
    refine_radius_z: int,
    refine_radius_xy: int,
    ncc_stride: int,
) -> AlignmentResult:
    tif_volume = read_volume_any(tif_path)
    mrc_depth, mrc_height, mrc_width = mrc_volume.shape
    tif_depth, tif_height, tif_width = tif_volume.shape

    if (mrc_depth, mrc_height, mrc_width) == (tif_depth, tif_height, tif_width):
        fast_score = normalized_cross_correlation(
            mrc_volume[::ncc_stride, ::ncc_stride, ::ncc_stride],
            tif_volume[::ncc_stride, ::ncc_stride, ::ncc_stride],
        )
        return AlignmentResult(
            tif_path=tif_path,
            z_offset=0,
            y_offset=0,
            x_offset=0,
            score=fast_score,
        )

    z_candidates = candidate_z_positions(
        mrc_volume=mrc_volume,
        tif_volume=tif_volume,
        max_candidates=max_z_candidates,
    )

    best_alignment: AlignmentResult | None = None
    for z_candidate in z_candidates:
        mrc_candidate = mrc_volume[z_candidate : z_candidate + tif_volume.shape[0]]
        y_guess, x_guess, _ = best_xy_from_template(mrc_candidate, tif_volume)
        z_refined, y_refined, x_refined, refined_score = refine_with_local_3d_ncc(
            mrc_volume=mrc_volume,
            tif_volume=tif_volume,
            z_guess=z_candidate,
            y_guess=y_guess,
            x_guess=x_guess,
            radius_z=refine_radius_z,
            radius_xy=refine_radius_xy,
            ncc_stride=ncc_stride,
        )

        result = AlignmentResult(
            tif_path=tif_path,
            z_offset=z_refined,
            y_offset=y_refined,
            x_offset=x_refined,
            score=refined_score,
        )
        if best_alignment is None or result.score > best_alignment.score:
            best_alignment = result

    if best_alignment is None:
        raise RuntimeError(f"Could not estimate alignment for {tif_path}")
    return best_alignment


def write_offsets_csv(results: list[AlignmentResult], output_csv_path: Path) -> None:
    output_csv_path.parent.mkdir(parents=True, exist_ok=True)
    with output_csv_path.open("w", newline="", encoding="utf-8") as csv_handle:
        writer = csv.writer(csv_handle)
        writer.writerow(["tif_file", "z_offset", "y_offset", "x_offset", "score"])
        for result in results:
            writer.writerow(
                [
                    result.tif_path.name,
                    result.z_offset,
                    result.y_offset,
                    result.x_offset,
                    f"{result.score:.6f}",
                ]
            )


def merge_tifs_into_volume(
    results: list[AlignmentResult],
    output_volume_shape: tuple[int, int, int],
    output_tif_path: Path,
) -> None:
    merged = np.zeros(output_volume_shape, dtype=np.float32)
    weights = np.zeros(output_volume_shape, dtype=np.float32)

    for result in results:
        tif_volume = read_volume_any(result.tif_path)
        tif_depth, tif_height, tif_width = tif_volume.shape
        z_start, y_start, x_start = result.z_offset, result.y_offset, result.x_offset
        z_end = min(output_volume_shape[0], z_start + tif_depth)
        y_end = min(output_volume_shape[1], y_start + tif_height)
        x_end = min(output_volume_shape[2], x_start + tif_width)

        if z_end <= z_start or y_end <= y_start or x_end <= x_start:
            continue

        tif_crop = tif_volume[: z_end - z_start, : y_end - y_start, : x_end - x_start]
        merged[z_start:z_end, y_start:y_end, x_start:x_end] += tif_crop
        weights[z_start:z_end, y_start:y_end, x_start:x_end] += 1.0

    valid_mask = weights > 0
    merged[valid_mask] /= weights[valid_mask]
    output_tif_path.parent.mkdir(parents=True, exist_ok=True)
    tifffile.imwrite(output_tif_path, merged, imagej=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Estimate where each TIFF stack belongs in a reference MRC volume, "
            "write offsets, and optionally merge TIFF stacks into one reconstructed volume."
        )
    )
    parser.add_argument("--mrc", type=Path, required=True, help="Path to reference .mrc full volume")
    parser.add_argument(
        "--tif-dir",
        type=Path,
        default=Path("dataset"),
        help="Folder with partial TIFF stacks to place in the MRC volume",
    )
    parser.add_argument(
        "--tif-pattern",
        type=str,
        default="*.tif",
        help="Glob pattern for TIFF stacks inside --tif-dir",
    )
    parser.add_argument(
        "--offsets-csv",
        type=Path,
        default=Path("output/tif_offsets_in_mrc.csv"),
        help="Output CSV path for estimated offsets",
    )
    parser.add_argument(
        "--max-z-candidates",
        type=int,
        default=3,
        help="How many top z candidates to test before local 3D refinement",
    )
    parser.add_argument(
        "--refine-radius-z",
        type=int,
        default=3,
        help="Local search radius in z around each candidate",
    )
    parser.add_argument(
        "--refine-radius-xy",
        type=int,
        default=8,
        help="Local search radius in x/y around template match guess",
    )
    parser.add_argument(
        "--ncc-stride",
        type=int,
        default=8,
        help="Stride for 3D NCC scoring (higher is faster, lower is more accurate)",
    )
    parser.add_argument(
        "--merged-tif",
        type=Path,
        default=None,
        help="Optional output path for merged reconstructed TIFF volume",
    )
    parser.add_argument(
        "--export-mrc-tif",
        type=Path,
        default=None,
        help="Optional output path to export the full MRC volume as a TIFF stack",
    )
    return parser.parse_args()


def main() -> None:
    arguments = parse_args()
    mrc_volume = read_volume_any(arguments.mrc)

    tif_paths = sorted(arguments.tif_dir.glob(arguments.tif_pattern))
    if not tif_paths:
        raise FileNotFoundError(
            f"No TIFF files found in {arguments.tif_dir} with pattern {arguments.tif_pattern}"
        )

    print(f"Loaded MRC volume shape={mrc_volume.shape}, dtype={mrc_volume.dtype}")
    print(f"Found {len(tif_paths)} TIFF file(s) to align")

    alignment_results: list[AlignmentResult] = []
    for tif_path in tif_paths:
        print(f"Aligning {tif_path.name} ...")
        result = align_single_tif(
            mrc_volume=mrc_volume,
            tif_path=tif_path,
            max_z_candidates=arguments.max_z_candidates,
            refine_radius_z=arguments.refine_radius_z,
            refine_radius_xy=arguments.refine_radius_xy,
            ncc_stride=max(1, arguments.ncc_stride),
        )
        alignment_results.append(result)
        print(
            f"  -> z={result.z_offset}, y={result.y_offset}, x={result.x_offset}, score={result.score:.5f}"
        )

    write_offsets_csv(alignment_results, arguments.offsets_csv)
    print(f"Offsets saved to: {arguments.offsets_csv}")

    if arguments.merged_tif is not None:
        merge_tifs_into_volume(
            results=alignment_results,
            output_volume_shape=mrc_volume.shape,
            output_tif_path=arguments.merged_tif,
        )
        print(f"Merged volume written to: {arguments.merged_tif}")

    if arguments.export_mrc_tif is not None:
        arguments.export_mrc_tif.parent.mkdir(parents=True, exist_ok=True)
        tifffile.imwrite(arguments.export_mrc_tif, mrc_volume, imagej=True)
        print(f"Full MRC exported as TIFF stack: {arguments.export_mrc_tif}")


if __name__ == "__main__":
    main()
