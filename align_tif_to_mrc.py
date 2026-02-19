#!/usr/bin/env python3
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import mrcfile
import numpy as np
import tifffile


MRC_PATH = Path("dataset/Annotations/20161129_SURFACTANTE_TOMO_09_norm_3DS30_bin2.mrc")
TIF_DIR = Path("dataset/Annotations")
TIF_PATTERN = "*.tif"
MAX_Z_CANDIDATES = 3
REFINE_RADIUS_Z = 3
REFINE_RADIUS_XY = 8
NCC_STRIDE = 8
OUTPUT_MERGED_TIF = Path("output/merged_from_annotations.tif")
OUTPUT_MERGED_CLASSES_TIF = Path("output/merged_from_annotations_classes.tif")


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


def merge_tifs_into_volume(
    results: list[AlignmentResult],
    output_volume_shape: tuple[int, int, int],
    output_tif_path: Path,
) -> np.ndarray:
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
    return merged


def merge_tifs_into_class_volume(
    results: list[AlignmentResult],
    merged_raw_volume: np.ndarray,
    output_volume_shape: tuple[int, int, int],
    output_tif_path: Path,
) -> None:
    merged_class_ids = np.zeros(output_volume_shape, dtype=np.uint16)
    merged_class_scores = np.full(output_volume_shape, -np.inf, dtype=np.float32)

    for class_id, result in enumerate(results, start=1):
        tif_volume = read_volume_any(result.tif_path)
        tif_depth, tif_height, tif_width = tif_volume.shape
        z_start, y_start, x_start = result.z_offset, result.y_offset, result.x_offset
        z_end = min(output_volume_shape[0], z_start + tif_depth)
        y_end = min(output_volume_shape[1], y_start + tif_height)
        x_end = min(output_volume_shape[2], x_start + tif_width)

        if z_end <= z_start or y_end <= y_start or x_end <= x_start:
            continue

        tif_crop = tif_volume[: z_end - z_start, : y_end - y_start, : x_end - x_start]
        tif_min = float(tif_crop.min())
        tif_max = float(tif_crop.max())
        if tif_max <= tif_min:
            continue
        threshold = 0.5 * (tif_min + tif_max)
        class_mask = tif_crop >= threshold

        target_ids = merged_class_ids[z_start:z_end, y_start:y_end, x_start:x_end]
        target_scores = merged_class_scores[z_start:z_end, y_start:y_end, x_start:x_end]
        update_mask = class_mask & (tif_crop > target_scores)
        target_ids[update_mask] = class_id
        target_scores[update_mask] = tif_crop[update_mask]

    raw_visual = normalize_for_matching(merged_raw_volume) * 255.0
    class_visual = raw_visual.copy()

    num_classes = len(results)
    if num_classes > 0:
        for class_id in range(1, num_classes + 1):
            class_mask = merged_class_ids == class_id
            class_offset = 20.0 + (class_id - 1) * (100.0 / max(1, num_classes - 1))
            class_visual[class_mask] = np.clip(
                0.5 * raw_visual[class_mask] + 128.0 + class_offset,
                0.0,
                255.0,
            )

    output_tif_path.parent.mkdir(parents=True, exist_ok=True)
    tifffile.imwrite(output_tif_path, class_visual.astype(np.uint8), imagej=True)


def main() -> None:
    mrc_volume = read_volume_any(MRC_PATH)

    excluded_names = {
        OUTPUT_MERGED_TIF.name,
        OUTPUT_MERGED_CLASSES_TIF.name,
    }
    tif_paths = sorted(
        path for path in TIF_DIR.glob(TIF_PATTERN) if path.name not in excluded_names
    )
    if not tif_paths:
        raise FileNotFoundError(
            f"No TIFF files found in {TIF_DIR} with pattern {TIF_PATTERN}"
        )

    print(f"Loaded MRC volume shape={mrc_volume.shape}, dtype={mrc_volume.dtype}")
    print(f"Found {len(tif_paths)} TIFF file(s) to align")

    alignment_results: list[AlignmentResult] = []
    for tif_path in tif_paths:
        print(f"Aligning {tif_path.name} ...")
        result = align_single_tif(
            mrc_volume=mrc_volume,
            tif_path=tif_path,
            max_z_candidates=MAX_Z_CANDIDATES,
            refine_radius_z=REFINE_RADIUS_Z,
            refine_radius_xy=REFINE_RADIUS_XY,
            ncc_stride=max(1, NCC_STRIDE),
        )
        alignment_results.append(result)
        print(
            f"  -> z={result.z_offset}, y={result.y_offset}, x={result.x_offset}, score={result.score:.5f}"
        )

    merged_volume = merge_tifs_into_volume(
        results=alignment_results,
        output_volume_shape=mrc_volume.shape,
        output_tif_path=OUTPUT_MERGED_TIF,
    )
    print(f"Merged volume written to: {OUTPUT_MERGED_TIF}")

    merge_tifs_into_class_volume(
        results=alignment_results,
        merged_raw_volume=merged_volume,
        output_volume_shape=mrc_volume.shape,
        output_tif_path=OUTPUT_MERGED_CLASSES_TIF,
    )
    print(f"Merged class volume written to: {OUTPUT_MERGED_CLASSES_TIF}")


if __name__ == "__main__":
    main()
