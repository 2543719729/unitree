from __future__ import annotations

from dataclasses import MISSING

import numpy as np

from isaaclab.utils import configclass

from .terrain_generator import TerrainGenerator
from .terrain_generator_cfg import TerrainGeneratorCfg


@configclass
class RowTerrainMixCfg:
    start_row: int = 0
    end_row: int = 0
    proportions: dict[str, float] = MISSING


class RowCurriculumTerrainGenerator(TerrainGenerator):
    def __init__(self, cfg: TerrainGeneratorCfg, device: str = "cpu"):
        super().__init__(cfg=cfg, device=device)

    def _generate_curriculum_terrains(self):
        if not hasattr(self.cfg, "row_mixes") or self.cfg.row_mixes is None:
            return super()._generate_curriculum_terrains()

        sub_terrain_names = list(self.cfg.sub_terrains.keys())
        sub_terrains_cfgs = list(self.cfg.sub_terrains.values())
        name_to_index = {name: i for i, name in enumerate(sub_terrain_names)}

        # validate row mixes cover all rows
        for sub_row in range(self.cfg.num_rows):
            _ = self._resolve_row_mix(sub_row, name_to_index)

        for sub_row in range(self.cfg.num_rows):
            mix_proportions = self._resolve_row_mix(sub_row, name_to_index)
            sub_indices_for_cols = self._indices_from_proportions(mix_proportions, self.cfg.num_cols)
            self.np_rng.shuffle(sub_indices_for_cols)

            lower, upper = self.cfg.difficulty_range
            difficulty = (sub_row + self.np_rng.uniform()) / self.cfg.num_rows
            difficulty = lower + (upper - lower) * difficulty

            for sub_col in range(self.cfg.num_cols):
                sub_index = int(sub_indices_for_cols[sub_col])
                mesh, origin = self._get_terrain_mesh(float(difficulty), sub_terrains_cfgs[sub_index])
                self._add_sub_terrain(mesh, origin, sub_row, sub_col, sub_terrains_cfgs[sub_index])

    def _resolve_row_mix(self, sub_row: int, name_to_index: dict[str, int]) -> np.ndarray:
        matches = [m for m in self.cfg.row_mixes if m.start_row <= sub_row <= m.end_row]
        if len(matches) != 1:
            raise ValueError(
                f"Expected exactly one row mix for row {sub_row}, but found {len(matches)}. "
                "Please ensure row_mixes covers every row without overlaps."
            )

        mix = matches[0]
        if not isinstance(mix.proportions, dict) or len(mix.proportions) == 0:
            raise ValueError(f"Row mix for rows [{mix.start_row}, {mix.end_row}] has empty proportions.")

        proportions = np.zeros(len(name_to_index), dtype=np.float64)
        for name, weight in mix.proportions.items():
            if name not in name_to_index:
                raise ValueError(
                    f"Row mix references unknown sub-terrain '{name}'. Available: {list(name_to_index.keys())}."
                )
            proportions[name_to_index[name]] = float(weight)

        total = float(np.sum(proportions))
        if total <= 0.0:
            raise ValueError(f"Row mix for rows [{mix.start_row}, {mix.end_row}] has non-positive total proportion.")
        proportions /= total
        return proportions

    @staticmethod
    def _indices_from_proportions(proportions: np.ndarray, num_cols: int) -> np.ndarray:
        if num_cols <= 0:
            raise ValueError("num_cols must be positive.")
        if proportions.ndim != 1:
            raise ValueError("proportions must be a 1D array.")

        raw_counts = proportions * float(num_cols)
        counts = np.floor(raw_counts).astype(np.int32)
        remainder = int(num_cols - int(np.sum(counts)))

        if remainder > 0:
            residuals = raw_counts - counts
            add_order = np.argsort(-residuals)
            for i in range(remainder):
                counts[int(add_order[i % len(add_order)])] += 1
        elif remainder < 0:
            residuals = raw_counts - counts
            sub_order = np.argsort(residuals)
            for i in range(-remainder):
                idx = int(sub_order[i % len(sub_order)])
                if counts[idx] > 0:
                    counts[idx] -= 1

        indices = []
        for i, c in enumerate(counts.tolist()):
            indices.extend([i] * int(c))

        if len(indices) != num_cols:
            raise RuntimeError(f"Internal error: expected {num_cols} column indices, got {len(indices)}")
        return np.asarray(indices, dtype=np.int32)


@configclass
class RowCurriculumTerrainGeneratorCfg(TerrainGeneratorCfg):
    class_type: type = RowCurriculumTerrainGenerator
    row_mixes: list[RowTerrainMixCfg] = MISSING
