"""Normalização de loudness (EBU R128) em duas passadas com o filtro loudnorm do FFmpeg.

Passada 1: mede o áudio já concatenado (saída nula, `print_format=json`).
Passada 2: o concat final aplica loudnorm com os valores medidos e `linear=true`,
que vira um ganho único quando possível (sem bombear o volume da fala).
"""

import json
import math
import re
from pathlib import Path
from typing import Literal, Mapping, Optional, Union

from pydantic import BaseModel, Field

from ai_video_editor.core.schema import LoudnessTarget

# Abaixo do gate absoluto da EBU R128 (-70 LUFS) não há conteúdo para medir.
SILENCE_GATE_LUFS = -70.0

# True peak de entrada a partir do qual a gravação provavelmente já saiu distorcida.
CLIPPING_TP_DBTP = 0.0

LoudnessStatus = Literal["applied", "skipped"]

_JSON_BLOCK = re.compile(r"\{[^{}]*\"input_i\"[^{}]*\}")


class LoudnessMeasurement(BaseModel):
    """Valores medidos pela passada 1 do loudnorm."""

    input_i: float = Field(..., description="Loudness integrado de entrada, em LUFS")
    input_tp: float = Field(..., description="True peak de entrada, em dBTP")
    input_lra: float = Field(..., description="Faixa de loudness de entrada, em LU")
    input_thresh: float = Field(..., description="Limiar do gate relativo, em LUFS")
    target_offset: float = Field(..., description="Ajuste fino calculado pelo loudnorm, em LU")

    @property
    def is_silent(self) -> bool:
        """Áudio sem conteúdo mensurável (todo em silêncio): -inf, NaN ou abaixo do gate absoluto."""
        values = (self.input_i, self.input_tp, self.input_lra, self.input_thresh, self.target_offset)
        return not all(math.isfinite(v) for v in values) or self.input_i <= SILENCE_GATE_LUFS


class LoudnessResult(BaseModel):
    """O que a normalização fez no render: alvo, medição e, se pulada, o motivo."""

    target: LoudnessTarget
    status: LoudnessStatus
    measured: Optional[LoudnessMeasurement] = Field(default=None, description="Só com medição válida")
    note: Optional[str] = Field(default=None, description="Motivo de ter sido pulada")
    measure_sec: Optional[float] = Field(default=None, description="Tempo da passada de medição")

    @property
    def warnings(self) -> list[str]:
        """Avisos para o usuário: normalização pulada ou gravação provavelmente distorcida."""
        if self.status == "skipped":
            return [f"normalização de volume pulada: {self.note}"]
        if self.measured is not None and self.measured.input_tp >= CLIPPING_TP_DBTP:
            return [
                f"o áudio original chega a {self.measured.input_tp:.1f} dBTP de true peak (≥ 0 dBTP): "
                "provável distorção na gravação, que a normalização não desfaz"
            ]
        return []

    def to_metrics(self) -> dict[str, Union[str, float]]:
        """Métricas planas (`loudness_*`) para o manifesto do projeto."""
        metrics: dict[str, Union[str, float]] = {
            "loudness_status": self.status,
            "loudness_preset": self.target.preset,
            "loudness_target_i": self.target.integrated_lufs,
            "loudness_target_tp": self.target.true_peak_dbtp,
            "loudness_target_lra": self.target.lra,
        }
        if self.measured is not None:
            metrics.update({f"loudness_{k}": v for k, v in self.measured.model_dump().items()})
        if self.note is not None:
            metrics["loudness_note"] = self.note
        if self.measure_sec is not None:
            metrics["loudness_measure_sec"] = self.measure_sec
        return metrics

    @classmethod
    def from_metrics(cls, metrics: Mapping[str, object]) -> Optional["LoudnessResult"]:
        """Inverso de `to_metrics`; None se o render não normalizou."""
        if "loudness_status" not in metrics:
            return None
        measured = None
        if "loudness_input_i" in metrics:
            measured = LoudnessMeasurement.model_validate(
                {k: metrics[f"loudness_{k}"] for k in LoudnessMeasurement.model_fields}
            )
        return cls(
            target=LoudnessTarget(
                preset=str(metrics["loudness_preset"]),
                integrated_lufs=metrics["loudness_target_i"],
                true_peak_dbtp=metrics["loudness_target_tp"],
                lra=metrics["loudness_target_lra"],
            ),
            status=metrics["loudness_status"],
            measured=measured,
            note=metrics.get("loudness_note"),
            measure_sec=metrics.get("loudness_measure_sec"),
        )


def _parse_number(value: object) -> float:
    try:
        return float(value)  # "-inf" e "inf" viram infinitos
    except (TypeError, ValueError):
        return math.nan


def parse_loudnorm_output(stderr: str) -> LoudnessMeasurement:
    """Extrai a medição do JSON que o loudnorm imprime no stderr (o último bloco, se houver mais de um).

    Raises:
        ValueError: nenhum bloco JSON do loudnorm na saída.
    """
    blocks = _JSON_BLOCK.findall(stderr)
    if not blocks:
        raise ValueError("o FFmpeg não imprimiu a medição do loudnorm")
    data = json.loads(blocks[-1])
    return LoudnessMeasurement(
        input_i=_parse_number(data.get("input_i")),
        input_tp=_parse_number(data.get("input_tp")),
        input_lra=_parse_number(data.get("input_lra")),
        input_thresh=_parse_number(data.get("input_thresh")),
        target_offset=_parse_number(data.get("target_offset")),
    )


def _target_args(target: LoudnessTarget) -> str:
    return f"I={target.integrated_lufs:g}:TP={target.true_peak_dbtp:g}:LRA={target.lra:g}"


def measure_filter(target: LoudnessTarget) -> str:
    """Filtro da passada 1."""
    return f"loudnorm={_target_args(target)}:print_format=json"


def apply_filter(target: LoudnessTarget, measured: LoudnessMeasurement) -> str:
    """Filtro da passada 2, com os valores medidos e ganho linear."""
    return (
        f"loudnorm={_target_args(target)}"
        f":measured_I={measured.input_i:.2f}"
        f":measured_TP={measured.input_tp:.2f}"
        f":measured_LRA={measured.input_lra:.2f}"
        f":measured_thresh={measured.input_thresh:.2f}"
        f":offset={measured.target_offset:.2f}"
        f":linear=true:print_format=none"
    )


def measure_command(ffmpeg_bin: Path, concat_list_path: Path, target: LoudnessTarget) -> list[str]:
    """Passada 1: lê só o áudio dos trechos concatenados e descarta a saída."""
    return [
        str(ffmpeg_bin),
        "-nostdin",
        "-hide_banner",
        "-nostats",
        "-loglevel",
        "info",  # o loudnorm imprime a medição em nível info
        "-f",
        "concat",
        "-safe",
        "0",
        "-i",
        str(concat_list_path),
        "-map",
        "0:a:0",
        "-af",
        measure_filter(target),
        "-f",
        "null",
        "-",
    ]
