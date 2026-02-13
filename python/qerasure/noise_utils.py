"""Python wrappers for qerasure noise parameters."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Mapping, Union

from ._bindings import cpp

NoiseChannel = cpp.NoiseChannel

NOISE_CHANNELS: Dict[str, NoiseChannel] = {
    "p_single_qubit_depolarize": NoiseChannel.SINGLE_QUBIT_DEPOLARIZE,
    "p_two_qubit_depolarize": NoiseChannel.TWO_QUBIT_DEPOLARIZE,
    "p_measurement_error": NoiseChannel.MEASUREMENT_ERROR,
    "p_single_qubit_erasure": NoiseChannel.SINGLE_QUBIT_ERASURE,
    "p_two_qubit_erasure": NoiseChannel.TWO_QUBIT_ERASURE,
    "p_erasure_check_error": NoiseChannel.ERASURE_CHECK_ERROR,
}


def _resolve_channel(channel: Union[str, NoiseChannel]) -> Union[str, NoiseChannel]:
    if isinstance(channel, str) and channel in NOISE_CHANNELS:
        return NOISE_CHANNELS[channel]
    return channel


@dataclass
class NoiseParams:
    """A lightweight Python facade over the C++ NoiseParams type."""

    _cpp_noise_params: object

    def __init__(self, **kwargs: float):
        self._cpp_noise_params = cpp.NoiseParams()
        for channel, probability in kwargs.items():
            self.set(channel, probability)

    def _to_cpp_noise_params(self):
        return self._cpp_noise_params

    def set(self, channel: Union[str, NoiseChannel], probability: float) -> None:
        self._cpp_noise_params.set(_resolve_channel(channel), probability)

    def get(self, channel: Union[str, NoiseChannel]) -> float:
        return float(self._cpp_noise_params.get(_resolve_channel(channel)))

    def update(self, values: Mapping[Union[str, NoiseChannel], float]) -> None:
        for channel, probability in values.items():
            self.set(channel, probability)

    def to_dict(self) -> dict[str, float]:
        return {key: self.get(key) for key in NOISE_CHANNELS}

    def __repr__(self) -> str:
        return repr(self._cpp_noise_params)
