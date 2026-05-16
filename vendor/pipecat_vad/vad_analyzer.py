#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#
# Vendored from pipecat-ai/pipecat v1.2.1 (f8caa0da2b4c)
# Source: src/pipecat/audio/vad/vad_analyzer.py
# Vendor rung: 1 (pinned copy + MANIFEST; no local patches)
# See mod3/vendor/MANIFEST.toml for provenance record.
#

"""Voice Activity Detection (VAD) analyzer base classes and utilities.

This module provides the abstract base class for VAD analyzers and associated
data structures for voice activity detection in audio streams. Includes state
management, parameter configuration, and audio analysis framework.
"""

import asyncio
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor
from enum import Enum

from loguru import logger
from pydantic import BaseModel

VAD_CONFIDENCE = 0.7
VAD_START_SECS = 0.2
VAD_STOP_SECS = 0.2
VAD_MIN_VOLUME = 0.6


def calculate_audio_volume(audio: bytes, sample_rate: int) -> float:
    """Calculate RMS volume of audio bytes (int16 PCM)."""
    import numpy as np

    if len(audio) < 2:
        return 0.0
    samples = np.frombuffer(audio, dtype=np.int16).astype(np.float32) / 32768.0
    rms = float(np.sqrt(np.mean(samples**2))) if len(samples) > 0 else 0.0
    return rms


def exp_smoothing(value: float, prev: float, factor: float) -> float:
    """Exponential smoothing: factor * value + (1 - factor) * prev."""
    return factor * value + (1.0 - factor) * prev


class VADState(Enum):
    """Voice Activity Detection states."""

    QUIET = 1
    STARTING = 2
    SPEAKING = 3
    STOPPING = 4


class VADParams(BaseModel):
    """Configuration parameters for Voice Activity Detection."""

    confidence: float = VAD_CONFIDENCE
    start_secs: float = VAD_START_SECS
    stop_secs: float = VAD_STOP_SECS
    min_volume: float = VAD_MIN_VOLUME


class VADAnalyzer(ABC):
    """Abstract base class for Voice Activity Detection analyzers."""

    def __init__(self, *, sample_rate: int | None = None, params: VADParams | None = None):
        self._init_sample_rate = sample_rate
        self._sample_rate = 0
        self._params = params or VADParams()
        self._num_channels = 1
        self._vad_buffer = b""
        self._smoothing_factor = 0.2
        self._prev_volume = 0
        self._executor = ThreadPoolExecutor(max_workers=1)

    @property
    def sample_rate(self) -> int:
        return self._sample_rate

    @property
    def num_channels(self) -> int:
        return self._num_channels

    @property
    def params(self) -> VADParams:
        return self._params

    @abstractmethod
    def num_frames_required(self) -> int:
        pass

    @abstractmethod
    def voice_confidence(self, buffer: bytes) -> float:
        pass

    def set_sample_rate(self, sample_rate: int):
        self._sample_rate = self._init_sample_rate or sample_rate
        self.set_params(self._params)

    def set_params(self, params: VADParams):
        logger.debug(f"Setting VAD params to: {params}")
        self._params = params
        self._vad_frames = self.num_frames_required()
        self._vad_frames_num_bytes = self._vad_frames * self._num_channels * 2
        vad_frames_per_sec = self._vad_frames / self.sample_rate
        self._vad_start_frames = round(self._params.start_secs / vad_frames_per_sec)
        self._vad_stop_frames = round(self._params.stop_secs / vad_frames_per_sec)
        self._vad_starting_count = 0
        self._vad_stopping_count = 0
        self._vad_state: VADState = VADState.QUIET

    def _get_smoothed_volume(self, audio: bytes) -> float:
        volume = calculate_audio_volume(audio, self.sample_rate)
        return exp_smoothing(volume, self._prev_volume, self._smoothing_factor)

    async def analyze_audio(self, buffer: bytes) -> VADState:
        loop = asyncio.get_running_loop()
        state = await loop.run_in_executor(self._executor, self._run_analyzer, buffer)
        return state

    def _run_analyzer(self, buffer: bytes) -> VADState:
        self._vad_buffer += buffer
        num_required_bytes = self._vad_frames_num_bytes
        if len(self._vad_buffer) < num_required_bytes:
            return self._vad_state

        while len(self._vad_buffer) >= num_required_bytes:
            audio_frames = self._vad_buffer[:num_required_bytes]
            self._vad_buffer = self._vad_buffer[num_required_bytes:]
            confidence = self.voice_confidence(audio_frames)
            volume = self._get_smoothed_volume(audio_frames)
            self._prev_volume = volume
            speaking = confidence >= self._params.confidence and volume >= self._params.min_volume

            if speaking:
                match self._vad_state:
                    case VADState.QUIET:
                        self._vad_state = VADState.STARTING
                        self._vad_starting_count = 1
                    case VADState.STARTING:
                        self._vad_starting_count += 1
                    case VADState.STOPPING:
                        self._vad_state = VADState.SPEAKING
                        self._vad_stopping_count = 0
            else:
                match self._vad_state:
                    case VADState.STARTING:
                        self._vad_state = VADState.QUIET
                        self._vad_starting_count = 0
                    case VADState.SPEAKING:
                        self._vad_state = VADState.STOPPING
                        self._vad_stopping_count = 1
                    case VADState.STOPPING:
                        self._vad_stopping_count += 1

        if (
            self._vad_state == VADState.STARTING
            and self._vad_starting_count >= self._vad_start_frames
        ):
            self._vad_state = VADState.SPEAKING
            self._vad_starting_count = 0

        if (
            self._vad_state == VADState.STOPPING
            and self._vad_stopping_count >= self._vad_stop_frames
        ):
            self._vad_state = VADState.QUIET
            self._vad_stopping_count = 0

        return self._vad_state

    async def cleanup(self):
        pass
