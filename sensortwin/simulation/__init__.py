"""Synthetic physics-inspired multichannel sensor simulator (project Layer 1).

The generator composes each sample as:

    base_dynamics + load_profile + channel_coupling + event_signature + noise + artifacts

with a known event label and full parameter metadata. Generation is deterministic given a seed.
See :mod:`sensortwin.simulation.generator` for the entry points.
"""

from sensortwin.simulation.events import CHANNELS, EVENT_CLASSES, EventClass
from sensortwin.simulation.generator import GenConfig, generate_dataset, generate_sample

__all__ = [
    "CHANNELS",
    "EVENT_CLASSES",
    "EventClass",
    "GenConfig",
    "generate_dataset",
    "generate_sample",
]
