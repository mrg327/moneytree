"""
Pipeline orchestration modules for MoneyTree.

This package provides optimized pipeline processing with audio-first workflow,
early duration estimation, and smart media trimming to prevent resource waste.
"""

from .media_controller import MediaController, MediaConfig

__all__ = [
    'MediaController',
    'MediaConfig'
]