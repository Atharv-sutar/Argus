"""Backbone architectures for Person Re-Identification."""

from src.reid.backbones.osnet import osnet_x0_25, osnet_x1_0, build_osnet

__all__ = ["osnet_x0_25", "osnet_x1_0", "build_osnet"]
