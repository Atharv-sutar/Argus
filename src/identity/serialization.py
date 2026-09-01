"""Serialization utilities for Identity and Embedding objects."""

import base64
import numpy as np
from typing import Optional, Dict, Any

from src.core.types import Embedding, Identity, TargetIdentityAnchor, ViewCluster

def _serialize_emb(emb: Embedding) -> Dict[str, Any]:
    return {
        "vector_b64": base64.b64encode(emb.vector.tobytes()).decode("ascii"),
        "dim": emb.dim,
        "model_name": emb.model_name,
        "version": emb.version,
        "crop_type": emb.crop_type,
        "quality_score": emb.quality_score,
        "camera_id": emb.camera_id,
        "timestamp_ms": emb.timestamp_ms
    }

def _deserialize_emb(data: Dict[str, Any]) -> Embedding:
    vector = np.frombuffer(base64.b64decode(data["vector_b64"]), dtype=np.float32)
    return Embedding(
        vector=vector,
        model_name=data.get("model_name", "reid"),
        version=data.get("version", "2.0"),
        crop_type=data.get("crop_type", "full"),
        quality_score=data.get("quality_score", 1.0),
        camera_id=data.get("camera_id", "camera_0"),
        timestamp_ms=data.get("timestamp_ms", 0.0)
    )

def _serialize_cluster(c: ViewCluster) -> Dict[str, Any]:
    return {
        "cluster_id": c.cluster_id,
        "label": c.label,
        "exemplars": [_serialize_emb(e) for e in c.exemplars],
        "centroid": _serialize_emb(c.centroid) if c.centroid else None
    }

def _deserialize_cluster(data: Dict[str, Any]) -> ViewCluster:
    c = ViewCluster(
        cluster_id=data["cluster_id"],
        label=data.get("label", "general"),
        exemplars=[_deserialize_emb(e) for e in data.get("exemplars", [])]
    )
    if data.get("centroid"):
        c.centroid = _deserialize_emb(data["centroid"])
    return c

def _serialize_anchor(a: TargetIdentityAnchor) -> Dict[str, Any]:
    return {
        "identity_id": a.identity_id,
        "label": a.label,
        "clusters": [_serialize_cluster(c) for c in a.clusters],
        "model_name": a.model_name,
        "feature_dim": a.feature_dim,
        "created_timestamp_ms": a.created_timestamp_ms,
        "anchor_hash": a.anchor_hash
    }

def _deserialize_anchor(data: Dict[str, Any]) -> TargetIdentityAnchor:
    return TargetIdentityAnchor(
        identity_id=data["identity_id"],
        label=data.get("label", "selected_target"),
        clusters=[_deserialize_cluster(c) for c in data.get("clusters", [])],
        model_name=data.get("model_name", "osnet_x0_25"),
        feature_dim=data.get("feature_dim", 512),
        created_timestamp_ms=data.get("created_timestamp_ms", 0.0),
        anchor_hash=data.get("anchor_hash", "")
    )

def serialize_identity(ident: Identity) -> Dict[str, Any]:
    return {
        "identity_id": ident.identity_id,
        "label": ident.label,
        "trusted_gallery": [_serialize_emb(e) for e in ident.trusted_gallery],
        "provisional_gallery": [_serialize_emb(e) for e in ident.provisional_gallery],
        "rejected_gallery": [_serialize_emb(e) for e in ident.rejected_gallery],
        "trusted_upper_gallery": [_serialize_emb(e) for e in ident.trusted_upper_gallery],
        "trusted_lower_gallery": [_serialize_emb(e) for e in ident.trusted_lower_gallery],
        "anchor": _serialize_anchor(ident.anchor) if ident.anchor else None,
        "view_clusters": [_serialize_cluster(c) for c in ident.view_clusters],
        "trusted_prototype": _serialize_emb(ident.trusted_prototype) if ident.trusted_prototype else None,
        "trusted_upper_proto": _serialize_emb(ident.trusted_upper_proto) if ident.trusted_upper_proto else None,
        "trusted_lower_proto": _serialize_emb(ident.trusted_lower_proto) if ident.trusted_lower_proto else None,
        "confidence": ident.confidence,
        "last_seen_timestamp_ms": ident.last_seen_timestamp_ms,
        "last_camera_id": ident.last_camera_id
    }

def deserialize_identity(data: Dict[str, Any]) -> Identity:
    ident = Identity(
        identity_id=data["identity_id"],
        label=data.get("label", "target_0")
    )
    ident.trusted_gallery = [_deserialize_emb(e) for e in data.get("trusted_gallery", [])]
    ident.provisional_gallery = [_deserialize_emb(e) for e in data.get("provisional_gallery", [])]
    ident.rejected_gallery = [_deserialize_emb(e) for e in data.get("rejected_gallery", [])]
    ident.trusted_upper_gallery = [_deserialize_emb(e) for e in data.get("trusted_upper_gallery", [])]
    ident.trusted_lower_gallery = [_deserialize_emb(e) for e in data.get("trusted_lower_gallery", [])]
    
    if data.get("anchor"):
        ident.anchor = _deserialize_anchor(data["anchor"])
    
    ident.view_clusters = [_deserialize_cluster(c) for c in data.get("view_clusters", [])]
    
    if data.get("trusted_prototype"):
        ident.trusted_prototype = _deserialize_emb(data["trusted_prototype"])
    if data.get("trusted_upper_proto"):
        ident.trusted_upper_proto = _deserialize_emb(data["trusted_upper_proto"])
    if data.get("trusted_lower_proto"):
        ident.trusted_lower_proto = _deserialize_emb(data["trusted_lower_proto"])
        
    ident.confidence = data.get("confidence", 0.0)
    ident.last_seen_timestamp_ms = data.get("last_seen_timestamp_ms", 0.0)
    ident.last_camera_id = data.get("last_camera_id", "camera_0")
    
    return ident
