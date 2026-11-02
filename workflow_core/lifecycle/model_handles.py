"""Compatibility surface for physical lifecycle model publication.

Canonical logic-facing model identities live in
``workflow_core.lifecycle.artifact_handles``.  The implementation reexported
here is artifact-owned; this module remains only while existing providers and
public imports migrate to that deep module.
"""

import workflow_artifacts.lifecycle.model_publication as _model_publication

HistoricalModelResolver = _model_publication.HistoricalModelResolver
ModelArtifact = _model_publication.ModelArtifact
PublishedModelHandle = _model_publication.PublishedModelHandle
publish_model_handles = _model_publication.publish_model_handles
publish_model_alias = _model_publication.publish_model_alias
resolve_model_handles = _model_publication.resolve_model_handles

# Preserve the existing test patch point during the bounded compatibility
# window.  Both names refer to the same standard-library module object used by
# the artifact-owned implementation.
os = _model_publication.os

__all__ = [
    "HistoricalModelResolver",
    "ModelArtifact",
    "PublishedModelHandle",
    "publish_model_alias",
    "publish_model_handles",
    "resolve_model_handles",
]
