"""FinSage v2 — advanced research extensions.

All v2 features are additive and gated by feature flags in
``configs/v2_features.yaml``.  When all flags are ``false`` the
existing pipeline is completely unaffected.
"""

__version__ = "2.0.0"
__all__: list[str] = []  # populated by each stage
