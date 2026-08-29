"""Local process-start compatibility for the fixed Megatron-Core runtime.

The installed nvidia-resiliency-ext exposes the async-checkpoint API used by
Megatron-Core but omits its package-level version attribute.  Megatron-Core
uses that attribute only to gate the optional API.  Supply the minimum
compatible version before Megatron-Core imports it; this does not alter the
installed package or enable any extra dependency.
"""

try:
    import nvidia_resiliency_ext as _nvrx
except ImportError:
    pass
else:
    if not hasattr(_nvrx, "__version__"):
        _nvrx.__version__ = "0.6.0"
