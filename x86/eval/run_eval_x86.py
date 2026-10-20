import sys, os, types, numpy

# numpy._core compatibility shim for pickle
if not hasattr(numpy, '_core'):
    nc = types.ModuleType('numpy._core')
    nc.__path__ = []
    sys.modules['numpy._core'] = nc
    for sub in ['multiarray', 'umath', 'numerictypes']:
        old = getattr(numpy.core, sub, None)
        if old:
            setattr(nc, sub, old)
            sys.modules[f'numpy._core.{sub}'] = old

# Run eval script
script = os.path.join(
    os.path.dirname(__file__),
    'eval_trt_vs_pt_nds.py'
)
exec(open(script).read())
