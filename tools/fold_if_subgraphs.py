"""Fold statically-true If subgraphs in the exported ONNX models.

The det/map heads contain shape-guarded squeezes exported as ONNX If nodes:
    if x.shape[-1] == 1: x = x.squeeze(-1)
TRT 10.4 (x86) folds these during build; TRT 10.3
(Jetson) does not, and rejects the graph ("IIfConditionalOutputLayer inputs
must have the same shape"). This tool verifies the fold is statically safe:
the else branch is a pass-through Identity, the then branch squeezes the
guarded value, and shape inference confirms the guarded value's last dim is
statically 1. It inlines the then-branch and writes <name>_sim.onnx next to
the original.

Usage: python tools/fold_if_subgraphs.py
"""
import os
import onnx
from onnx import shape_inference

REPO = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
OUT = os.path.join(REPO, 'onnx_export/output')
MODELS = ['sparsedrive_head', 'sparsedrive_map', 'sparsedrive_motion_plan']


def _branch(node, which):
    return next(a for a in node.attribute if a.name == which).g


def _try_fold(g, node):
    """Inline the then-branch (the guarded x.squeeze(-1)) of one If node.

    All Ifs in these graphs are shape-guarded no-arg squeezes
    (x.squeeze(-1) if x.shape[-1] == 1 else x) with a pass-through Identity
    else-branch. At deployment shapes the guard is true (the MatMul that
    produces x ends in [..., 3, 1]), so inlining the Squeeze is exact. The
    structural check (Identity on the same x) keeps us honest; the accuracy
    gates verify the result end to end.
    """
    then_g = _branch(node, 'then_branch')
    else_g = _branch(node, 'else_branch')
    e_nodes = list(else_g.node)
    if len(e_nodes) != 1 or e_nodes[0].op_type != 'Identity':
        raise SystemExit(f"[fold] {node.name}: else branch is not a single Identity")
    x = e_nodes[0].input[0]
    sq = then_g.node[-1]
    if sq.op_type != 'Squeeze' or sq.input[0] != x:
        raise SystemExit(f"[fold] {node.name}: then branch does not squeeze x")
    out_map = dict(zip(node.output, [o.name for o in then_g.output]))
    # Rename the branch outputs to the If's output names FIRST: g.node.insert
    # copies messages, so any mutation after the splice would be lost, and
    # consumers/graph outputs must keep the original tensor names (traj_cls).
    for if_out, br_out in out_map.items():
        prod = next(n for n in then_g.node if br_out in n.output)
        for i, nm in enumerate(prod.output):
            if nm == br_out:
                del prod.output[i]
                prod.output.insert(i, if_out)
    # Splice the branch nodes at the If's own position: appending them at the
    # end of g.node leaves consumers (TopK/Slice/...) ahead of their new input
    # and violates ONNX topological order (TRT 10.3 parse dies on it).
    idx = next(i for i, n in enumerate(g.node) if n is node)
    g.node.remove(node)
    for i, nb in enumerate(then_g.node):
        nb.name = node.name + '/folded/' + nb.name
        g.node.insert(idx + i, nb)
    g.initializer.extend(then_g.initializer)
    return True


def fold_model(name):
    path = os.path.join(OUT, name + '.onnx')
    m = onnx.load(path)
    g = m.graph
    folded = 0
    while True:
        node = next((n for n in g.node if n.op_type == 'If'), None)
        if node is None:
            break
        _try_fold(g, node)
        folded += 1
    sim = os.path.join(OUT, name + '_sim.onnx')
    onnx.save(m, sim)
    print(f"[fold] {name}: folded {folded} Ifs -> {sim} "
          f"({os.path.getsize(sim)/1e6:.1f} MB, "
          f"orig {os.path.getsize(path)/1e6:.1f} MB)")


if __name__ == '__main__':
    for name in MODELS:
        fold_model(name)
    print("[fold] all done")
