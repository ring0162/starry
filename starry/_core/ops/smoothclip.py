# -*- coding: utf-8 -*-
"""
Smooth clipping of the occultor's chord at the stellar limb, as a self-contained Theano op.

    smooth_clip(a, d) = a - R(a - 1, d) + R(-1 - a, d),   a clipped to [-2, 2] first,
    R(z, d) = E[relu(z + t)], t ~ triangular on [-d, d]
            = [relu(z + d)^3 - 2 relu(z)^3 + relu(z - d)^3] / (6 d^2)

i.e. the hard clip of `a` to [-1, 1] averaged with a triangular window of half-width `d`: a C2 cubic spline
that never goes beyond the hard clip. It is an op of its own (NumPy perform, analytic gradient) rather than an
expression graph because Theano's optimiser hung while compiling the gradient of the equivalent expression
inside the DopplerMap occultation graph.
"""
from ...compat import Apply, Op, tt
import numpy as np

__all__ = ["smooth_clip"]


def _c2(v):
    v = np.maximum(v, 0.0)
    return v * v


def _c3(v):
    v = np.maximum(v, 0.0)
    return v * v * v


def _R(z, d):
    return (_c3(z + d) - 2.0 * _c3(z) + _c3(z - d)) / (6.0 * d * d)


def _R_z(z, d):
    return (_c2(z + d) - 2.0 * _c2(z) + _c2(z - d)) / (2.0 * d * d)


def _R_d(z, d):
    return (_c2(z + d) - _c2(z - d)) / (2.0 * d * d) - 2.0 * _R(z, d) / d


def smooth_clip_np(a, d):
    a = np.clip(a, -2.0, 2.0)
    return a - _R(a - 1.0, d) + _R(-1.0 - a, d)


def smooth_clip_grad_np(a, d):
    """Partial derivatives of smooth_clip_np with respect to a and d (same shape as a)."""
    inside = (a > -2.0) & (a < 2.0)
    ac = np.clip(a, -2.0, 2.0)
    da = (1.0 - _R_z(ac - 1.0, d) - _R_z(-1.0 - ac, d)) * inside
    dd = -_R_d(ac - 1.0, d) + _R_d(-1.0 - ac, d)
    return da, dd


class SmoothClipGradOp(Op):
    __props__ = ()

    def make_node(self, a, d):
        a = tt.as_tensor_variable(a)
        d = tt.as_tensor_variable(d)
        return Apply(self, [a, d], [a.type(), a.type()])

    def infer_shape(self, *args):
        shapes = args[-1]
        return [shapes[0], shapes[0]]

    def perform(self, node, inputs, outputs):
        a, d = inputs
        d = np.asarray(d, dtype=np.float64).reshape((1,) * (np.ndim(a) - np.ndim(d)) + np.shape(d))
        da, dd = smooth_clip_grad_np(np.asarray(a, dtype=np.float64), d)
        outputs[0][0] = np.asarray(np.broadcast_to(da, a.shape), dtype=node.outputs[0].dtype)
        outputs[1][0] = np.asarray(np.broadcast_to(dd, a.shape), dtype=node.outputs[1].dtype)


class SmoothClipOp(Op):
    __props__ = ()

    def make_node(self, a, d):
        a = tt.as_tensor_variable(a)
        d = tt.as_tensor_variable(d)
        return Apply(self, [a, d], [a.type()])

    def infer_shape(self, *args):
        shapes = args[-1]
        return [shapes[0]]

    def perform(self, node, inputs, outputs):
        a, d = inputs
        d = np.asarray(d, dtype=np.float64).reshape((1,) * (np.ndim(a) - np.ndim(d)) + np.shape(d))
        out = smooth_clip_np(np.asarray(a, dtype=np.float64), d)
        outputs[0][0] = np.asarray(np.broadcast_to(out, a.shape), dtype=node.outputs[0].dtype)

    def grad(self, inputs, gradients):
        a, d = inputs
        g = gradients[0]
        da, dd = SmoothClipGradOp()(a, d)
        gd = g * dd
        # d has fewer dimensions than a (e.g. one width per kernel tap, (nk,), against a (nt, nk)):
        # sum the gradient over a's leading axes
        lead = a.ndim - d.ndim
        if lead > 0:
            gd = tt.sum(gd, axis=list(range(lead)))
        return [g * da, gd]


_smooth_clip_op = SmoothClipOp()


def smooth_clip(a, d):
    """Smooth clip of `a` to [-1, 1]. `d` (the window half-width) has a's trailing shape, e.g. a (nt, nk), d (nk,)."""
    return _smooth_clip_op(a, d)
