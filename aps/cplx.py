#!/usr/bin/env python
"""
Provide some simple operations on the complex-valued tensor (that was used by APS)
Will be removed when PyTorch supports complex matrices well
Reference the code in:
https://github.com/kamo-naoyuki/pytorch_complex
"""

import torch as th
import numpy as np

from numbers import Number
from typing import Optional, Union

OpObjType = Union[th.Tensor, Number, "ComplexTensor"]


class ComplexTensor(object):
    """
    Complex-valued tensor class
    """

    def __init__(self,
                 real: th.Tensor,
                 imag: Optional[th.Tensor] = None,
                 polar: bool = False) -> None:
        imag = th.zeros_like(real) if imag is None else imag
        if polar:
            self.real = th.cos(imag) * real
            self.imag = th.sin(imag) * real
        else:
            self.real = real
            self.imag = imag

    def __add__(self, other: OpObjType) -> "ComplexTensor":
        """
        self + other
        """
        return _add(self, other)

    def __radd__(self, other: OpObjType) -> "ComplexTensor":
        """
        other + self
        """
        return _add(self, other)

    def __sub__(self, other: OpObjType) -> "ComplexTensor":
        """
        self - other
        """
        return _lsub(self, other)

    def __rsub__(self, other: OpObjType) -> "ComplexTensor":
        """
        other - self
        """
        return _rsub(other, self)

    def __mul__(self, other: OpObjType) -> "ComplexTensor":
        """
        self * other
        """
        return _mul(self, other)

    def __rmul__(self, other: OpObjType) -> "ComplexTensor":
        """
        other * self
        """
        return _mul(self, other)

    def __truediv__(self, other: OpObjType) -> "ComplexTensor":
        """
        self / other
        """
        return _ldiv(self, other)

    def __rtruediv__(self, other: OpObjType) -> "ComplexTensor":
        """
        other / self
        """
        return _rdiv(other, self)

    def __matmul__(self, other: OpObjType) -> "ComplexTensor":
        """
        self @ other
        """
        return _lmatmul(self, other)

    def __rmatmul__(self, other: OpObjType) -> "ComplexTensor":
        """
        other @ self
        """
        return _rmatmul(other, self)

    def abs(self) -> th.Tensor:
        """
        |self|
        """
        return (self.real**2 + self.imag**2).sqrt()

    def angle(self) -> th.Tensor:
        """
        \\angle{self}
        """
        return th.atan2(self.imag, self.real)

    def inverse(self) -> "ComplexTensor":
        """
        {self}^{-1}
        """
        return _inverse(self)

    def conj(self) -> "ComplexTensor":
        """
        {self}^*
        """
        return ComplexTensor(self.real, -1.0 * self.imag)

    def transpose(self, dim0, dim1) -> "ComplexTensor":
        """
        {self}^T
        """
        return ComplexTensor(self.real.transpose(dim0, dim1),
                             self.imag.transpose(dim0, dim1))

    def conj_transpose(self, dim0, dim1) -> "ComplexTensor":
        """
        {self}^H
        """
        return self.transpose(dim0, dim1).conj()

    def sum(self,
            dim: Optional[int] = None,
            keepdim: bool = False) -> "ComplexTensor":
        return ComplexTensor(self.real.sum(dim=dim, keepdim=keepdim),
                             self.imag.sum(dim=dim, keepdim=keepdim))

    def view(self, *shape) -> "ComplexTensor":
        return ComplexTensor(self.real.view(*shape), self.imag.view(*shape))

    def to(self, *args, **kwargs) -> "ComplexTensor":
        return ComplexTensor(self.real.to(*args, **kwargs),
                             self.imag.to(*args, **kwargs))

    def cpu(self) -> "ComplexTensor":
        return ComplexTensor(self.real.cpu(), self.imag.cpu())

    def cuda(self) -> "ComplexTensor":
        return ComplexTensor(self.real.cuda(), self.imag.cuda())

    def dim(self) -> int:
        return self.real.dim()

    @property
    def shape(self) -> th.Size:
        return self.real.shape

    @property
    def device(self) -> th.device:
        return self.real.device

    @property
    def dtype(self) -> th.dtype:
        return self.real.dtype

    def size(self) -> th.Size:
        return self.real.size()

    def masked_select(self, mask: th.Tensor) -> "ComplexTensor":
        return ComplexTensor(self.real.masked_select(mask),
                             self.imag.masked_select(mask))

    def masked_fill(self, mask: th.Tensor, value: Number) -> "ComplexTensor":
        return ComplexTensor(self.real.masked_fill(mask, value),
                             self.imag.masked_fill(mask, value))

    def contiguous(self) -> "ComplexTensor":
        return ComplexTensor(self.real.contiguous(), self.imag.contiguous())

    def __getitem__(self, item) -> "ComplexTensor":
        return ComplexTensor(self.real[item], self.imag[item])


def _is_complex(other: OpObjType) -> bool:
    return isinstance(other, (ComplexTensor, complex))


def _add(tensor: ComplexTensor, other: OpObjType) -> ComplexTensor:
    if _is_complex(other):
        return ComplexTensor(tensor.real + other.real, tensor.imag + other.imag)
    else:
        return ComplexTensor(tensor.real + other, tensor.imag)


def _lsub(tensor: ComplexTensor, other: OpObjType) -> ComplexTensor:
    if _is_complex(other):
        return ComplexTensor(tensor.real - other.real, tensor.imag - other.imag)
    else:
        return ComplexTensor(tensor.real - other, tensor.imag)


def _rsub(other: OpObjType, tensor: ComplexTensor) -> ComplexTensor:
    if _is_complex(other):
        return ComplexTensor(other.real - tensor.real, other.imag - tensor.imag)
    else:
        return ComplexTensor(other - tensor.real, -tensor.imag)


def _mul(tensor: ComplexTensor, other: OpObjType) -> ComplexTensor:
    if _is_complex(other):
        return ComplexTensor(
            tensor.real * other.real - tensor.imag * other.imag,
            tensor.imag * other.real + tensor.real * other.imag)
    else:
        return ComplexTensor(tensor.real * other, tensor.imag * other)


def _ldiv(tensor: ComplexTensor, other: OpObjType) -> ComplexTensor:
    if _is_complex(other):
        scale = other.real**2 + other.imag**2
        return ComplexTensor(
            (tensor.real * other.real + tensor.imag * other.imag) / scale,
            (tensor.imag * other.real - tensor.real * other.imag) / scale)
    else:
        return ComplexTensor(tensor.real / other, tensor.imag / other)


def _rdiv(other: OpObjType, tensor: ComplexTensor) -> ComplexTensor:
    scale = tensor.real**2 + tensor.imag**2
    if _is_complex(other):
        return ComplexTensor(
            (other.real * tensor.real + other.imag * tensor.imag) / scale,
            (other.imag * tensor.real - other.real * tensor.imag) / scale)
    else:
        other = other / scale
        return ComplexTensor(other * tensor.real, -other * tensor.imag)


def _lmatmul(tensor: ComplexTensor,
             other: Union[th.Tensor, ComplexTensor]) -> ComplexTensor:
    if _is_complex(other):
        return ComplexTensor(
            th.matmul(tensor.real, other.real) -
            th.matmul(tensor.imag, other.imag),
            th.matmul(tensor.imag, other.real) +
            th.matmul(tensor.real, other.imag))
    else:
        return ComplexTensor(th.matmul(tensor.real, other),
                             th.matmul(tensor.imag, other))


def _rmatmul(other: Union[th.Tensor, ComplexTensor],
             tensor: ComplexTensor) -> ComplexTensor:
    if _is_complex(other):
        return ComplexTensor(
            th.matmul(other.real, tensor.real) -
            th.matmul(other.imag, tensor.imag),
            th.matmul(other.imag, tensor.real) +
            th.matmul(other.real, tensor.imag))
    else:
        return ComplexTensor(th.matmul(other, tensor.real),
                             th.matmul(other, tensor.imag))


def _inverse(tensor: ComplexTensor) -> ComplexTensor:
    """
    Refer: A note on the inversion of complex matrices
    """
    m = th.cat([tensor.real, -1.0 * tensor.imag], dim=-1)
    n = th.cat([tensor.imag, tensor.real], dim=-1)
    r = th.cat([m, n], dim=-2)
    r_inv = r.inverse()
    m, _ = th.chunk(r_inv, 2, dim=-2)
    real, imag = th.chunk(m, 2, dim=-1)
    return ComplexTensor(real, -imag)


# -----------------------------------------------------------------


def _random_cplx_mat(shape, cplx=True):
    R = th.rand(*shape)
    if cplx:
        I = th.rand(*shape)
        pt_mat = ComplexTensor(R, I)
        np_mat = R.numpy() + I.numpy() * 1j
    else:
        pt_mat = R
        np_mat = R.numpy()
    return pt_mat, np_mat


def _assert_allclose(pt_mat, np_mat):
    th.testing.assert_allclose(pt_mat.real, th.from_numpy(np_mat.real))
    th.testing.assert_allclose(pt_mat.imag, th.from_numpy(np_mat.imag))


def test_add_sub_mul_div():
    pt_mat1, np_mat1 = _random_cplx_mat((8, 10))
    pt_mat2, np_mat2 = _random_cplx_mat((8, 10))
    pt_mat3, np_mat3 = _random_cplx_mat((8, 10), cplx=False)
    s1, c1 = 3.6, 2.5 + 3.4j
    params = [(pt_mat2, np_mat2), (pt_mat3, np_mat3), (s1, s1), (c1, c1)]
    for v1, v2 in params:
        # + & -
        _assert_allclose(pt_mat1 + v1, np_mat1 + v2)
        _assert_allclose(pt_mat1 - v1, np_mat1 - v2)
        _assert_allclose(v1 + pt_mat1, v2 + np_mat1)
        _assert_allclose(v1 - pt_mat1, v2 - np_mat1)
        # * & /
        _assert_allclose(pt_mat1 * v1, np_mat1 * v2)
        _assert_allclose(pt_mat1 / v1, np_mat1 / v2)
        _assert_allclose(v1 * pt_mat1, v2 * np_mat1)
        _assert_allclose(v1 / pt_mat1, v2 / np_mat1)


def test_matmul():
    pt_mat1, np_mat1 = _random_cplx_mat((8, 8))
    pt_mat2, np_mat2 = _random_cplx_mat((8, 8))
    pt_mat3, np_mat3 = _random_cplx_mat((8, 8), cplx=False)
    _assert_allclose(pt_mat1 @ pt_mat2, np_mat1 @ np_mat2)
    _assert_allclose(pt_mat2 @ pt_mat1, np_mat2 @ np_mat1)
    _assert_allclose(pt_mat1 @ pt_mat2.conj_transpose(0, 1),
                     np_mat1 @ np_mat2.T.conj())
    _assert_allclose(pt_mat1 @ pt_mat3, np_mat1 @ np_mat3)
    _assert_allclose(pt_mat3 @ pt_mat1, np_mat3 @ np_mat1)


def test_for_mvdr_ops():
    N, M = 10, 8
    # trace
    pt_mat, np_mat = _random_cplx_mat((N, M, M))
    diag_index = th.eye(M, dtype=th.bool).expand((N, M, M))
    pt_trace = pt_mat.masked_select(diag_index).view(N, M).sum(-1)
    np_trace = np.trace(np_mat, axis1=1, axis2=2)
    _assert_allclose(pt_trace, np_trace)

    # inverse
    pt_mat, np_mat = _random_cplx_mat((N, M, M))
    pt_mat += th.eye(M)
    np_mat += np.eye(M)
    pt_mat_inv = pt_mat.inverse()
    np_mat_inv = np.linalg.inv(np_mat)
    _assert_allclose(pt_mat_inv, np_mat_inv)

    # A*B^H
    pt_cplx, np_cplx = _random_cplx_mat((N, M, M))
    pt_real, np_real = _random_cplx_mat((N, M, M), cplx=False)

    np_mat = np.einsum("...it,...jt->...ij", np_cplx * np_real, np_cplx.conj())
    pt_mat = (pt_cplx * pt_real) @ pt_cplx.conj_transpose(-1, -2)
    _assert_allclose(pt_mat, np_mat)

    # A^H*B*A
    T = 20
    pt_mat1, np_mat1 = _random_cplx_mat((N, M, T))
    pt_mat2, np_mat2 = _random_cplx_mat((N, M, M))
    np_mat = np.einsum("...xt,...xy,...yt->...t", np_mat1.conj(), np_mat2,
                       np_mat1)
    pt_mat = (pt_mat1.conj() * (pt_mat2 @ pt_mat1)).sum(-2)
    _assert_allclose(pt_mat, np_mat)


if __name__ == "__main__":
    for r in range(3):
        test_add_sub_mul_div()
        test_matmul()
        test_for_mvdr_ops()
        print(f"Round {r}: Pass")
