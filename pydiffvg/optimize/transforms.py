"""Transform helper utilities extracted from optimize_svg."""

import math
from collections import namedtuple

import numpy as np
import torch


class TransformTools:
    TransformDecomposition = namedtuple("TransformDecomposition", "theta scale shear translate")
    TransformProperties = namedtuple(
        "TransformProperties", "has_rotation has_scale has_mirror scale_uniform has_shear has_translation"
    )

    @staticmethod
    def parse_matrix(vals):
        assert len(vals) == 6
        return np.array([[vals[0], vals[2], vals[4]], [vals[1], vals[3], vals[5]], [0, 0, 1]])

    @staticmethod
    def parse_translate(vals):
        assert len(vals) >= 1 and len(vals) <= 2
        mat = np.eye(3)
        mat[0, 2] = vals[0]
        if len(vals) > 1:
            mat[1, 2] = vals[1]
        return mat

    @staticmethod
    def parse_rotate(vals):
        assert len(vals) == 1 or len(vals) == 3
        mat = np.eye(3)
        rads = math.radians(vals[0])
        sint = math.sin(rads)
        cost = math.cos(rads)
        mat[0:2, 0:2] = np.array([[cost, -sint], [sint, cost]])
        if len(vals) > 1:
            tr1 = TransformTools.parse_translate(vals[1:3])
            tr2 = TransformTools.parse_translate([-vals[1], -vals[2]])
            mat = tr1 @ mat @ tr2
        return mat

    @staticmethod
    def parse_scale(vals):
        assert len(vals) >= 1 and len(vals) <= 2
        diag = np.array([vals[0], vals[1] if len(vals) > 1 else vals[0], 1])
        return np.diag(diag)

    @staticmethod
    def parse_skewx(vals):
        assert len(vals) == 1
        mat = np.eye(3)
        mat[0, 1] = vals[0]
        return mat

    @staticmethod
    def parse_skewy(vals):
        assert len(vals) == 1
        mat = np.eye(3)
        mat[1, 0] = vals[0]
        return mat

    @staticmethod
    def transformPoints(pointsTensor, transform):
        assert transform is not None
        one = torch.ones((pointsTensor.shape[0], 1), device=pointsTensor.device)
        homo_points = torch.cat([pointsTensor, one], dim=1)
        mult = transform.mm(homo_points.permute(1, 0)).permute(1, 0)
        tfpoints = mult[:, 0:2].contiguous()
        assert pointsTensor.shape == tfpoints.shape
        return tfpoints

    @staticmethod
    def promote_numpy(M):
        ret = np.eye(3)
        ret[0:2, 0:2] = M
        return ret

    @staticmethod
    def recompose_numpy(Theta, ScaleXY, ShearX, TXY):
        cost = math.cos(Theta)
        sint = math.sin(Theta)
        Rot = np.array([[cost, -sint], [sint, cost]])
        Scale = np.diag(ScaleXY)
        Shear = np.eye(2)
        Shear[0, 1] = ShearX

        Translate = np.eye(3)
        Translate[0:2, 2] = TXY

        M = TransformTools.promote_numpy(Rot @ Scale @ Shear) @ Translate
        return M

    @staticmethod
    def promote(m):
        M = torch.eye(3).to(m.device)
        M[0:2, 0:2] = m
        return M

    @staticmethod
    def make_rot(Theta):
        sint = Theta.sin().squeeze()
        cost = Theta.cos().squeeze()
        Rot = torch.stack((torch.stack((cost, -sint)), torch.stack((sint, cost))))
        return Rot

    @staticmethod
    def make_scale(ScaleXY):
        if ScaleXY.squeeze().dim() == 0:
            ScaleXY = ScaleXY.squeeze()
            return torch.diag(torch.stack([ScaleXY, ScaleXY])).to(ScaleXY.device)
        else:
            return torch.diag(ScaleXY).to(ScaleXY.device)

    @staticmethod
    def make_shear(ShearX):
        m = torch.eye(2).to(ShearX.device)
        m[0, 1] = ShearX
        return m

    @staticmethod
    def make_translate(TXY):
        m = torch.eye(3).to(TXY.device)
        m[0:2, 2] = TXY
        return m

    @staticmethod
    def recompose(Theta, ScaleXY, ShearX, TXY):
        Rot = TransformTools.make_rot(Theta)
        Scale = TransformTools.make_scale(ScaleXY)
        Shear = TransformTools.make_shear(ShearX)
        Translate = TransformTools.make_translate(TXY)

        return TransformTools.promote(Rot.mm(Scale).mm(Shear)).mm(Translate)

    @staticmethod
    def make_named(decomp):
        if not isinstance(decomp, TransformTools.TransformDecomposition):
            decomp = TransformTools.TransformDecomposition(
                theta=decomp[0], scale=decomp[1], shear=decomp[2], translate=decomp[3]
            )
        return decomp

    @staticmethod
    def analyze_transform(decomp):
        decomp = TransformTools.make_named(decomp)
        epsilon = 1e-3
        has_rotation = abs(decomp.theta) > epsilon
        has_scale = abs((abs(decomp.scale) - 1)).max() > epsilon
        scale_len = (
            decomp.scale.squeeze().ndim > 0
            if isinstance(decomp.scale, np.ndarray)
            else decomp.scale.squeeze().dim() > 0
        )
        has_mirror = scale_len and decomp.scale[0] * decomp.scale[1] < 0
        scale_uniform = (not scale_len) or abs(abs(decomp.scale[0]) - abs(decomp.scale[1])) < epsilon
        has_shear = abs(decomp.shear) > epsilon
        has_translate = max(abs(decomp.translate[0]), abs(decomp.translate[1])) > epsilon

        return TransformTools.TransformProperties(
            has_rotation=has_rotation,
            has_scale=has_scale,
            has_mirror=has_mirror,
            scale_uniform=scale_uniform,
            has_shear=has_shear,
            has_translation=has_translate,
        )

    @staticmethod
    def check_and_decomp(M):
        decomp = (
            TransformTools.decompose(M)
            if M is not None
            else TransformTools.TransformDecomposition(theta=0, scale=(1, 1), shear=0, translate=(0, 0))
        )
        props = TransformTools.analyze_transform(decomp)
        return (decomp, props)

    @staticmethod
    def tf_to_string(M):
        return "matrix({} {} {} {} {} {})".format(M[0, 0], M[1, 0], M[0, 1], M[1, 1], M[0, 2], M[1, 2])

    @staticmethod
    def decomp_to_string(decomp):
        decomp = TransformTools.make_named(decomp)
        ret = ""
        props = TransformTools.analyze_transform(decomp)
        if props.has_rotation:
            ret += "rotate({}) ".format(math.degrees(decomp.theta.item()))
        if props.has_scale:
            if decomp.scale.dim() == 0:
                ret += "scale({}) ".format(decomp.scale.item())
            else:
                ret += "scale({} {}) ".format(decomp.scale[0], decomp.scale[1])
        if props.has_shear:
            ret += "skewX({}) ".format(decomp.shear.item())
        if props.has_translation:
            ret += "translate({} {}) ".format(decomp.translate[0], decomp.translate[1])

        return ret

    @staticmethod
    def decompose(M):
        m = M[0:2, 0:2]
        t0 = M[0:2, 2]
        TXY = np.linalg.solve(m, t0)

        q, r = np.linalg.qr(m)

        ref = np.array([[1, 0], [0, np.sign(np.linalg.det(q))]])

        Rot = np.dot(q, ref)

        ref2 = np.array([[1, 0], [0, np.sign(np.linalg.det(r))]])

        r2 = np.dot(ref2, r)

        Ref = np.dot(ref, ref2)

        sc = np.diag(r2)
        Scale = np.diagflat(sc)

        Shear = np.eye(2)
        Shear[0, 1] = r2[0, 1] / sc[0]
        ShearX = r2[0, 1] / sc[0]

        if np.sum(sc) < 0:
            Rot = np.dot(Rot, -np.eye(2))
            Scale = -Scale

        Theta = math.atan2(Rot[1, 0], Rot[0, 0])
        ScaleXY = np.array([Scale[0, 0], Scale[1, 1] * Ref[1, 1]])

        return TransformTools.TransformDecomposition(theta=Theta, scale=ScaleXY, shear=ShearX, translate=TXY)
