import numpy as np
from astropy import units as u

from poliastro.constants import J2000
from poliastro.frames import Planes
from poliastro.twobody.states import (
    # ClassicalState,
    # ModifiedEquinoctialState,
    RVState,
)


class OrbitCreationMixin:
    """
    Mixin-class containing class-methods to create Orbit objects
    """

    def __init__(self, *_, **__):  # HACK stub to make mypy happy
        ...

    @classmethod
    @u.quantity_input(r=u.m, v=u.m / u.s)
    def from_vectors(
        cls, attractor, r, v, epoch=J2000, plane=Planes.EARTH_EQUATOR
    ):
        """Return `Orbit` from position and velocity vectors.

        Parameters
        ----------
        attractor : Body
            Main attractor.
        r : ~astropy.units.Quantity
            Position vector wrt attractor center.
        v : ~astropy.units.Quantity
            Velocity vector.
        epoch : ~astropy.time.Time, optional
            Epoch, default to J2000.
        plane : ~poliastro.frames.Planes
            Fundamental plane of the frame.

        """
        assert np.any(r.value), "Position vector must be non zero"

        if r.ndim != 1 or v.ndim != 1:
            raise ValueError(
                f"Vectors must have dimension 1, got {r.ndim} and {v.ndim}"
            )

        ss = RVState(attractor, (r, v), plane)
        return cls(ss, epoch)
