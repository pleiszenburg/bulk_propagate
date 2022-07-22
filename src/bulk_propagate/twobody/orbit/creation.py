import numpy as np
from astropy import units as u

from poliastro.constants import J2000
from poliastro.frames import Planes
from poliastro.frames.util import get_frame

from ..states import (
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

    @classmethod
    def from_coords(cls, attractor, coord, plane=Planes.EARTH_EQUATOR):
        """Creates an `Orbit` from an attractor and astropy `SkyCoord`
        or `BaseCoordinateFrame` instance.

        This method accepts position
        and velocity in any reference frame unlike `Orbit.from_vector`
        which can accept inputs in only inertial reference frame centred
        at attractor. Also note that the frame information is lost after
        creation of the orbit and only the inertial reference frame at
        body centre will be used for all purposes.

        Parameters
        ----------
        attractor : Body
            Main attractor
        coord : ~astropy.coordinates.SkyCoord or ~astropy.coordinates.BaseCoordinateFrame
            Position and velocity vectors in any reference frame. Note that coord must have
            a representation and its differential with respect to time.
        plane : ~poliastro.frames.Planes, optional
            Final orbit plane, default to Earth Equator.

        """
        if "s" not in coord.cartesian.differentials:
            raise ValueError(
                "Coordinate instance doesn't have a differential with respect to time"
            )
        if coord.size != 1:
            raise ValueError(
                "Coordinate instance must represents exactly 1 position, found: %d"
                % coord.size
            )

        # Reshape coordinate to 0 dimension if it is not already dimensionless.
        coord = coord.reshape(())

        # Get an inertial reference frame parallel to ICRS and centered at
        # attractor
        inertial_frame_at_body_centre = get_frame(
            attractor, plane, coord.obstime
        )

        if not coord.is_equivalent_frame(inertial_frame_at_body_centre):
            coord_in_irf = coord.transform_to(inertial_frame_at_body_centre)
        else:
            coord_in_irf = coord

        pos = coord_in_irf.cartesian.xyz
        vel = coord_in_irf.cartesian.differentials["s"].d_xyz

        return cls.from_vectors(
            attractor, pos, vel, epoch=coord.obstime, plane=plane
        )
