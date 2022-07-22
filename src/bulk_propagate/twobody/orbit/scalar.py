from functools import cached_property
from warnings import warn

import numpy as np
from astropy import time, units as u
from astropy.coordinates import (
    # ICRS,
    CartesianDifferential,
    CartesianRepresentation,
    # get_body_barycentric,
)

from poliastro.frames.util import get_frame
from poliastro.twobody.elements import eccentricity_vector, energy
from poliastro.twobody.sampling import TrueAnomalyBounds
from poliastro.util import norm

from ..propagation import FarnocchiaPropagator, PropagatorKind
from ..states import BaseState
from .creation import OrbitCreationMixin

ORBIT_FORMAT = "{r_p:.0f} x {r_a:.0f} x {inc:.1f} ({frame}) orbit around {body} at epoch {epoch} ({scale})"
# String representation for orbits around bodies without predefined
# Reference frame
ORBIT_NO_FRAME_FORMAT = "{r_p:.0f} x {r_a:.0f} x {inc:.1f} orbit around {body} at epoch {epoch} ({scale})"


class Orbit(OrbitCreationMixin):
    """Position and velocity of a body with respect to an attractor
    at a given time (epoch).

    Regardless of how the Orbit is created, the implicit
    reference system is an inertial one. For the specific case
    of the Solar System, this can be assumed to be the
    International Celestial Reference System or ICRS.

    """

    def __init__(self, state, epoch):  # pylint: disable=super-init-not-called
        """Constructor.

        Parameters
        ----------
        state : BaseState
            Position and velocity or orbital elements.
        epoch : ~astropy.time.Time
            Epoch of the orbit.

        """
        self._state = state  # type: BaseState
        self._epoch = epoch  # type: time.Time

    @property
    def attractor(self):
        """Main attractor."""
        return self._state.attractor

    @property
    def epoch(self):
        """Epoch of the orbit."""
        return self._epoch

    @property
    def plane(self):
        """Fundamental plane of the frame."""
        return self._state.plane

    @cached_property
    def r(self):
        """Position vector."""
        return self._state.to_vectors().r

    @cached_property
    def v(self):
        """Velocity vector."""
        return self._state.to_vectors().v

    @cached_property
    def a(self):
        """Semimajor axis."""
        return self._state.to_classical().a

    @cached_property
    def p(self):
        """Semilatus rectum."""
        return self._state.to_classical().p

    @cached_property
    def r_p(self):
        """Radius of pericenter."""
        return self._state.r_p

    @cached_property
    def r_a(self):
        """Radius of apocenter."""
        return self._state.r_a

    @cached_property
    def ecc(self):
        """Eccentricity."""
        return self._state.to_classical().ecc

    @cached_property
    def inc(self):
        """Inclination."""
        return self._state.to_classical().inc

    @cached_property
    def raan(self):
        """Right ascension of the ascending node."""
        return self._state.to_classical().raan

    @cached_property
    def argp(self):
        """Argument of the perigee."""
        return self._state.to_classical().argp

    @property
    def nu(self):
        """True anomaly."""
        return self._state.to_classical().nu

    @cached_property
    def f(self):
        """Second modified equinoctial element."""
        return self._state.to_equinoctial().f

    @cached_property
    def g(self):
        """Third modified equinoctial element."""
        return self._state.to_equinoctial().g

    @cached_property
    def h(self):
        """Fourth modified equinoctial element."""
        return self._state.to_equinoctial().h

    @cached_property
    def k(self):
        """Fifth modified equinoctial element."""
        return self._state.to_equinoctial().k

    @cached_property
    def L(self):
        """True longitude."""
        return self.raan + self.argp + self.nu

    @cached_property
    def period(self):
        """Period of the orbit."""
        return self._state.period

    @cached_property
    def n(self):
        """Mean motion."""
        return self._state.n

    @cached_property
    def energy(self):
        """Specific energy."""
        return energy(self.attractor.k, self.r, self.v)

    @cached_property
    def e_vec(self):
        """Eccentricity vector."""
        return eccentricity_vector(self.attractor.k, self.r, self.v)

    @cached_property
    def h_vec(self):
        """Specific angular momentum vector."""
        h_vec = (
            np.cross(self.r.to_value(u.km), self.v.to(u.km / u.s))
            * u.km**2
            / u.s
        )
        return h_vec

    @cached_property
    def h_mag(self):
        """Specific angular momentum."""
        h_mag = norm(self.h_vec)
        return h_mag

    @cached_property
    def arglat(self):
        """Argument of latitude."""
        arglat = (self.argp + self.nu) % (360 * u.deg)
        return arglat

    @cached_property
    def t_p(self):
        """Elapsed time since latest perifocal passage."""
        return self._state.t_p

    def get_frame(self):
        """Get equivalent reference frame of the orbit.

        .. versionadded:: 0.14.0

        """
        return get_frame(self.attractor, self.plane, self.epoch)

    def change_plane(self, plane):
        """Changes fundamental plane.

        Parameters
        ----------
        plane : ~poliastro.frames.Planes
            Fundamental plane of the frame.

        """
        if plane is self.plane:
            return self

        coords_orig = self.get_frame().realize_frame(
            self.represent_as(CartesianRepresentation, CartesianDifferential)
        )

        dest_frame = get_frame(self.attractor, plane, obstime=self.epoch)

        coords_dest = coords_orig.transform_to(dest_frame)
        coords_dest.representation_type = CartesianRepresentation

        return Orbit.from_coords(self.attractor, coords_dest, plane=plane)

    def represent_as(self, representation, differential_class=None):
        """Converts the orbit to a specific representation.

        .. versionadded:: 0.11.0

        Parameters
        ----------
        representation : ~astropy.coordinates.BaseRepresentation
            Representation object to use. It must be a class, not an instance.
        differential_class : ~astropy.coordinates.BaseDifferential, optional
            Class in which the differential should be represented, default to None.

        Examples
        --------
        >>> from poliastro.examples import iss
        >>> from astropy.coordinates import SphericalRepresentation
        >>> iss.represent_as(CartesianRepresentation)
        <CartesianRepresentation (x, y, z) in km
            (859.07256, -4137.20368, 5295.56871)>
        >>> iss.represent_as(CartesianRepresentation).xyz
        <Quantity [  859.07256, -4137.20368,  5295.56871] km>
        >>> iss.represent_as(CartesianRepresentation, CartesianDifferential).differentials['s']
        <CartesianDifferential (d_x, d_y, d_z) in km / s
            (7.37289205, 2.08223573, 0.43999979)>
        >>> iss.represent_as(CartesianRepresentation, CartesianDifferential).differentials['s'].d_xyz
        <Quantity [7.37289205, 2.08223573, 0.43999979] km / s>
        >>> iss.represent_as(SphericalRepresentation, CartesianDifferential)
        <SphericalRepresentation (lon, lat, distance) in (rad, rad, km)
            (4.91712525, 0.89732339, 6774.76995296)
         (has differentials w.r.t.: 's')>

        """
        # As we do not know the differentials, we first convert to cartesian,
        # then let the frame represent_as do the rest
        # TODO: Perhaps this should be public API as well?
        cartesian = CartesianRepresentation(
            *self.r, differentials=CartesianDifferential(*self.v)
        )

        return cartesian.represent_as(representation, differential_class)

    def pqw(self):
        """Perifocal frame (PQW) vectors."""
        warn(
            "Orbit.pqw is deprecated and will be removed in a future release",
            DeprecationWarning,
            stacklevel=2,
        )

        if self.ecc < 1e-8:
            if abs(self.inc.to_value(u.rad)) > 1e-8:
                node = np.cross([0, 0, 1], self.h_vec) / norm(self.h_vec)
                p_vec = node / norm(node)  # Circular inclined
            else:
                p_vec = [1, 0, 0] * u.one  # Circular equatorial
        else:
            p_vec = self.e_vec / self.ecc
        w_vec = self.h_vec / norm(self.h_vec)
        q_vec = np.cross(w_vec, p_vec) * u.one
        return p_vec, q_vec, w_vec

    def __str__(self):
        if self.a > 1e7 * u.km:
            unit = u.au
        else:
            unit = u.km

        try:
            return ORBIT_FORMAT.format(
                r_p=self.r_p.to_value(unit),
                r_a=self.r_a.to(unit),
                inc=self.inc.to(u.deg),
                frame=self.get_frame().__class__.__name__,
                body=self.attractor,
                epoch=self.epoch,
                scale=self.epoch.scale.upper(),
            )
        except NotImplementedError:
            return ORBIT_NO_FRAME_FORMAT.format(
                r_p=self.r_p.to_value(unit),
                r_a=self.r_a.to(unit),
                inc=self.inc.to(u.deg),
                body=self.attractor,
                epoch=self.epoch,
                scale=self.epoch.scale.upper(),
            )

    def __repr__(self):
        return self.__str__()

    def propagate(self, value, method=FarnocchiaPropagator()):
        """Propagates an orbit a specified time.

        If value is true anomaly, propagate orbit to this anomaly and return the result.
        Otherwise, if time is provided, propagate this `Orbit` some `time` and return the result.

        Parameters
        ----------
        value : ~astropy.units.Quantity, ~astropy.time.Time, ~astropy.time.TimeDelta
            Scalar time to propagate.
        method : function, optional
            Method used for propagation, default to farnocchia.

        Returns
        -------
        Orbit
            New orbit after propagation.

        """
        if value.ndim != 0:
            raise ValueError(
                "propagate only accepts scalar values for time of flight"
            )

        if isinstance(value, time.Time) and not isinstance(
            value, time.TimeDelta
        ):
            time_of_flight = value - self.epoch
        else:
            # Works for both Quantity and TimeDelta objects
            time_of_flight = time.TimeDelta(value)

        # Check if propagator fulfills orbit requirements
        # Note there's a potential conversion here purely for convenience that could be skipped
        if self.ecc < 1.0 and not (method.kind & PropagatorKind.ELLIPTIC):
            raise ValueError(
                "Can not use an parabolic/hyperbolic propagator for elliptical/circular orbits."
            )
        elif self.ecc == 1.0 and not (method.kind & PropagatorKind.PARABOLIC):
            raise ValueError(
                "Can not use an elliptic/hyperbolic propagator for parabolic orbits."
            )
        elif self.ecc > 1.0 and not (method.kind & PropagatorKind.HYPERBOLIC):
            raise ValueError(
                "Can not use an elliptic/parabolic propagator for hyperbolic orbits."
            )

        new_state = method.propagate(
            self._state,
            time_of_flight,
        )
        new_epoch = self.epoch + time_of_flight

        return self.__class__(new_state, new_epoch)

    def to_ephem(self, strategy=TrueAnomalyBounds()):
        """Samples Orbit to return an ephemerides.

        .. versionadded:: 0.17.0

        """
        from poliastro.ephem import Ephem

        coordinates, epochs = strategy.sample(self)
        return Ephem(coordinates, epochs, self.plane)

    def sample(self, values=100, *, min_anomaly=None, max_anomaly=None):
        r"""Samples an orbit to some specified time values.

        .. versionadded:: 0.8.0

        Parameters
        ----------
        values : int
            Number of interval points (default to 100).
        min_anomaly, max_anomaly : ~astropy.units.Quantity, optional
            Anomaly limits to sample the orbit.
            For elliptic orbits the default will be :math:`E \in \left[0, 2 \pi \right]`,
            and for hyperbolic orbits it will be :math:`\nu \in \left[-\nu_c, \nu_c \right]`,
            where :math:`\nu_c` is either the current true anomaly
            or a value that corresponds to :math:`r = 3p`.

        Returns
        -------
        positions: ~astropy.coordinates.CartesianRepresentation
            Array of x, y, z positions.

        Notes
        -----
        When specifying a number of points, the initial and final
        position is present twice inside the result (first and
        last row). This is more useful for plotting.

        Examples
        --------
        >>> from astropy import units as u
        >>> from poliastro.examples import iss
        >>> iss.sample()  # doctest: +ELLIPSIS
        <CartesianRepresentation (x, y, z) in km ...
        >>> iss.sample(10)  # doctest: +ELLIPSIS
        <CartesianRepresentation (x, y, z) in km ...

        """
        if min_anomaly is not None or max_anomaly is not None:
            warn(
                "Specifying min_anomaly and max_anomaly in method `sample` is deprecated "
                "and will be removed in a future release, "
                "use `Orbit.to_ephem(strategy=TrueAnomalyBounds(min_nu=..., max_nu=...))` instead",
                DeprecationWarning,
                stacklevel=2,
            )

        ephem = self.to_ephem(
            strategy=TrueAnomalyBounds(
                min_nu=min_anomaly,
                max_nu=max_anomaly,
                num_values=values,
            ),
        )
        # We call .sample() at the end to retrieve the coordinates for the same epochs
        return ephem.sample()

    def plot(self, label=None, use_3d=False, interactive=False):
        """Plots the orbit.

        Parameters
        ----------
        label : str, optional
            Label for the orbit, defaults to empty.
        use_3d : bool, optional
            Produce a 3D plot, default to False.
        interactive : bool, optional
            Produce an interactive (rather than static) image of the orbit, default to False.
            This option requires Plotly properly installed and configured for your environment.

        """
        if not interactive and use_3d:
            raise ValueError(
                "The static plotter does not support 3D, use `interactive=True`"
            )
        elif not interactive:
            from poliastro.plotting.static import StaticOrbitPlotter

            return StaticOrbitPlotter().plot(self, label=label)
        elif use_3d:
            from poliastro.plotting.interactive import OrbitPlotter3D

            return OrbitPlotter3D().plot(self, label=label)
        else:
            from poliastro.plotting.interactive import OrbitPlotter2D

            return OrbitPlotter2D().plot(self, label=label)
