"""Internal helpers shared by observing-planning modules."""

from astroplan import Observer
from astropy.coordinates import EarthLocation


def resolve_observer(location):
    """Return an ``astroplan.Observer`` from a site name or location object."""
    if isinstance(location, Observer):
        return location
    if isinstance(location, EarthLocation):
        return Observer(location=location)
    return Observer.at_site(str(location))
