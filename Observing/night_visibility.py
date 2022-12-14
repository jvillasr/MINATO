def NVTC(name, RA, DEC, time0, location):
    '''Night Visibility Time Calculator'''

    import numpy as np
    import matplotlib.pyplot as plt
    import astropy.units as u
    from matplotlib import dates
    from astropy.coordinates import get_sun, get_moon
    from astropy.time import Time
    from astropy.coordinates import SkyCoord, AltAz, EarthLocation
    from astroplan import Observer
    import myRC
       
    coords = EarthLocation.of_site(location)
    # print(coords)

    obj_time = Time(time0, scale='utc', location=coords)
    delta_time = np.linspace(0, 24, 25)*u.hour
    # print(delta_time)

    
    observer_location = Observer.at_site(location)

    civil_twilight_pm = observer_location.twilight_evening_civil(time=obj_time, which='nearest')#.datetime
    nautic_twilight_pm = observer_location.twilight_evening_nautical(time=obj_time, which='nearest')#.datetime
    astron_twilight_pm = observer_location.twilight_evening_astronomical(time=obj_time, which='nearest')#.datetime

    civil_twilight_am = observer_location.twilight_morning_civil(time=obj_time+10*u.hour, which='nearest')#.datetime
    nautic_twilight_am = observer_location.twilight_morning_nautical(time=obj_time+10*u.hour, which='nearest')#.datetime
    astron_twilight_am = observer_location.twilight_morning_astronomical(time=obj_time+10*u.hour, which='nearest')#.datetime

    beg_night = civil_twilight_pm
    end_night = civil_twilight_am
    print(type(beg_night), type(end_night))
    delta_t = end_night - beg_night
    observe_time = beg_night + delta_t*np.linspace(0, 1, 75)
    altaz_frame = AltAz(obstime=observe_time, location=coords)
    # print(observe_time)
    t = Time(observe_time, format='jyear', scale='utc')

    fig, ax = plt.subplots(figsize=(8, 6))

    for ra, dec in zip(RA, DEC):
        obj_coords = SkyCoord(ra=ra, dec=dec, unit=(u.hourangle, u.deg))
        obj_altaz = obj_coords.transform_to(AltAz(obstime=observe_time,location=coords))        
        obj_airmass = obj_altaz.secz
        # ax.plot(t.plot_date, obj_altaz.alt, '-', label=name, lw=4, zorder=3)
        ax.plot(t.plot_date, obj_airmass, '-', label=name, lw=4, zorder=3)
    ax.set_ylim(2.8, 1)
    # sunaltazs_frame = get_sun(observe_time).transform_to(altaz_frame)
    moonaltazs_frame = get_moon(observe_time).transform_to(altaz_frame)
    moon_airmass = moonaltazs_frame.secz
    ax.plot_date(t.plot_date, moon_airmass, ls='-', ms=6, color='xkcd:manilla', label='Moon', zorder=2)
    # ax.plot_date(t.plot_date, sunaltazs_frame.alt, ls='-', ms=12, color='xkcd:sunny yellow', label='Sun')

    y_min, y_max = ax.get_ylim()
    ax.fill_betweenx(np.linspace(y_min,95,100), civil_twilight_pm.datetime, civil_twilight_am.datetime, color='k', alpha=0.4, zorder=0)
    ax.fill_betweenx(np.linspace(y_min,95,100), nautic_twilight_pm.datetime, nautic_twilight_am.datetime, color='k', alpha=0.6, zorder=0)
    ax.fill_betweenx(np.linspace(y_min,95,100), astron_twilight_pm.datetime, astron_twilight_am.datetime, color='k', alpha=0.7, zorder=0)
    plt.grid(alpha=0.5, zorder=1)
        # ax.colorbar().set_label('Azimuth [deg]')
    # plt.gcf().autofmt_xdate()
        # ax.set_title('$t_{\\rm ini} = $'+time.to_value('iso', subfmt='date_hm')+' hrs')
    ax.legend(loc=0, fontsize=14, framealpha=0.9)

    ax.xaxis.set_major_locator(dates.HourLocator(interval=1))
    ax.xaxis.set_major_formatter(dates.DateFormatter('%Hh'))        
            # ax.set_xticks(t.to_value('iso', subfmt='date_hm'))    
    # ax.set_yticks(range(10, 100, 10))
    ax.tick_params(axis='both', direction='out', color='white')#, rotation=30)

    ax.set(xlim=(beg_night.datetime, end_night.datetime), xlabel=('Time (UT)'), ylabel=('Altitude (deg)'))
    plt.tight_layout()
    plt.show()
    # plt.savefig('visibility_curves.png')

    return obj_altaz
    # print(f"Objects's Altitude = {obj_altaz.alt:.4}")


''' Test '''
# from astropy.time import Time
# from astropy.coordinates import SkyCoord, AltAz, EarthLocation
# import astropy.units as u

# coords = EarthLocation.of_site('Roque de los Muchachos')
# obj_time = Time('2022-09-11 01:16', scale='utc', location=coords)
# obj_coords = SkyCoord(ra='20 03 52.1', dec='23 20 26.5', unit=(u.hourangle, u.deg))
# obj_altaz = obj_coords.transform_to(AltAz(obstime=obj_time,location=coords))
# print(obj_altaz.alt.to_string(unit=u.deg, precision=5))

# ORM = 'Roque de los Muchachos'
# ra, dec = ['20 18 00.62'], ['36 39 03.06']
# time = '2022-12-01 21:00'
# NVTC('[NH52] 63', ra, dec, time, ORM)