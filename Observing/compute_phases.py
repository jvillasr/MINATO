def compute_phases(time, location, period, num_epochs, phase_tolerance, max_date, ra, dec, name, twilight='nautical',
                      alt_min=10, alt_max=90, airmass_max=2.5, save_table=True):
    '''
    Computes dates for binary stars follow-up observations
    
    time:       time of first observation
    period:     Orbital period in days
    location:   Observer's locations as in Astropy's EarthLocation.get_site_names()
    num_epochs: Number of epochs to observe covering the full orbital period
    max_date:   Maximum date to which phases will be computed
    ra:         Right asencion of target
    dec:        Decliination of target
    name:       Name of the target
    '''
    
    import astropy.units as u
    from astropy.time import Time
    from astropy.coordinates import SkyCoord, EarthLocation
    from astroplan import FixedTarget, Observer, is_observable
    from astroplan import AltitudeConstraint, AirmassConstraint, AtNightConstraint


    if twilight=='civil':
         twilight_type = AtNightConstraint.twilight_civil()
    elif twilight=='nautical':
        twilight_type = AtNightConstraint.twilight_nautical()
    elif twilight=='astronomical':
        twilight_type = AtNightConstraint.twilight_astronomical()
    
    object_coords = SkyCoord(ra, dec, unit=(u.hourangle, u.deg))
    target = FixedTarget(object_coords, name='Target')

    coords = EarthLocation.of_site(location)  
    observer_location = Observer.at_site(location)
    # current_ut = Time('2022-09-04 21:25:15', scale='utc', location=coords)
    # current_st = Time('2022-09-04 19:09:38', scale='utc', location=coords)
    # LST_conv = current_ut - current_st
    
    first_epoch = Time(time, scale='utc', location=coords)
    max_date = Time(max_date, scale='utc', location=coords)
    
    phase = 1/num_epochs
    
    dP = phase*period*u.day
    dT = dP.to('hour')
    tolerance = dT*phase_tolerance
    print('\n  You have selected', num_epochs, 'epochs spaced by', f"{dT:.2f}")
    print('  The tolerance of your observations is', f"{tolerance:.2f}", 'or', f"{tolerance.to('minute'):.2f}", '\n')

    
    # df_table = pd.
    
    for i in range(num_epochs):
        epochs = first_epoch + i*dT
        
        epochs_min = epochs - tolerance
        epochs_max = epochs + tolerance
        time_range = Time([epochs_min, epochs_max])

        constraints = [AltitudeConstraint(10*u.deg, 90*u.deg), AirmassConstraint(2.5), twilight_type]
        
        ever_observable = is_observable(constraints, observer_location, target, time_range=time_range, \
                            time_grid_resolution=0.25*u.hour)
       
        
        if epochs < max_date:
            if ever_observable:
                print('   OB'+str(i+1).zfill(2)+'   00', epochs, ' UT')         
            else:
                print('   OB'+str(i+1).zfill(2)+'   00', epochs, ' UT (not observable)')
            j=0
            next_phase = epochs + (j+1)*period*u.day
            while next_phase < max_date:
                # print(j, next_phase, max_date)
                next_phase = epochs + (j+1)*period*u.day
                next_phase_min = next_phase-tolerance
                next_phase_max = next_phase+tolerance
                ever_observable = is_observable(constraints, observer_location, target, \
                                    time_range=Time([next_phase_min, next_phase_max]), time_grid_resolution=0.25*u.hour)
                if ever_observable:
                    print('         ', str(j).zfill(2), next_phase, ' UT')
                j+=1
            print('\n')           
        else:
            print('    OB'+str(i+1).zfill(2) + ' beyond maximum date')