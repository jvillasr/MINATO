def fit_sb2_probmod(line, wavelengths, fluxes, f_errors, lines_dic, Hlines, neblines, shift, fig, axes, path):
    '''
    Probabilistic model for SB2 line profile fitting. It uses Numpyro for Bayesian inference, 
    sampling from the posterior distribution of the model parameters using MCMC with the NUTS algorithm. 
    The model includes plates for vectorized computations over epochs and wavelengths. 
    '''
    # Trim the data to the region of interest
    region_start = lines_dic[line]['region'][0]
    region_end = lines_dic[line]['region'][1]
    x_waves = [wave[(wave > region_start) & (wave < region_end)] for wave in wavelengths]
    y_fluxes = [flux[(wave > region_start) & (wave < region_end)] for flux, wave in zip(fluxes, wavelengths)]
    y_errors = [f_err[(wave > region_start) & (wave < region_end)] for f_err, wave in zip(f_errors[:7], wavelengths)]
    # min_flux_wavelengths = [wave[np.argmin(flux)] for wave, flux in zip(x_waves, y_fluxes)]

    # Initial guess for the central wavelength and width
    cen_ini = line+shift
    wid_ini = lines_dic[line]['wid_ini']
    amp_ini = 0.9 - min([flux.min() for flux in y_fluxes])
    
    def sb2_model(wavelengths=None, fluxes=None):

        n_epoch, n_wavelength = fluxes.shape

        #spectral window
        cen = npro.param("cen", (cen_ini))
        # delta_cen = npro.param("delta_cen", (wavelengths[0].max() - wavelengths[0].min()) / 8)
        # cen = npro.param("cen", (wavelengths[0].max() + wavelengths[0].min())/2)
        delta_cen = npro.param("delta_cen", (wavelengths[0].max() - wavelengths[0].min()))
        # print('centre =', cen, '+/-', delta_cen)

        # Model parameters
        continuum = npro.sample('continuum', dist.Normal(1, 0.1))
        amp1 = npro.sample('amp1', dist.Uniform(0, 0.8))
        amp2 = npro.sample('amp2', dist.Uniform(0, amp1))
        wid1 = npro.sample('wid1', dist.Uniform(0.5, 11))
        wid2 = npro.sample('wid2', dist.Uniform(0.5, 11))
        # print(f'amp1_ini = {amp_ini} +/- {amp_ini*0.2}')
        # print(f'amp2_ini = {amp_ini*0.6} +/- {amp_ini*0.6*0.2}')
        # amp1 = npro.sample('amp1', dist.Normal(amp_ini, amp_ini*0.2))
        # amp2 = npro.sample('amp2', dist.Normal(amp_ini*0.6, amp_ini*0.6*0.2))
        # wid1 = npro.sample('wid1', dist.Normal(wid_ini, wid_ini*0.2))
        # wid2 = npro.sample('wid2', dist.Normal(wid_ini, wid_ini*0.2))
        # logsig = npro.sample('logsig', dist.Normal(-2, 1))

        with npro.plate(f'epoch=1..{n_epoch}', n_epoch, dim=-2):
            mean1 = npro.sample('mean1', dist.Uniform(cen - 0.2 * delta_cen, cen + 0.2 * delta_cen))
            # mean1 = npro.sample('mean1', dist.Normal(min_flux_wavelengths, 2))
            mean2 = npro.sample('mean2', dist.Uniform(cen - 0.2 * delta_cen, cen + 0.2 * delta_cen))
            with npro.plate(f'wavelength=1..{n_wavelength}', n_wavelength, dim=-1):
                if line in Hlines:
                    # comp1 = -amp1 * jnp.exp(dist.Cauchy(mean1, wid1).log_prob(wavelengths))
                    # comp2 = -amp2 * jnp.exp(dist.Cauchy(mean2, wid2).log_prob(wavelengths))
                    comp1 = lorentzian(wavelengths, amp1, mean1, wid1)
                    comp2 = lorentzian(wavelengths, amp2, mean2, wid2)
                else:
                    # comp1 = -amp1 * jnp.exp(dist.Normal(mean1, wid1).log_prob(wavelengths))
                    # comp2 = -amp2 * jnp.exp(dist.Normal(mean2, wid2).log_prob(wavelengths))
                    comp1 = gaussian(wavelengths, amp1, mean1, wid1)
                    comp2 = gaussian(wavelengths, amp2, mean2, wid2)
                pred = continuum + comp1 + comp2
                npro.deterministic('pred_1', continuum + comp1)
                npro.deterministic('pred_2', continuum + comp2)
                model = npro.deterministic('pred', pred)
                npro.sample('obs', dist.Normal(model, 0.05), obs=fluxes)
     
    rng_key = random.PRNGKey(0)

    kernel = NUTS(sb2_model)
    mcmc = MCMC(kernel, 
                num_warmup=5000, 
                num_samples=5000)

    mcmc.run(rng_key, wavelengths=jnp.array(x_waves), fluxes=jnp.array(y_fluxes))
    mcmc.print_summary()
    trace = mcmc.get_samples()

    n_lines = 100
    n_epochs = len(x_waves)
    for epoch, ax in zip(range(n_epochs), axes.ravel()):
        ax.plot(x_waves[epoch], trace['pred'][-n_lines:, epoch, :].T, rasterized=True, color='C2', alpha=0.1)
        ax.plot(x_waves[epoch], trace['pred_1'][-n_lines:, epoch, :].T, rasterized=True, color='C0', alpha=0.1)
        ax.plot(x_waves[epoch], trace['pred_2'][-n_lines:, epoch, :].T, rasterized=True, color='C1', alpha=0.1)
        ax.plot(x_waves[epoch], y_fluxes[epoch], color='k', lw=1, alpha=0.8)
        # ax.set_ylim(0.4, 1.1)
        
    fig.supxlabel('Wavelength', size=22)
    fig.supylabel('Flux', size=22)  
    plt.savefig(path+str(line)+'_fits_SB2_.png', bbox_inches='tight', dpi=150)
    plt.close()

    return trace, x_waves, y_fluxes