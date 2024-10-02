def fit_sb2_probmod(lines, wavelengths, fluxes, f_errors, lines_dic, Hlines, neblines, shift, path):
    '''
    Probabilistic model for SB2 line profile fitting. It uses Numpyro for Bayesian inference, 
    sampling from the posterior distribution of the model parameters using MCMC with the NUTS algorithm. 
    The model includes plates for vectorized computations over epochs and wavelengths. 
    '''
    c_kms = c.to('km/s').value   
    n_lines = len(lines)
    n_epochs = len(wavelengths)
    print('Number of lines:', n_lines)
    print('Number of epochs:', n_epochs)

    # Check if lines are Hydrogen lines
    is_hline = jnp.array([line in Hlines for line in lines])

    # Interpolate fluxes and errors to the same length, but with different wavelength values
    x_waves_interp = []
    y_fluxes_interp = []
    y_errors_interp = []

    # Choose a consistent number of points for interpolation (e.g., 1000 points)
    common_grid_length = 200

    for i, line in enumerate(lines):
        region_start, region_end = lines_dic[line]['region']

        x_waves = []
        y_fluxes = []
        y_errors = []

        for wave_set, flux_set, error_set in zip(wavelengths, fluxes, f_errors):
            # Mask the regions
            mask = (wave_set > region_start) & (wave_set < region_end)
            wave_masked = wave_set[mask]
            flux_masked = flux_set[mask]
            error_masked = error_set[mask]

            # Interpolate to a common grid of the same length
            common_wavelength_grid = np.linspace(wave_masked.min(), wave_masked.max(), common_grid_length)
            interp_flux = interp1d(wave_masked, flux_masked, bounds_error=False, fill_value="extrapolate")(common_wavelength_grid)
            interp_error = interp1d(wave_masked, error_masked, bounds_error=False, fill_value="extrapolate")(common_wavelength_grid)

            x_waves.append(common_wavelength_grid)
            y_fluxes.append(interp_flux)
            y_errors.append(interp_error)

        x_waves_interp.append(x_waves)
        y_fluxes_interp.append(y_fluxes)
        y_errors_interp.append(y_errors)

    # Convert to JAX arrays (now all have the same length)
    x_waves = jnp.array(x_waves_interp)
    y_fluxes = jnp.array(y_fluxes_interp)
    y_errors = jnp.array(y_errors_interp)

    # Initial guess for the central wavelength
    # cen_ini = jnp.array([line+shift for line in lines])
    cen_ini = jnp.array([lines_dic[line]['centre'][0] for line in lines])

    def model2(λ=None, fλ=None, σ_fλ=None, K=2, shift=shift, is_hline=None):
        nλ, nτ, ndata = λ.shape

        # spectral window
        λ0 = npro.param("λ0", cen_ini)
        dλ0 = [lines_dic[line]['region'] for line in lines]
        dλ0 = jnp.array([k1 - k0 for k0, k1 in dλ0])

        # Expand λ0 and dλ0 to shape (1, nτ, nλ)
        λ0 = λ0[None, :, None]                
        dλ0 = dλ0[None, :, None]

        # Compute lower and upper bounds
        lower = (λ0 - 0.2 * dλ0) + shift
        upper = (λ0 + 0.2 * dλ0) + shift

        # prior on Δv
        Δv = npro.param('Δv', 0)
        σ_Δv = npro.param('σ_Δv', 500)

        # Continuum
        logσ_ε = npro.sample('logσ_ε', dist.Uniform(-5, 0))
        σ_ε = jnp.exp(logσ_ε)
        ε = npro.sample('ε', dist.Normal(1.0, σ_ε))

        with npro.plate(f"k=1..{K}", K, dim=-3): # Component plate
            # Sample the velocity shift per component (and epoch?)
            Δv_τk = npro.sample("Δv_τk", dist.Uniform(Δv, σ_Δv), sample_shape=(nτ,))
            Δv_τk_expanded = Δv_τk[:, :, :, jnp.newaxis]  

            with npro.plate(f'λ=1..{nλ}', nλ, dim=-2): # Lines plate
                # Reparameterize amplitudes
                # Component 0: alpha0 ~ Uniform(0.05, 0.4)
                # Component 1: alpha1 = delta1 * alpha0
                # Component 2: alpha2 ~ Uniform(0.0, 0.1)
                
                # Sample alpha0
                alpha0 = npro.sample('alpha0', dist.Uniform(0.05, 0.4))
                # Sample delta1
                delta1 = npro.sample('delta1', dist.Uniform(0.0, 1.0))
                # Compute alpha1
                alpha1 = delta1 * alpha0
                # Sample alpha2
                alpha2 = npro.sample('alpha2', dist.Uniform(0.0, 0.1))
                # Stack amplitudes
                Ak = jnp.stack([alpha0, alpha1, alpha2])  # Shape: (3,)
                print('Ak: ', Ak.shape)
                # Reshape for broadcasting
                Ak = Ak[:, :, :, :, jnp.newaxis]  # Shape: (3, 1, 1, 1)

                # Sample widths
                sigma0 = npro.sample('sigma0', dist.Uniform(0.5, 8.0))
                sigma1 = npro.sample('sigma1', dist.Uniform(0.5, 8.0))
                sigma2 = npro.sample('sigma2', dist.Uniform(0.1, 2.0))
                σk = jnp.stack([sigma0, sigma1, sigma2])  # Shape: (3,)
                print('σk: ', σk.shape)
                σk = σk[:, :, :, :, jnp.newaxis]  # Shape: (3, 1, 1, 1)

                # Ak = npro.sample('𝛼_kλ', dist.Uniform(0.05, 0.4))
                # Ak = Ak[:, :, :, jnp.newaxis]
                # σk = npro.sample('σ_kλ', dist.Uniform(0.5, 8))
                # σk = σk[:, :, :, jnp.newaxis]
                print('Ak: ', Ak.shape)
                print('σk: ', σk.shape)

                # Sample the line center
                # λ_kλ = npro.sample("λ_kλ", dist.Uniform(lower, upper))
                λ_kλ = npro.deterministic("λ_kλ", λ0)
                λ_kλ_expanded = λ_kλ[:, :, :, jnp.newaxis]   #

                λ_expanded = λ[jnp.newaxis, jnp.newaxis, :, :, :]
                μ = λ_kλ_expanded * (1 + Δv_τk_expanded / c_kms)

                # Expand is_hline for broadcasting
                is_hline_expanded = is_hline[None, :, None, None]  # Shape: (1, nτ, 1, 1)

                # Compute both profiles
                print('λ_expanded: ', λ_expanded.shape)
                print('μ', μ.shape)
                μ = μ[jnp.newaxis, :, :, :, :]
                gaussian_profile = gaussian(λ_expanded, Ak, μ, σk)
                lorentzian_profile = lorentzian(λ_expanded, Ak, μ, σk)

                with npro.plate(f'τ=1..{nτ}', nτ, dim=-1): # Epoch plate

                    # Select the appropriate profile
                    comp = jnp.where(is_hline_expanded, lorentzian_profile, gaussian_profile)

                    Ck = npro.deterministic("C_λk", comp)
                    fλ_pred = npro.deterministic("fλ_pred", ε + Ck.sum(axis=0))
        
        npro.sample("fλ", dist.Normal(fλ_pred, σ_fλ), obs=fλ)

    # rendered_model = npro.render_model(model2, model_args=(x_waves, y_fluxes, y_errors), model_kwargs={'K':3, 'is_hline': is_hline},
    #              render_distributions=True, render_params=True)
    # rendered_model.render(filename=path+'output_graph.gv', format='png')


    rng_key = random.PRNGKey(0)
    # model_init = initialize_model(rng_key, sb2_model, model_args=(x_waves, y_fluxes, y_errors))

    # kernel = NUTS(sb2_model)
    kernel = NUTS(model2)
    mcmc = MCMC(kernel, num_warmup=500, num_samples=500)
    # mcmc.run(rng_key, wavelengths=jnp.array(x_waves), fluxes=jnp.array(y_fluxes), f_errors=jnp.array(y_errors))
    mcmc.run(rng_key, λ=jnp.array(x_waves), fλ=jnp.array(y_fluxes), σ_fλ=jnp.array(y_errors), K=2, is_hline=is_hline)
    # mcmc.run(rng_key, λ=jnp.array(x_waves), fλ=jnp.array(y_fluxes), σ_fλ=jnp.array(y_errors), is_hline=is_hline, t=jnp.array(time))
    mcmc.print_summary()
    trace = mcmc.get_samples()

    print(trace.keys())
    for key in trace:
        print(f"{key}: {trace[key].shape}")


    n_sol = 100

    for idx, line in enumerate(lines): 
        fig, axes = setup_fits_plots(wavelengths)
        for epoch_idx, (epoch, ax) in enumerate(zip(range(n_epochs), axes.ravel())):
            # Extract the posterior samples for the total prediction
            fλ_pred_samples = trace['fλ_pred'][-n_sol:, idx, epoch, :]  # Shape: (n_sol, ndata)
            
            # Extract the posterior samples for the continuum
            continuum_pred_samples = trace['ε'][-n_sol:, None, None]
            # Extract the posterior samples for each component
            print('C_λk: ', trace['C_λk'].shape)
            print('continuum_pred_samples: ', continuum_pred_samples.shape)
            fλ_pred_comp1_samples = continuum_pred_samples + trace['C_λk'][-n_sol:, :, 0, idx, epoch, :]
            fλ_pred_comp2_samples = continuum_pred_samples + trace['C_λk'][-n_sol:, :, 1, idx, epoch, :]
            
            # Plot the posterior predictive samples without labels
            print('x_waves: ', x_waves.shape)
            print('fλ_pred_comp1_samples: ', fλ_pred_comp1_samples.shape)
            ax.plot(x_waves[idx][epoch], fλ_pred_comp1_samples.T, rasterized=True, color='C0', alpha=0.1)
            ax.plot(x_waves[idx][epoch], fλ_pred_comp2_samples.T, rasterized=True, color='C1', alpha=0.1)
            ax.plot(x_waves[idx][epoch], fλ_pred_samples.T, rasterized=True, color='C2', alpha=0.1)

            # Plot the observed data without label
            ax.plot(x_waves[idx][epoch], y_fluxes[idx][epoch], color='k', lw=1, alpha=0.8)
            
            # Plot vertical lines at the line center for each component
            # for comp in range(2):
            #     λ_kλ = trace['λ_kλ'][-1, comp, idx, 0]
            #     ax.axvline(x=λ_kλ, color='C{}'.format(comp), linestyle='--')

        # Create custom legend entries
        custom_lines = [
            Line2D([0], [0], color='C2', alpha=0.5, lw=2),
            Line2D([0], [0], color='C0', alpha=0.5, lw=2),
            Line2D([0], [0], color='C1', alpha=0.5, lw=2),
            Line2D([0], [0], color='k', lw=2)
        ]
        axes[0].legend(custom_lines, ['Total Prediction', 'Component 1', 'Component 2', 'Observed Data'], fontsize=10)
            
        fig.supxlabel('Wavelength', size=22)
        fig.supylabel('Flux', size=22)  
        plt.savefig(path + f'{line}_fits_SB2_.png', bbox_inches='tight', dpi=150)
        plt.close()

    return trace, x_waves, y_fluxes