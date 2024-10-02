def model2(λ=None, fλ=None, σ_fλ=None, K=2, is_hline=None, t=None):
    nτ, n_epochs, ndata = λ.shape

    # Known rest-frame line centers (assuming they're the same for all components)
    λ0_mean = jnp.mean(cen_ini)  # Or use a known value
    σ_λ0 = 0.1  # Adjust based on uncertainty

    # Sample rest-frame line centers per component
    λ_τk = npro.sample("λ_τk", dist.Normal(λ0_mean, σ_λ0), sample_shape=(K,))
    λ_τk_expanded = λ_τk[:, None, None, None]  # Shape: (K, 1, 1, 1)

    # Continuum level
    σ_ε = 0.1
    ε = npro.sample('ε', dist.Normal(1.0, σ_ε))

    # Model the radial velocities over time
    # Assuming t is an array of times for each epoch
    min_period = 1.0    # Minimum expected period in days
    max_period = 100.0  # Maximum expected period in days
    max_amplitude = 300.0
    # Sample orbital parameters per component
    period = npro.sample('period', dist.Uniform(min_period, max_period), sample_shape=(K,))
    amplitude = npro.sample('amplitude', dist.Uniform(0, max_amplitude), sample_shape=(K,))
    phase = npro.sample('phase', dist.Uniform(0, 2 * jnp.pi), sample_shape=(K,))

    # Reshape parameters for broadcasting
    period = period[:, None]     # Shape: (K, 1)
    amplitude = amplitude[:, None]  # Shape: (K, 1)
    phase = phase[:, None]       # Shape: (K, 1)
    
    # Compute radial velocities over time for each component
    Δv_kλ = amplitude * jnp.sin(2 * jnp.pi * t[None, :] / period + phase)  # Shape: (K, n_epochs)
    Δv_kλ_expanded = Δv_kλ[:, None, :, None]  # Shape: (K, 1, n_epochs, 1)

    with npro.plate(f"k=1..{K}", K, dim=-4):
        Ak = npro.sample('𝛼_k', dist.Uniform(0.1, 0.6))
        Ak = Ak[:, None, None, None]  # Shape: (K, 1, 1, 1)
        σk = npro.sample('σ_k', dist.Uniform(0.5, 8))
        σk = σk[:, None, None, None]  # Shape: (K, 1, 1, 1)

        # Compute observed line centers μ
        c_kms = 299792.458
        μ = λ_τk_expanded * (1 + Δv_kλ_expanded / c_kms)  # Shape: (K, 1, n_epochs, 1)

        # Expand λ for broadcasting
        λ_expanded = λ[None, :, :, :]  # Shape: (1, nτ, n_epochs, ndata)

        # Expand is_hline for broadcasting
        is_hline_expanded = is_hline[None, :, None, None]  # Shape: (1, nτ, 1, 1)

        # Compute both profiles
        gaussian_profile = gaussian(λ_expanded, Ak, μ, σk)
        lorentzian_profile = lorentzian(λ_expanded, Ak, μ, σk)

        # Select the appropriate profile
        comp = jnp.where(is_hline_expanded, lorentzian_profile, gaussian_profile)  # Shape: (K, nτ, n_epochs, ndata)

        # Store per-line component contributions
        Ck_per_line = npro.deterministic("Ck_per_line", comp)  # Shape: (K, nτ, n_epochs, ndata)

        # Sum over components to get predicted flux per line
        fλ_pred_per_line = npro.deterministic("fλ_pred_per_line", ε + Ck_per_line.sum(axis=0))  # Shape: (nτ, n_epochs, ndata)

        # Sum over lines to get the total predicted flux
        fλ_pred = npro.deterministic("fλ_pred", fλ_pred_per_line.sum(axis=0))  # Shape: (n_epochs, ndata)

    # Observe the fluxes
    npro.sample("fλ", dist.Normal(fλ_pred_per_line, σ_fλ), obs=fλ)


# Extract variables from trace
n_sol = 100  # Number of samples to use for plotting (adjust as needed)
fλ_pred_per_line_samples = trace['fλ_pred_per_line'][-n_sol:, :, :, :]  # Shape: (n_sol, nτ, n_epochs, ndata)
Ck_per_line_samples = trace['Ck_per_line'][-n_sol:, :, :, :, :]  # Shape: (n_sol, K, nτ, n_epochs, ndata)
ε_samples = trace['ε'][-n_sol:]  # Shape: (n_sol,)

for idx, line in enumerate(lines): 
    fig, axes = setup_fits_plots(wavelengths)
    for epoch_idx, (epoch, ax) in enumerate(zip(range(n_epochs), axes.ravel())):
        # Extract the posterior samples for the total prediction for this line and epoch
        fλ_pred_samples = fλ_pred_per_line_samples[:, idx, epoch_idx, :]  # Shape: (n_sol, ndata)
        
        # Extract the posterior samples for each component for this line and epoch
        # ε_expanded = ε_samples[:, None]  # Shape: (n_sol, 1)
        # Ck_line_epoch = Ck_per_line_samples[:, :, idx, epoch_idx, :]  # Shape: (n_sol, K, ndata)
        # fλ_pred_comp1_samples = ε_expanded + Ck_line_epoch[:, 0, :]  # Component 1
        # fλ_pred_comp2_samples = ε_expanded + Ck_line_epoch[:, 1, :]  # Component 2

        # Ensure idx and epoch_idx are integers
        idx = int(idx)
        epoch_idx = int(epoch_idx)

        # Extract the posterior samples for each component for this line and epoch
        ε_expanded = ε_samples[:, None]  # Shape: (n_sol, 1)
        Ck_line_epoch = Ck_per_line_samples[:, :, idx, epoch_idx, :]  # Shape: (n_sol, K, ndata)

        # Expand ε to match the shape (n_sol, ndata)
        ε_expanded = jnp.broadcast_to(ε_expanded, (n_sol, Ck_line_epoch.shape[-1]))

        # Compute the component predictions
        fλ_pred_comp1_samples = ε_expanded + Ck_line_epoch[:, 0, :]  # Shape: (n_sol, ndata)
        fλ_pred_comp2_samples = ε_expanded + Ck_line_epoch[:, 1, :]  # Shape: (n_sol, ndata)


        # Plot the posterior predictive samples
        ax.plot(x_waves[idx][epoch_idx], fλ_pred_samples.T, rasterized=True, color='C2', alpha=0.1)
        ax.plot(x_waves[idx][epoch_idx], fλ_pred_comp1_samples.T, rasterized=True, color='C0', alpha=0.1)
        ax.plot(x_waves[idx][epoch_idx], fλ_pred_comp2_samples.T, rasterized=True, color='C1', alpha=0.1)
        
        # Plot the observed data
        ax.plot(x_waves[idx][epoch_idx], y_fluxes[idx][epoch_idx], color='k', lw=1, alpha=0.8)
        
    # Create custom legend entries
    custom_lines = [
        Line2D([0], [0], color='C2', alpha=0.5, lw=2),
        Line2D([0], [0], color='C0', alpha=0.5, lw=2),
        Line2D([0], [0], color='C1', alpha=0.5, lw=2),
        Line2D([0], [0], color='k', lw=2)
    ]
    axes[0].legend(custom_lines, ['Total Prediction', 'Component 1', 'Component 2', 'Observed Data'], fontsize=12)
        
    fig.supxlabel('Wavelength', size=22)
    fig.supylabel('Flux', size=22)  
    plt.savefig(path + f'{line}_fits_SB2_.png', bbox_inches='tight', dpi=150)
    plt.close()