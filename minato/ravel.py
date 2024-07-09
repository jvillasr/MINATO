import sys
import copy
import os
import matplotlib
import csv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import StrMethodFormatter
from lmfit import Model, Parameters, models, minimize, report_fit
from datetime import date
from iteration_utilities import duplicates, unique_everseen
from astropy.io import fits
from astropy.timeseries import LombScargle
from scipy.optimize import basinhopping
from scipy.signal import find_peaks
from scipy.interpolate import interp1d
import numpyro as npro
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS
from jax import numpy as jnp
from jax import random

def gaussian(x, amp, cen, wid):
    "1-d gaussian: gaussian(x, amp, cen, wid)"
    return -amp*jnp.exp(-(x-cen)**2 /(2*(wid/2.355)**2))

def lorentzian(x, amp, cen, wid):
    "1-d lorentzian: lorentzian(x, amp, cen, wid)"
    return -amp*(wid**2/( 4*(x-cen)**2 + wid**2 ))

def nebu(x, amp, cen, wid):
    "1-d gaussian: gaussian(x, amp, cen, wid)"
    return amp*np.exp(-(x-cen)**2 /(2*(wid/2.355)**2))

def double_gaussian_with_continuum(x, amp1, cen1, wid1, amp2, cen2, wid2, cont):
    return (-amp1 * np.exp(-(x - cen1)**2 / (2*(wid1/2.355)**2)) -
            amp2 * np.exp(-(x - cen2)**2 / (2*(wid2/2.355)**2)) + cont)

# can be replaced by axes.flatten()
def trim_axs(axs, N):
    """little helper to massage the axs list to have correct length..."""
    self.axs = axs
    self.N = N
    axs = axs.flat
    for ax in axs[N:]:
        ax.remove()
    return axs[:N]

def read_fits(fits_file):
    with fits.open(fits_file) as hdul:
        # hdul.info()
        header = hdul[0].header
        star_epoch = header['OBJECT']+'_'+ header['EPOCH_ID']
        mjd = header['MJD_MID']
        wave = hdul[1].data['WAVELENGTH']
        flux = hdul[1].data['SCI_NORM']
        ferr = hdul[1].data['SCI_NORM_ERR']
        return wave, flux, ferr, star_epoch, mjd

def read_spectra(filelist, path, file_type):
    wavelengths, fluxes, f_errors, names, jds = [], [], [], [], []
    for spec in filelist:
        if file_type in ['dat', 'txt', 'csv']:
            names.append(spec.replace(f'.{file_type}', ''))
            df = pd.read_csv(path + spec, sep='\t', header=None)
            wavelengths.append(np.array(df[0]))
            fluxes.append(np.array(df[1]))
            try:
                f_errors.append(np.array(df[2]))
            except:
                f_errors.append(1 / (np.array(df[1])) ** 2)
            jds = None
        elif file_type == 'fits':
            
            wave, flux, ferr, star, mjd = read_fits(spec)
            wavelengths.append(wave)
            fluxes.append(flux)
            f_errors.append(ferr)
            names.append(star)
            jds.append(mjd)
    return wavelengths, fluxes, f_errors, names, jds

def setup_star_directory_and_save_jds(names, jds, path):
    star = names[0].split('_')[0] + '_' + names[0].split('_')[1] + '/'
    path = path.replace('FITS/', '') + star
    if not os.path.exists(path):
        os.makedirs(path)
    if jds:    
        print(jds)
        df_mjd = pd.DataFrame()
        df_mjd['epoch'] = names
        df_mjd['JD'] = jds
        df_mjd.to_csv(path + 'JDs.txt', index=False, header=False, sep='\t')
    return path

def setup_line_dictionary():
    lines_dic = {
        3995: { 'region':[3990, 4005], 'wid_ini':2, 'title':'N II $\lambda$3995'},
        4009: { 'region':[4005, 4018], 'wid_ini':3, 'title':'He I $\lambda$4009'},
        4026: { 'region':[4017, 4043], 'wid_ini':3, 'title':'He I $\lambda$4026'},
        4102: { 'region':[4085, 4120], 'wid_ini':5, 'title':'H$\delta$'},
        4121: { 'region':[4118, 4130], 'wid_ini':3, 'title':'He I $\lambda$4121'},
        4128: { 'region':[4124, 4136], 'wid_ini':2, 'title':'Si II $\lambda$4128'},
        4131: { 'region':[4128, 4140], 'wid_ini':2, 'title':'Si II $\lambda$4131'},
        4144: { 'region':[4135, 4160], 'wid_ini':4, 'title':'He I $\lambda$4144'},
        4233: { 'region':[4229, 4241], 'wid_ini':2, 'title':'Fe II $\lambda$4233'},
        4267: { 'region':[4263, 4275], 'wid_ini':2, 'title':'C II $\lambda$4267'},
        4340: { 'region':[4320, 4360], 'wid_ini':6, 'title':'H$\gamma$'},
        4388: { 'region':[4380, 4405], 'wid_ini':4, 'title':'He I $\lambda$4388'},
        4471: { 'region':[4462, 4487], 'wid_ini':4, 'title':'He I $\lambda$4471'},
        4481: { 'region':[4478, 4490], 'wid_ini':2, 'title':'Mg II $\lambda$4481'},
        4542: { 'region':[4537, 4552], 'wid_ini':3, 'title':'He II $\lambda$4542'},
        4553: { 'region':[4547, 4562], 'wid_ini':3, 'title':'Si III $\lambda$4553'},
        4861: { 'region':[4840, 4875], 'wid_ini':5, 'title':'H$\beta$'}, 
        4922: { 'region':[4915, 4930], 'wid_ini':4, 'title':'He I $\lambda$4922'}, 
        5412: { 'region':[5405, 5419], 'wid_ini':4, 'title':'He II $\lambda$5412'},
        5876: { 'region':[5865, 5888], 'wid_ini':4, 'title':'He I $\lambda$5876'},  
        5890: { 'region':[5881, 5905], 'wid_ini':3, 'title':'Na I $\lambda$5890'}, 
        6562: { 'region':[6542, 6583], 'wid_ini':5, 'title':'H$\alpha$'}, 
        6678: { 'region':[6668, 6690], 'wid_ini':4, 'title':'He I $\lambda$6678'}, 
        7774: { 'region':[7762, 7786], 'wid_ini':3, 'title':'O I $\lambda$7774'}
    }
    return lines_dic

def initialize_fit_variables(lines):
    cen1, cen1_er = [[] for _ in range(len(lines))], [[] for _ in range(len(lines))]
    amp1, amp1_er = [[] for _ in range(len(lines))], [[] for _ in range(len(lines))]
    wid1, wid1_er = [[] for _ in range(len(lines))], [[] for _ in range(len(lines))]
    cen2, cen2_er = [[] for _ in range(len(lines))], [[] for _ in range(len(lines))]
    amp2, amp2_er = [[] for _ in range(len(lines))], [[] for _ in range(len(lines))]
    wid2, wid2_er = [[] for _ in range(len(lines))], [[] for _ in range(len(lines))]
    dely, sdev = [[] for _ in range(len(lines))], [[] for _ in range(len(lines))]
    results, comps = [[] for _ in range(len(lines))], [[] for _ in range(len(lines))]
    delta_cen, chisqr = [[] for _ in range(len(lines))], [[] for _ in range(len(lines))]
    return (
        cen1, cen1_er, amp1, amp1_er, wid1, wid1_er, 
        cen2, cen2_er, amp2, amp2_er, wid2, wid2_er, 
        dely, sdev, results, comps, delta_cen, chisqr
    )

def setup_fits_plots(wavelengths):
    nplots = len(wavelengths)
    ncols = int(np.sqrt(nplots))
    nrows = nplots // ncols
    if ncols * nrows < nplots:
        nrows += 1
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(3 * ncols, 2 * nrows), sharey=True, sharex=True)
    fig.subplots_adjust(wspace=0., hspace=0.)
    axes = axes.flatten()
    return fig, axes

def fit_sb2(line, wavelengths, fluxes, f_errors, lines_dic, Hlines, neblines, shift, axes, path):
    # Trim the data to the region of interest
    region_start = lines_dic[line]['region'][0]
    region_end = lines_dic[line]['region'][1]
    x_waves = [wave[(wave > region_start) & (wave < region_end)] for wave in wavelengths]
    y_fluxes = [flux[(wave > region_start) & (wave < region_end)] for flux, wave in zip(fluxes, wavelengths)]
    y_errors = [f_err[(wave > region_start) & (wave < region_end)] for f_err, wave in zip(f_errors, wavelengths)]

    # Initial guess for the central wavelength and width
    cen_ini = line+shift
    wid_ini = lines_dic[line]['wid_ini']
    amp_ini = 0.9 - min([flux.min() for flux in y_fluxes])

    # Define the objective function for global optimization
    def objective(params, x, data, errors):
        resid = []
        for i, (x_vals, data_vals, err_vals) in enumerate(zip(x, data, errors)):
            cen1 = np.float64(params[f'cen1_{i+1}'].value)
            cen2 = np.float64(params[f'cen2_{i+1}'].value)
            model = double_gaussian_with_continuum(x_vals, np.float64(params['amp1'].value), cen1, np.float64(params['wid1'].value), 
                                                np.float64(params['amp2'].value), cen2, np.float64(params['wid2'].value), np.float64(params['cont'].value))
            resid.append((data_vals - model) / err_vals)  # Weighted residuals
        return np.concatenate(resid)  # Return the weighted residual array

    # Scalar objective function for basinhopping
    def scalar_objective(params, x, data, errors):
        return np.sum(objective(params, x, data, errors)**2)  # Sum of squares of weighted residuals

    # Initialize parameters for each epoch
    def initialize_params(wavelengths):
        params = Parameters()
        params.add('amp1', value=amp_ini, min=0.05, max=1)
        params.add('wid1', value=wid_ini, min=0.5, max=10.0)
        params.add('amp2', value=amp_ini*0.7, min=0.05, max=1)
        params.add('wid2', value=wid_ini, min=0.5, max=10.0)
        params.add('cont', value=1.0, min=0.9, max=1.1)
        for i in range(len(wavelengths)):
            params.add(f'cen1_{i+1}', value=np.random.uniform(cen_ini-5, cen_ini+5), min=cen_ini-10, max=cen_ini+10)
            params.add(f'cen2_{i+1}', value=np.random.uniform(cen_ini-5, cen_ini+5), min=cen_ini-10, max=cen_ini+10)
        return params

    # Perform global optimization using Basin Hopping
    def global_optimization(wavelengths, fluxes, flux_errors, params):
        initial_params = params_to_array(params)
        minimizer_kwargs = {"method": "L-BFGS-B", "args": (wavelengths, fluxes, flux_errors, params)}
        result = basinhopping(scalar_objective_wrapper, initial_params, minimizer_kwargs=minimizer_kwargs, niter=100)
        array_to_params(result.x, params)
        return params

    # Perform local optimization using lmfit
    def local_optimization(wavelengths, fluxes, flux_errors, params):
        result = minimize(objective, params, args=(wavelengths, fluxes, flux_errors))
        return result

    # Convert lmfit Parameters to a flat array
    def params_to_array(params):
        return np.array([param.value for param in params.values()], dtype=np.float64)

    # Convert a flat array back to lmfit Parameters
    def array_to_params(param_array, params):
        for i, (name, param) in enumerate(params.items()):
            param.set(value=np.float64(param_array[i]))

    # Wrapper function for the scalar objective to use with scipy.optimize
    def scalar_objective_wrapper(param_array, x, data, errors, params):
        array_to_params(param_array, params)
        return scalar_objective(params, x, data, errors)  # Return the scalar value

    # Initialize and optimize parameters
    params = initialize_params(x_waves)
    params = global_optimization(x_waves, y_fluxes, y_errors, params)
    result = local_optimization(x_waves, y_fluxes, y_errors, params)

    # Plot the results
    # for i, (x_vals, data_vals) in enumerate(zip(x_waves, y_fluxes)):
    #     ax = axes[i]
    #     ax.plot(x_vals, data_vals, '-', label=f'Dataset {i+1}')
    #     model_fit = double_gaussian_with_continuum(x_vals, result.params['amp1'].value, result.params[f'cen1_{i+1}'].value, result.params['wid1'].value, 
    #                                             result.params['amp2'].value, result.params[f'cen2_{i+1}'].value, result.params['wid2'].value, result.params['cont'].value)
    #     component1 = gaussian(x_vals, result.params['amp1'].value, result.params[f'cen1_{i+1}'].value, result.params['wid1'].value)
    #     component2 = gaussian(x_vals, result.params['amp2'].value, result.params[f'cen2_{i+1}'].value, result.params['wid2'].value)
        
    #     ax.plot(x_vals, model_fit, '-', label=f'Fit {i+1}')
    #     ax.plot(x_vals, component1 + result.params['cont'].value, '--', label=f'Component 1 - Dataset {i+1}')
    #     ax.plot(x_vals, component2 + result.params['cont'].value, '--', label=f'Component 2 - Dataset {i+1}')
    #     # ax.legend()
    #     ax.set_xlabel('Wavelength')
    #     ax.set_ylabel('Flux')
    #     ax.set_title(f'Epoch {i+1}')

    # plt.tight_layout()
    # plt.show()
    # plt.show()

    # Save fit results
    with open(path + str(line) + '_stats.txt', 'w') as f:
        original_stdout = sys.stdout  # Save a reference to the original standard output
        sys.stdout = f  # Change the standard output to the file we created.
        report_fit(result)
        sys.stdout = original_stdout  # Reset the standard output to its original value
    
    # Print optimized parameters
    # for name, param in result.params.items():
    #     print(f"{name}: {param.value}")

    return result, x_waves, y_fluxes

def fit_sb2_probmod(line, wavelengths, fluxes, f_errors, lines_dic, Hlines, neblines, shift, axes, path):
    '''
    Probabilistic model for SB2 line profile fitting. It uses Numpyro for Bayesian inference, 
    sampling from the posterior distribution of the model parameters using MCMC with the NUTS algorithm. 
    The model includes plates for vectorized computations over epochs and wavelengths. 
    '''
    # Trim the data to the region of interest
    region_start = lines_dic[line]['region'][0]
    region_end = lines_dic[line]['region'][1]
    x_waves = [wave[(wave > region_start) & (wave < region_end)] for wave in wavelengths[:6]]
    y_fluxes = [flux[(wave > region_start) & (wave < region_end)] for flux, wave in zip(fluxes[:6], wavelengths[:6])]
    y_errors = [f_err[(wave > region_start) & (wave < region_end)] for f_err, wave in zip(f_errors[:6], wavelengths[:6])]

    # x_waves = [
    # [4017.062037193338, 4017.262030324732, 4017.4620234561266, 4017.662016587521, 4017.862009718916, 4018.0620028503104, 4018.261995981705, 4018.4619891131006, 4018.661982244495, 4018.8619753758894, 4019.0619685072843, 4019.2619616386787, 4019.461954770073, 4019.6619479014685, 4019.861941032863, 4020.0619341642573, 4020.261927295652, 4020.4619204270466, 4020.661913558441, 4020.861906689836, 4021.061899821231, 4021.261892952625, 4021.4618860840205, 4021.661879215415, 4021.8618723468094, 4022.0618654782043, 4022.2618586095987, 4022.4618517409936, 4022.661844872388, 4022.861838003783, 4023.0618311351773, 4023.2618242665717, 4023.461817397967, 4023.6618105293614, 4023.8618036607563, 4024.061796792151, 4024.2617899235456, 4024.46178305494, 4024.661776186335, 4024.8617693177293, 4025.0617624491238, 4025.261755580519, 4025.4617487119135, 4025.661741843308, 4025.861734974703, 4026.0617281060972, 4026.2617212374917, 4026.461714368887, 4026.661707500282, 4026.8617006316763, 4027.061693763071, 4027.2616868944656, 4027.46168002586, 4027.6616731572544, 4027.8616662886493, 4028.061659420044, 4028.2616525514386, 4028.461645682834, 4028.6616388142284, 4028.861631945623, 4029.0616250770177, 4029.261618208412, 4029.461611339807, 4029.661604471202, 4029.8615976025962, 4030.0615907339907, 4030.2615838653855, 4030.46157699678, 4030.661570128175, 4030.8615632595697, 4031.061556390964, 4031.2615495223586, 4031.461542653754, 4031.6615357851483, 4031.8615289165427, 4032.061522047938, 4032.2615151793325, 4032.461508310727, 4032.6615014421213, 4032.861494573516, 4033.0614877049106, 4033.261480836305, 4033.461473967701, 4033.6614670990953, 4033.8614602304897, 4034.0614533618846, 4034.261446493279, 4034.4614396246734, 4034.6614327560683, 4034.861425887463, 4035.0614190188576, 4035.2614121502525, 4035.461405281647, 4035.6613984130413, 4035.861391544436, 4036.0613846758306, 4036.2613778072255, 4036.461370938621, 4036.661364070015, 4036.8613572014096, 4037.0613503328045, 4037.261343464199, 4037.4613365955934, 4037.6613297269882, 4037.861322858383, 4038.0613159897775, 4038.261309121172, 4038.4613022525673, 4038.6612953839617, 4038.861288515356, 4039.0612816467515, 4039.261274778146, 4039.4612679095403, 4039.661261040935, 4039.8612541723296, 4040.061247303724, 4040.2612404351194, 4040.461233566514, 4040.661226697908, 4040.861219829303, 4041.0612129606975, 4041.261206092092, 4041.4611992234873, 4041.661192354882, 4041.8611854862766, 4042.0611786176714, 4042.261171749066, 4042.4611648804603, 4042.6611580118547, 4042.8611511432496],
    # [4017.031123675372, 4017.2311152677084, 4017.431106860045, 4017.6310984523816, 4017.8310900447186, 4018.031081637055, 4018.2310732293918, 4018.431064821729, 4018.6310564140654, 4018.831048006402, 4019.031039598739, 4019.2310311910755, 4019.431022783412, 4019.631014375749, 4019.8310059680857, 4020.0309975604223, 4020.2309891527593, 4020.430980745096, 4020.6309723374325, 4020.8309639297695, 4021.0309555221056, 4021.230947114442, 4021.4309387067797, 4021.6309302991162, 4021.830921891453, 4022.03091348379, 4022.2309050761264, 4022.430896668463, 4022.6308882607996, 4022.8308798531366, 4023.030871445473, 4023.2308630378097, 4023.430854630147, 4023.6308462224833, 4023.83083781482, 4024.030829407157, 4024.2308209994935, 4024.43081259183, 4024.630804184167, 4024.8307957765037, 4025.0307873688403, 4025.2307789611773, 4025.430770553514, 4025.6307621458504, 4025.8307537381875, 4026.030745330524, 4026.23073692286, 4026.4307285151976, 4026.630720107534, 4026.830711699871, 4027.030703292208, 4027.2306948845444, 4027.430686476881, 4027.6306780692175, 4027.8306696615546, 4028.030661253891, 4028.2306528462277, 4028.430644438565, 4028.6306360309018, 4028.830627623238, 4029.030619215575, 4029.2306108079115, 4029.430602400248, 4029.630593992585, 4029.8305855849217, 4030.0305771772582, 4030.2305687695953, 4030.430560361932, 4030.6305519542684, 4030.8305435466054, 4031.030535138942, 4031.2305267312786, 4031.4305183236156, 4031.630509915952, 4031.8305015082888, 4032.030493100626, 4032.2304846929624, 4032.430476285299, 4032.6304678776355, 4032.8304594699725, 4033.030451062309, 4033.2304426546457, 4033.430434246983, 4033.6304258393197, 4033.8304174316563, 4034.0304090239933, 4034.2304006163295, 4034.430392208666, 4034.630383801003, 4034.8303753933396, 4035.030366985676, 4035.2303585780132, 4035.43035017035, 4035.6303417626864, 4035.8303333550234, 4036.03032494736, 4036.2303165396966, 4036.430308132034, 4036.6302997243706, 4036.8302913167067, 4037.0302829090438, 4037.2302745013803, 4037.430266093717, 4037.6302576860535, 4037.8302492783905, 4038.030240870727, 4038.2302324630637, 4038.430224055401, 4038.6302156477377, 4038.8302072400743, 4039.0301988324113, 4039.230190424748, 4039.430182017084, 4039.630173609421, 4039.8301652017576, 4040.030156794094, 4040.230148386431, 4040.430139978768, 4040.6301315711044, 4040.8301231634414, 4041.030114755778, 4041.2301063481145, 4041.430097940452, 4041.6300895327886, 4041.830081125125, 4042.0300727174617, 4042.2300643097983, 4042.430055902135, 4042.6300474944715, 4042.8300390868085],
    # [4017.0525146620735, 4017.25250731938, 4017.4524999766863, 4017.652492633993, 4017.8524852913, 4018.0524779486063, 4018.252470605913, 4018.4524632632206, 4018.652455920527, 4018.8524485778335, 4019.0524412351406, 4019.252433892447, 4019.4524265497535, 4019.65241920706, 4019.852411864367, 4020.0524045216735, 4020.25239717898, 4020.452389836287, 4020.6523824935934, 4020.8523751509, 4021.052367808207, 4021.2523604655134, 4021.4523531228206, 4021.6523457801272, 4021.852338437434, 4022.0523310947406, 4022.2523237520472, 4022.4523164093534, 4022.65230906666, 4022.8523017239672, 4023.0522943812734, 4023.25228703858, 4023.4522796958877, 4023.652272353194, 4023.8522650105006, 4024.0522576678077, 4024.252250325114, 4024.4522429824206, 4024.6522356397277, 4024.852228297034, 4025.0522209543406, 4025.252213611647, 4025.452206268954, 4025.6521989262606, 4025.852191583567, 4026.052184240874, 4026.2521768981806, 4026.4521695554877, 4026.6521622127943, 4026.852154870101, 4027.0521475274077, 4027.2521401847143, 4027.452132842021, 4027.652125499327, 4027.8521181566343, 4028.0521108139405, 4028.252103471247, 4028.452096128555, 4028.652088785861, 4028.8520814431677, 4029.052074100475, 4029.252066757781, 4029.4520594150877, 4029.652052072395, 4029.852044729701, 4030.0520373870077, 4030.252030044315, 4030.452022701621, 4030.6520153589277, 4030.8520080162343, 4031.052000673541, 4031.2519933308477, 4031.451985988155, 4031.6519786454614, 4031.851971302768, 4032.0519639600748, 4032.2519566173814, 4032.451949274688, 4032.6519419319943, 4032.8519345893014, 4033.0519272466076, 4033.2519199039143, 4033.451912561222, 4033.651905218528, 4033.8518978758348, 4034.051890533142, 4034.251883190448, 4034.4518758477548, 4034.651868505062, 4034.851861162368, 4035.0518538196748, 4035.251846476982, 4035.451839134288, 4035.6518317915948, 4035.8518244489014, 4036.051817106208, 4036.2518097635148, 4036.451802420822, 4036.6517950781285, 4036.851787735435, 4037.051780392742, 4037.2517730500485, 4037.451765707355, 4037.6517583646614, 4037.8517510219685, 4038.051743679275, 4038.2517363365814, 4038.451728993889, 4038.651721651195, 4038.851714308502, 4039.051706965809, 4039.251699623115, 4039.451692280422, 4039.651684937729, 4039.851677595035, 4040.051670252342, 4040.251662909649, 4040.451655566955, 4040.651648224262, 4040.851640881569, 4041.051633538875, 4041.251626196182, 4041.451618853489, 4041.6516115107956, 4041.8516041681023, 4042.051596825409, 4042.2515894827156, 4042.4515821400223, 4042.6515747973285, 4042.8515674546356],
    # [4017.060093856534, 4017.2600868911777, 4017.4600799258214, 4017.6600729604647, 4017.860065995109, 4018.0600590297527, 4018.2600520643964, 4018.460045099041, 4018.660038133685, 4018.8600311683285, 4019.0600242029723, 4019.260017237616, 4019.4600102722598, 4019.660003306904, 4019.8599963415477, 4020.0599893761914, 4020.2599824108356, 4020.4599754454794, 4020.6599684801226, 4020.859961514767, 4021.0599545494106, 4021.2599475840543, 4021.459940618699, 4021.6599336533427, 4021.8599266879864, 4022.05991972263, 4022.259912757274, 4022.4599057919177, 4022.6598988265614, 4022.8598918612056, 4023.0598848958493, 4023.259877930493, 4023.4598709651373, 4023.659863999781, 4023.8598570344247, 4024.059850069069, 4024.2598431037127, 4024.4598361383564, 4024.6598291730006, 4024.859822207644, 4025.0598152422876, 4025.259808276932, 4025.4598013115756, 4025.6597943462193, 4025.8597873808635, 4026.0597804155072, 4026.2597734501505, 4026.459766484795, 4026.659759519439, 4026.8597525540827, 4027.059745588727, 4027.2597386233706, 4027.4597316580143, 4027.6597246926576, 4027.859717727302, 4028.0597107619456, 4028.2597037965893, 4028.459696831234, 4028.6596898658777, 4028.8596829005214, 4029.0596759351656, 4029.259668969809, 4029.4596620044526, 4029.659655039097, 4029.8596480737406, 4030.0596411083843, 4030.2596341430285, 4030.4596271776722, 4030.6596202123155, 4030.8596132469597, 4031.0596062816035, 4031.259599316247, 4031.459592350892, 4031.6595853855356, 4031.8595784201793, 4032.059571454823, 4032.259564489467, 4032.4595575241106, 4032.6595505587543, 4032.8595435933985, 4033.059536628042, 4033.259529662686, 4033.45952269733, 4033.659515731974, 4033.8595087666176, 4034.059501801262, 4034.2594948359056, 4034.4594878705493, 4034.6594809051935, 4034.859473939837, 4035.0594669744805, 4035.2594600091247, 4035.4594530437685, 4035.659446078412, 4035.8594391130564, 4036.0594321477, 4036.2594251823434, 4036.459418216988, 4036.659411251632, 4036.8594042862755, 4037.0593973209197, 4037.2593903555635, 4037.459383390207, 4037.659376424851, 4037.8593694594947, 4038.0593624941384, 4038.259355528782, 4038.459348563427, 4038.6593415980706, 4038.8593346327143, 4039.0593276673585, 4039.259320702002, 4039.4593137366455, 4039.6593067712897, 4039.8592998059335, 4040.059292840577, 4040.2592858752214, 4040.459278909865, 4040.6592719445084, 4040.8592649791526, 4041.0592580137964, 4041.25925104844, 4041.4592440830847, 4041.6592371177285, 4041.859230152372, 4042.059223187016, 4042.2592162216597, 4042.4592092563034, 4042.659202290947, 4042.8591953255914],
    # [4017.0299891243703, 4017.2299806602223, 4017.4299721960742, 4017.629963731926, 4017.829955267778, 4018.02994680363, 4018.229938339482, 4018.429929875335, 4018.629921411187, 4018.829912947039, 4019.0299044828907, 4019.2298960187427, 4019.4298875545946, 4019.629879090447, 4019.829870626299, 4020.029862162151, 4020.229853698003, 4020.429845233855, 4020.629836769707, 4020.829828305559, 4021.029819841411, 4021.2298113772626, 4021.4298029131155, 4021.6297944489675, 4021.8297859848194, 4022.029777520672, 4022.2297690565238, 4022.4297605923753, 4022.629752128227, 4022.8297436640796, 4023.0297351999316, 4023.2297267357835, 4023.4297182716364, 4023.629709807488, 4023.82970134334, 4024.029692879192, 4024.229684415044, 4024.429675950896, 4024.6296674867485, 4024.8296590226, 4025.029650558452, 4025.2296420943044, 4025.4296336301563, 4025.6296251660083, 4025.8296167018607, 4026.029608237712, 4026.229599773564, 4026.429591309417, 4026.629582845269, 4026.829574381121, 4027.0295659169733, 4027.229557452825, 4027.4295489886767, 4027.6295405245287, 4027.829532060381, 4028.029523596233, 4028.229515132085, 4028.4295066679374, 4028.6294982037894, 4028.8294897396413, 4029.0294812754937, 4029.2294728113457, 4029.4294643471976, 4029.6294558830496, 4029.8294474189015, 4030.0294389547535, 4030.229430490606, 4030.429422026458, 4030.6294135623093, 4030.8294050981617, 4031.0293966340137, 4031.2293881698656, 4031.4293797057185, 4031.6293712415704, 4031.829362777422, 4032.0293543132743, 4032.2293458491263, 4032.429337384978, 4032.62932892083, 4032.8293204566826, 4033.029311992534, 4033.229303528386, 4033.429295064239, 4033.629286600091, 4033.829278135943, 4034.029269671795, 4034.2292612076467, 4034.4292527434986, 4034.629244279351, 4034.829235815203, 4035.029227351055, 4035.2292188869073, 4035.429210422759, 4035.629201958611, 4035.829193494463, 4036.029185030315, 4036.229176566167, 4036.42916810202, 4036.6291596378715, 4036.8291511737234, 4037.029142709576, 4037.2291342454278, 4037.4291257812797, 4037.6291173171317, 4037.8291088529836, 4038.0291003888356, 4038.2290919246875, 4038.4290834605404, 4038.6290749963923, 4038.8290665322443, 4039.029058068096, 4039.229049603948, 4039.4290411398, 4039.6290326756525, 4039.8290242115045, 4040.029015747356, 4040.2290072832084, 4040.4289988190603, 4040.6289903549123, 4040.8289818907647, 4041.0289734266166, 4041.228964962468, 4041.428956498321, 4041.628948034173, 4041.828939570025, 4042.0289311058773, 4042.2289226417292, 4042.4289141775807, 4042.6289057134327, 4042.828897249285]
    # ]

    # y_fluxes = [
    # [1.0365711, 0.9768954, 0.98638314, 1.001783, 1.0047108, 1.0204816, 0.9795359, 0.95766217, 1.0053893, 1.0390307, 0.9932781, 1.0384886, 1.045276, 1.0022849, 1.0140483, 0.93227553, 1.0012269, 0.9763614, 1.0202705, 1.03607, 1.0308069, 1.0210824, 1.0076542, 1.0229312, 1.0374628, 1.0219605, 1.0179822, 1.031191, 1.0218502, 0.995002, 1.066809, 1.0149295, 1.08358, 0.9680812, 1.0192014, 0.95528525, 1.0120476, 0.9807692, 0.98886704, 0.98853487, 0.9480535, 1.0062481, 0.9423176, 0.96405756, 0.92859215, 0.97659135, 0.88732123, 0.8626976, 0.9183166, 0.8441385, 0.8239039, 0.82816553, 0.7657798, 0.80787337, 0.7541934, 0.7990375, 0.7505147, 0.7548348, 0.76056933, 0.77855223, 0.7613986, 0.77160466, 0.81129915, 0.8181417, 0.86761385, 0.8726302, 0.902131, 0.92498714, 0.9101526, 0.92961895, 0.9897279, 0.9506742, 0.95321476, 0.99821776, 1.0178407, 0.9933658, 1.0396487, 1.032047, 1.0306785, 1.0592979, 1.019957, 0.9561388, 1.0488002, 0.9985422, 0.95584714, 1.0275805, 1.0300206, 0.9729931, 1.0580631, 0.988706, 1.0029925, 1.0097492, 0.9788001, 1.0123173, 0.9877799, 1.0125675, 1.0637815, 1.0650324, 0.9977725, 1.0129892, 1.0451808, 1.0042682, 1.0176269, 0.9876863, 1.0313834, 1.0459179, 1.0226116, 1.0267103, 0.94959825, 0.9740749, 0.9902298, 1.0591068, 1.0074713, 1.0391608, 1.0276724, 0.9833921, 1.0175946, 0.9857064, 1.0431888, 1.0171081, 1.0356139, 1.0114903, 1.0019301, 1.0173764, 1.00125, 1.0334258, 1.0288585, 1.0063682, 1.0249637, 0.99701786],
    # [0.97011775, 1.0305183, 1.032309, 0.9867761, 0.9811406, 0.985959, 0.98548377, 1.0515476, 1.0349228, 1.0571538, 1.013771, 1.0451639, 1.0173583, 1.0281571, 1.0292904, 1.0396109, 1.0148294, 1.0105776, 1.0588888, 1.0496107, 1.019716, 1.0337692, 1.0158151, 1.0192244, 1.0500853, 1.0435739, 0.9974787, 0.98274976, 1.012843, 0.9830342, 1.0412837, 1.0122912, 0.949481, 1.0012771, 0.9677644, 0.9410719, 1.0194453, 0.9328845, 0.93537754, 0.9269576, 0.9272389, 0.9035089, 0.95555866, 0.91638815, 0.91071737, 0.9240134, 0.9157084, 0.96245134, 0.9281526, 0.9701013, 1.000895, 0.9718367, 0.9152908, 0.91405016, 0.95897883, 1.0036516, 0.95582086, 0.94139004, 0.9419555, 0.9799846, 0.92144483, 0.8980317, 0.89621174, 0.88571614, 0.8844947, 0.83739245, 0.8331504, 0.8192727, 0.85233474, 0.80918145, 0.83357334, 0.8524969, 0.8576101, 0.853852, 0.8922781, 0.88796747, 0.9002296, 0.9079178, 0.9229068, 0.9711517, 0.9684656, 0.9596181, 0.985619, 0.98558766, 1.001291, 0.9960128, 0.98295647, 0.9703761, 1.0535048, 0.9720864, 1.0302843, 1.0475457, 0.99570626, 0.98744464, 1.0103327, 1.0120775, 1.042057, 1.0398167, 0.9992239, 0.98293567, 1.0431694, 0.96530885, 0.97575206, 1.0093975, 1.010389, 1.0049753, 1.0220374, 1.0003477, 1.0415512, 1.0035727, 1.0636072, 1.0389409, 1.0203192, 1.0021188, 0.9703461, 0.99113125, 1.0194697, 0.99659586, 0.9713872, 0.97621083, 0.9706113, 1.0278566, 1.0135558, 0.9707083, 1.0404419, 0.99301994, 0.97377247, 1.0451472, 1.058964, 1.0139333],
    # [0.95530343, 1.0020808, 1.0469904, 1.0396913, 1.025194, 1.0700486, 1.0237552, 1.0383129, 0.9811945, 1.0632024, 0.98848796, 1.0206715, 1.025023, 0.98188174, 1.0376012, 1.0353575, 1.0075251, 1.0212331, 1.0248553, 1.0015751, 1.0057902, 1.0163251, 1.0287212, 0.97194964, 0.9897062, 1.0035253, 0.97681075, 0.9729947, 0.98854864, 0.9710496, 1.0165315, 0.9764483, 1.0161413, 0.9706739, 0.972617, 0.9690871, 0.9699416, 0.8871392, 0.913515, 0.95802766, 0.9571625, 0.9156809, 0.9093651, 0.86046803, 0.8026198, 0.8411327, 0.84465116, 0.7796615, 0.81914407, 0.8366865, 0.8516987, 0.8366314, 0.82412505, 0.85082674, 0.84197104, 0.84977084, 0.88429934, 0.9447425, 0.9669197, 0.9520885, 0.9234005, 0.9178586, 0.91373366, 0.8880646, 0.8806101, 0.88246053, 0.9046563, 0.94274354, 0.9238107, 0.93796647, 0.93765366, 0.90566826, 0.9416052, 0.95071113, 0.9376853, 0.9408436, 1.0066495, 1.0274764, 0.9681071, 0.9840492, 0.9883867, 0.98149246, 1.0076009, 1.0188286, 1.0100697, 1.0632002, 0.9941461, 1.000036, 1.0433987, 1.0150914, 1.0182315, 0.9424984, 1.0625135, 0.9585225, 0.9794065, 1.0077395, 0.9879925, 1.0136784, 1.0271329, 1.0254494, 1.0125592, 0.9900873, 1.0216737, 1.0024389, 1.0324026, 1.0183542, 1.0208915, 1.0687866, 1.032029, 0.9864168, 1.00522, 1.0266457, 1.01028, 1.0131234, 1.0126264, 1.0419556, 1.0153245, 1.0435573, 1.0419601, 1.0080413, 0.9924771, 0.97125345, 1.0194556, 1.0282047, 1.025353, 0.9672464, 1.0042803, 0.97416687, 1.0053865, 0.97449285],
    # [1.026424, 1.1392804, 1.0176027, 0.92712104, 0.9848659, 1.0292405, 1.0297757, 0.98180866, 1.00972, 1.0240983, 1.0750796, 1.0699902, 1.0693805, 1.0534731, 1.0375769, 1.0543935, 1.0149837, 0.98250043, 1.0164622, 1.0639033, 1.0587802, 0.9992037, 1.0577267, 1.0651157, 1.0177882, 1.0792288, 1.0176198, 1.0512284, 1.0205203, 0.9655673, 0.9630565, 1.0223382, 0.99315935, 1.000291, 1.0202674, 0.92717254, 0.91533357, 0.9586661, 1.0355749, 0.9680057, 0.92301476, 1.0213778, 0.95556074, 0.9821869, 0.8426362, 0.94450974, 0.84882003, 0.87861663, 0.7547542, 0.8294706, 0.83856106, 0.94214535, 0.9425041, 0.84850156, 0.882601, 0.8830405, 0.89564747, 0.88663256, 0.84998715, 0.898139, 0.9680941, 0.8679872, 0.9145096, 0.92210597, 0.8117006, 0.9109672, 0.8481284, 0.98784214, 0.9362053, 0.9586364, 0.97689295, 0.9475865, 0.883123, 0.94015265, 0.97941107, 0.94383854, 1.0004884, 1.0039219, 0.9313635, 0.9923585, 0.96628845, 1.0126202, 1.0113131, 1.0385442, 1.091165, 1.0064355, 1.0612043, 0.98327893, 1.0426421, 1.0689384, 0.965017, 0.974335, 1.0852014, 1.0845002, 1.0988073, 1.0481683, 1.0230733, 1.0471505, 1.0583085, 0.9824952, 1.0603001, 0.9984797, 0.9728503, 1.0726761, 1.0096534, 1.097258, 1.0836276, 1.021318, 1.0844324, 1.1472623, 1.0950618, 1.020356, 1.0434496, 1.0464745, 1.1115543, 0.99858075, 0.96518815, 1.0126077, 0.91829616, 1.057891, 1.0552108, 0.9818161, 0.9945642, 0.99403554, 0.966893, 1.0169847, 0.95671093, 1.0275836, 0.9969526, 0.9463117],
    # [1.0053939, 0.9699029, 1.0008448, 1.0061666, 0.9852719, 1.043723, 1.0558555, 1.0269719, 0.99038607, 1.018488, 1.010574, 1.0391306, 1.0018638, 1.0233217, 1.0409865, 1.0320332, 1.0151329, 1.0215657, 1.0610503, 1.0336599, 1.0370654, 0.9810114, 1.0417558, 1.0274202, 1.0286332, 1.0062119, 1.00518, 1.0396911, 0.9938716, 1.0033634, 1.0078957, 0.99539584, 0.99231684, 1.0598829, 0.9597232, 0.9686972, 0.9665276, 1.0444437, 0.97765213, 1.0018717, 0.98026806, 0.9848836, 0.96423835, 0.9833444, 0.9586694, 0.9635007, 0.94042075, 0.88200986, 0.897695, 0.8561885, 0.8570953, 0.8699853, 0.80102223, 0.79689157, 0.80810213, 0.7760512, 0.77406985, 0.799909, 0.78259706, 0.7577657, 0.7970236, 0.7826806, 0.75979555, 0.8050847, 0.8478229, 0.8681931, 0.8349736, 0.8642883, 0.9004191, 0.92118335, 1.0050763, 0.9992571, 0.97660774, 0.9402248, 1.0010667, 1.0043321, 1.0052642, 1.040995, 1.0278955, 0.97680384, 0.9880213, 1.0002406, 0.98094547, 0.9863615, 0.99373597, 0.9647836, 0.99555117, 0.9975856, 0.9780219, 1.0192041, 1.027393, 0.9907604, 0.97567254, 0.9914972, 1.0385785, 1.0460367, 1.0372077, 0.9503998, 1.0109466, 1.0033596, 1.061107, 1.0365574, 1.014221, 1.0390404, 1.024896, 1.0399278, 1.0325663, 1.032078, 0.9817865, 1.0223548, 0.98910743, 1.0184752, 1.0263975, 1.0068758, 0.9639284, 1.0269496, 1.0094719, 1.0096223, 0.994668, 1.0220337, 0.98077106, 0.9692075, 0.9778751, 1.0287042, 1.0164129, 0.98450214, 1.0091518, 1.0184892, 1.05225, 0.9762847]
    # ]
    print(len(y_fluxes[0]), len(x_waves))

    # Initial guess for the central wavelength and width
    cen_ini = line+shift
    wid_ini = lines_dic[line]['wid_ini']
    # amp_ini = 0.9 - min([flux.min() for flux in y_fluxes])
    
    def sb2_model(wavelengths=None, fluxes=None):
        # Model parameters
        continuum = npro.sample('continuum', dist.Normal(1, 0.1))
        amp1 = npro.sample('amp1', dist.Uniform(0, 1))
        amp2 = npro.sample('amp2', dist.Uniform(0, 1))
        wid1 = npro.sample('wid1', dist.Uniform(0.5, 10))
        wid2 = npro.sample('wid2', dist.Uniform(0.5, 10))
        # amp1 = npro.sample('amp1', dist.Normal(amp_ini, amp_ini*0.2))
        # amp2 = npro.sample('amp2', dist.Normal(amp_ini*0.6, amp_ini*0.6*0.2))
        # wid1 = npro.sample('wid1', dist.Normal(wid_ini, wid_ini*0.2))
        # wid2 = npro.sample('wid2', dist.Normal(wid_ini, wid_ini*0.2))
        # logsig = npro.sample('logsig', dist.Normal(-2, 1))
        # cen = cen_ini
        cen = (wavelengths[0].max() + wavelengths[0].min())/2
        delta_cen = (wavelengths[0].max() - wavelengths[0].min()) / 8
        # print('centre =', cen, '+/-', delta_cen)
        n_epoch, n_wavelength = fluxes.shape

        # Model definition
        with npro.plate('epoch=1..n', n_epoch, dim=-2):
            mean1 = npro.sample('mean1', dist.Normal(cen, delta_cen))
            mean2 = npro.sample('mean2', dist.Normal(cen, delta_cen))
            with npro.plate('wavelength=1..k', n_wavelength, dim=-1):
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
                num_warmup=10000, 
                num_samples=10000)

    mcmc.run(rng_key, wavelengths=jnp.array(x_waves), fluxes=jnp.array(y_fluxes))
    mcmc.print_summary()
    trace = mcmc.get_samples()

    n_lines = 100
    n_epochs = len(x_waves)
    for epoch, ax in zip(range(n_epochs), axes.ravel()):
        ax.plot(x_waves[epoch], trace['pred'][-n_lines:, epoch, :].T, rasterized=True, color='C4', alpha=0.1)
        ax.plot(x_waves[epoch], trace['pred_1'][-n_lines:, epoch, :].T, rasterized=True, color='C0', alpha=0.1)
        ax.plot(x_waves[epoch], trace['pred_2'][-n_lines:, epoch, :].T, rasterized=True, color='C1', alpha=0.1)
        ax.plot(x_waves[epoch], y_fluxes[epoch], color='k', lw=0.5)
    plt.savefig(path+str(line)+'_fits_SB2_.png', bbox_inches='tight', dpi=150)
    plt.close()


def fit_sb1(line, wave, flux, ferr, lines_dic, Hlines, neblines, doubem, shift):  
    """
    Fits a Gaussian or Lorentzian profile to a spectral line.

    Parameters:
    line (int): Label (wavelength) of the line to fit.
    wave (array-like): Spectrum wavelength.
    flux (array-like): Spectrum flux.
    ferr (array-like): The flux errors.
    lines_dic (dict): A dictionary containing information about the lines.
    Hlines (array-like): A list of lines to be fitted with a Lorentzian profile.
    neblines (array-like): A list of lines to be fitted with a nebular emission model.
    doubem (array-like): A list of lines to be fitted with two nebular emission models.
    shift (float): The shift to apply to the central wavelength of the line.

    Returns:
    result (ModelResult): The result of the fit.
    x_wave (array-like): The wavelengths used in the fit.
    y_flux (array-like): The fluxes used in the fit.
    wave_region (array-like): A boolean array indicating the wavelengths used in the fit.
    """
    wave_region = np.logical_and(wave > lines_dic[line]['region'][0], wave < lines_dic[line]['region'][1])

    x_wave = wave[wave_region]
    y_flux = flux[wave_region]

    cen_ini = line+shift
    wid_ini = lines_dic[line]['wid_ini']

    pars = Parameters()
    gauss1 = Model(gaussian, prefix='g1_')
    loren1 = Model(lorentzian, prefix='l1_')
    nebem1 = Model(nebu, prefix='neb1_')
    nebem2 = Model(nebu, prefix='neb2_')
    cont = models.LinearModel(prefix='continuum_')

    # Define parameters and set initial values
    pars.update(cont.make_params())
    pars['continuum_slope'].set(0, vary=False)
    pars['continuum_intercept'].set(1, min=0.9)

    if line in Hlines:
        prefix = 'l1'
        indiv_mod = loren1
    else:
        prefix = 'g1'
        indiv_mod = gauss1

    pars.update(indiv_mod.make_params())
    pars[f'{prefix}_amp'].set(1.2-y_flux.min(), min=0.01, max=2. )
    pars[f'{prefix}_wid'].set(wid_ini, min=1, max=11)
    pars[f'{prefix}_cen'].set(cen_ini, min=cen_ini-4, max=cen_ini+4)
    mod = indiv_mod + cont
    # if line==5890:
    #     pars.add('g1_wid', value=wid_ini, min=0.1, max=4.)
    #     pars.update(gauss2.make_params())
    #     pars.add('g2_cen', value=cen_ini+6., vary=True)
    #     pars.add('g2_amp', 0.8*(1-y_flux.min()), min=0.05, max=0.5)
    #     pars.add('g2_wid', value=wid_ini-1.5, min=0.1, max=4.)
    #     mod = gauss1 + gauss2 + cont

    # Define a list of nebular emission models
    nebem_models = [nebem1, nebem2]
    # Loop over the nebular emission models
    for i, nebem in enumerate(nebem_models, start=1):
        if line in neblines:
            pars.update(nebem.make_params())
            # if y_flux.max() < 1.2:
            #     pars['neb_amp'].set((1-y_flux.min())/2, min=0.001, max=0.5)
            # else:
            #     pars['neb_amp'].set((y_flux.max()-y_flux.min())/1.1, min=0.05)#, max=65)
            pars[f'neb{i}_amp'].set(y_flux.max()-y_flux.min(), min=0.01)
            pars[f'neb{i}_wid'].set(1, min=0.05, max=2)
            if i == 2:
                pars[f'neb{i}_cen'].set(cen_ini+0.2)
            else:
                pars[f'neb{i}_cen'].set(cen_ini-0.2)
            mod += nebem
            if i == 2 and line not in doubem:
                break

    result = mod.fit(y_flux, pars, x=x_wave, weights=1/ferr[wave_region])

    return result, x_wave, y_flux, wave_region

def sb2_results_to_file(result, wavelengths, names, line, writer, csvfile):
    for i in range(len(wavelengths)):
        results_dict = {}
        for j in range(1, 3):
            prefix = f'cen{j}_{i+1}'
            if prefix in result.params:
                component_dict = {
                    'epoch': names[i],
                    'line': line,
                    f'cen{j}': result.params[f'{prefix}'].value,
                    f'cen{j}_er': result.params[f'{prefix}'].stderr,
                }
                results_dict.update(component_dict)

        # Add common parameters
        common_params = ['amp1', 'wid1', 'amp2', 'wid2']
        for param in common_params:
            if param in result.params:
                results_dict[param] = result.params[param].value
                results_dict[f'{param}_er'] = result.params[param].stderr

        # If writer is None, this is the first dictionary, so get the fieldnames from its keys
        if writer is None:
            fieldnames = results_dict.keys()
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
        writer.writerow(results_dict)

    return writer

def SLfit(spectra_list, path, lines, file_type='fits', plots=True, balmer=True, neblines=[4102, 4340], doubem=[], SB2=False, shift=0, use_init_pars=False):
    '''
    spectra_list
    path
    lines:        lines to be fitted
    file_type
    plots
    balmer
    neblines
    doubem
    SB2
    shift:        wavelength shift to be applied for initial guess used for the fit
    '''

    if len(spectra_list) == 1:
        print("\n   WARNING: There is only 1 epoch to compute RVs.")
        return    
    # doubem = doubem[0]
    Hlines = [4102, 4340, 6562]

    print('*** SB2 set to: ', SB2, ' ***\n')
    
    wavelengths, fluxes, f_errors, names, jds = read_spectra(spectra_list, path, file_type)

    path = setup_star_directory_and_save_jds(names, jds, path)

    lines_dic = setup_line_dictionary()

    print('these are the lines: ', lines)

    (
        cen1, cen1_er, amp1, amp1_er, wid1, wid1_er, 
        cen2, cen2_er, amp2, amp2_er, wid2, wid2_er, 
        dely, sdev, results, comps, delta_cen, chisqr
    ) = initialize_fit_variables(lines)

    with open(path + 'fit_values.csv', 'w', newline='') as csvfile:
        writer = None

        for i, line in enumerate(lines):
            print('Fitting line ', line)
            if plots:
                fig, axes = setup_fits_plots(wavelengths)
            else:
                axes = [None] * len(wavelengths)
            if SB2:
                result, x_wave, y_flux = fit_sb2_probmod(line, wavelengths, fluxes, f_errors, lines_dic, Hlines, neblines, shift, axes, path)
                writer = sb2_results_to_file(result, wavelengths, names, line, writer, csvfile)
                
                # if plots:
                #     for i, (x_vals, y_vals, ax) in enumerate(zip(x_wave, y_flux, axes)):
                #         ax.plot(x_vals, y_vals, 'k-', label=str(names[i].replace('./','').replace('_',' ')))
                #         model_fit = double_gaussian_with_continuum(x_vals, result.params['amp1'].value, result.params[f'cen1_{i+1}'].value, result.params['wid1'].value, 
                #                         result.params['amp2'].value, result.params[f'cen2_{i+1}'].value, result.params['wid2'].value, result.params['cont'].value)
                #         component1 = gaussian(x_vals, result.params['amp1'].value, result.params[f'cen1_{i+1}'].value, result.params['wid1'].value)
                #         component2 = gaussian(x_vals, result.params['amp2'].value, result.params[f'cen2_{i+1}'].value, result.params['wid2'].value)
                #         ax.plot(x_vals, component1 + result.params['cont'].value, '--', c='limegreen', label=f'Component 1 - Epoch {i+1}')
                #         ax.plot(x_vals, component2 + result.params['cont'].value, '--', c='dodgerblue', label=f'Component 2 - Epoch {i+1}')
                #         ax.plot(x_vals, model_fit, 'r-', label=f'Fit {i+1}')
                #         # if line in neblines:
                #         #     ax.plot(x_wave, component['continuum_']+component['neb_'], '--', zorder=3, c='orange', lw=2)#, label='nebular emision')
                #         # if dely is not None:
                #         #     ax.fill_between(x_wave, result.best_fit-dely, result.best_fit+dely, zorder=2, color="#ABABAB", alpha=0.5)
                #         # ax.legend(loc='upper center',fontsize=10, handlelength=0, handletextpad=0.4, borderaxespad=0., frameon=False)
                #         # ax.legend(fontsize=14) 
                #         # ax.set_ylim(0.9*y_flux.min(), 1.1*y_flux.max())
                #     fig.supxlabel('Wavelength [\AA]', size=22)
                #     fig.supylabel('Flux', size=22)
                #     plt.savefig(path+str(line)+'_fits_SB2_.png', bbox_inches='tight', dpi=150)
                #     plt.close()
            else:
                for wave, flux, ferr, name, ax in zip(wavelengths, fluxes, f_errors, names, axes):
                    result, x_wave, y_flux, wave_region = fit_sb1(line, wave, flux, ferr, lines_dic, Hlines, neblines, doubem, shift)
                    results[i].append(result)
                    chisqr[i].append(result.chisqr)
                    if line in neblines or SB2==True:
                        component = result.eval_components(result.params, x=x_wave)
                        comps[i].append(component)
                    else:
                        component = None
                        comps[i].append(component)
                    with open(path + str(line) + '_stats.txt', 'w') as out_file:
                        out_file.write(name+'\n')
                        out_file.write('-' * len(name.strip()) + '\n')
                        out_file.write(result.fit_report()+'\n\n')

                    if 'g1_cen' in result.params:
                        prefix = 'g1'
                    elif 'l1_cen' in result.params:
                        prefix = 'l1'
                    else:
                        raise ValueError("Unexpected fit type")

                    results_dict = {
                        'epoch': name,
                        'line': line,
                        'cen1': result.params[f'{prefix}_cen'].value,
                        'cen1_er': result.params[f'{prefix}_cen'].stderr,
                        'amp1': result.params[f'{prefix}_amp'].value,
                        'amp1_er': result.params[f'{prefix}_amp'].stderr,
                        'wid1': result.params[f'{prefix}_wid'].value,
                        'wid1_er': result.params[f'{prefix}_wid'].stderr,
                    }
                    if results_dict['cen1_er'] is None or results_dict['amp1_er'] is None:
                        print('errors computation failed for line ', line, 'epoch ', j+1)
                    if results_dict['cen1_er'] != 0 and results_dict['cen1_er'] is not None:
                        dely = result.eval_uncertainty(sigma=3)
                    else:
                        dely = None

                    # If writer is None, this is the first dictionary, so get the fieldnames from its keys
                    if writer is None:
                        fieldnames = results_dict.keys()
                        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                        writer.writeheader()
                    writer.writerow(results_dict)



                    '''
                    Plotting 1st fits of the lines
                    '''
                    if plots:
                        xx = np.linspace(min(x_wave), max(x_wave), 500)
                        result_new = result.eval(x=xx)
                        init_new = result.eval(params=result.init_params, x=xx)
                        ax.plot(x_wave, y_flux, 'k-', lw=3, ms=4, zorder=1, label=str(name.replace('./','').replace('_',' ')))
                        ax.plot(xx, init_new, '--', c='grey', zorder=5)
                        ax.plot(xx, result_new, 'r-', lw=2, zorder=4)

                        if component is not None:
                            if line in Hlines:
                                ax.plot(x_wave, component['continuum_']+component['l1_'], '--', zorder=3, c='limegreen', lw=2)#, label='spectral line')
                            if SB2==True and not line in Hlines:
                                ax.plot(x_wave, component['continuum_']+component['g1_'], '--', zorder=3, c='blue', lw=2)#, label='spectral line')
                                ax.plot(x_wave, component['continuum_']+component['g2_'], '--', zorder=3, c='blue', lw=2)#, label='spectral line')
                            if line in neblines:
                                ax.plot(x_wave, component['continuum_']+component['neb_'], '--', zorder=3, c='orange', lw=2)#, label='nebular emision')
                            if line in neblines and not line in Hlines and SB2==False:
                                ax.plot(x_wave, component['continuum_']+component['g1_'], '--', zorder=3, c='limegreen', lw=2)#, label='spectral line')

                        if dely is not None:
                            ax.fill_between(x_wave, result.best_fit-dely, result.best_fit+dely, zorder=2, color="#ABABAB", alpha=0.5)

                        # ax.legend(loc='upper center',fontsize=10, handlelength=0, handletextpad=0.4, borderaxespad=0., frameon=False)

                        ax.set_ylim(0.9*y_flux.min(), 1.1*y_flux.max())
                if plots:        
                    fig.supxlabel('Wavelength', size=22)
                    fig.supylabel('Flux', size=22)
                    if SB2==True:
                        plt.savefig(path+str(line)+'_fits_0_.png', bbox_inches='tight', dpi=150)
                    else:
                        plt.savefig(path+str(line)+'_fits.png', bbox_inches='tight', dpi=150)
                    plt.close()

    return path

import copy
from datetime import date
from iteration_utilities import duplicates, unique_everseen

def sinu(x, A, w, phi, h):
    "Sinusoidal: sinu(data, amp, freq, phase, height)"
    return A*np.sin(w*(x-phi))+h

def outlier_killer(data, thresh=2, print_output=True):
    ''' returns the positions of elements in a list that
        are not outliers. Based on a median-absolute-deviation
        (MAD) test '''
    # diff = abs(data - np.nanmedian(data))
    diff = data - np.nanmedian(data)
    # diff = data - np.nanmean(data)
    if print_output==True:
        print('     mean(x)        =', f'{np.nanmean(data):.3f}')
        print('     median(x)        =', f'{np.nanmedian(data):.3f}')
        print('     x-median(x)      =', [f'{x:.3f}' for x in diff])
        print('     abs(x-median(x)) =', [f'{abs(x):.3f}' for x in diff])
    mad = 1.4826*np.nanmedian(abs(diff))
    if print_output==True:
        print('     mad =', f'{mad:.3f}')
        print('     abs(x-median(x)) <', (f'{thresh*mad:.3f}'), '(thresh*mad) =', [x<thresh*mad for x in diff])
    inliers, outliers = [], []
    for i in range(len(data)):
        if diff[i] < thresh*mad:
            inliers.append(i)
        else:
            outliers.append(i)
    return inliers, outliers

def weighted_mean(data, errors):
    weights = [1/(dx**2) for dx in errors]
    mean = sum([wa*a for a,wa in zip(data, weights)])/sum(weights)
    mean_err = np.sqrt(sum( [(da*wa)**2 for da,wa in zip(errors, weights)] ))/sum(weights)
    return mean, mean_err

def getrvs(fit_values, path, JDfile, balmer=False, SB2=False, use_lines=None, lines_ok_thrsld=2, epochs_ok_thrsld=2, \
        min_sep=2, print_output=True, random_eps=False, rndm_eps_n=29, rndm_eps_exc=[], plots=True, \
        err_type='wid', neps_table=False, eot_from_table=False, rm_epochs=True):
    '''
    
    '''
    
    import warnings
    warnings.filterwarnings("ignore", message="Font family ['serif'] not found. Falling back to DejaVu Sans.")
    warnings.filterwarnings("ignore", message="Mean of empty slice")
    '''
    Computation of radial velocities and L-S test
    '''
    # ti=time.time()
    fecha = str(date.today()) # '2017-12-26'

    if print_output==True:
        print('\n')
        print( '*******************************************************************************' )
        print( '******************                RV Analisys                ******************' )
        print( '*******************************************************************************\n' )
        print('epochs_ok_thrsld =', epochs_ok_thrsld)
        print('min_sep =', min_sep, '\n')


    if print_output==True:
        print('*** SB2 set to: ', SB2, ' ***\n')
    # Reading data from fit_states.csv
    # print(path+fit_values)
    df_SLfit = pd.read_csv(path+fit_values)

    x1, dx1, y1, dy1, z1, dz1, lines, chi2, epoch = \
                                                df_SLfit['cen1'], df_SLfit['cen1_er'], df_SLfit['amp1'], \
                                                df_SLfit['amp1_er'], df_SLfit['wid1'], df_SLfit['wid1_er'], \
                                                df_SLfit['line'], df_SLfit['chisqr'], df_SLfit['Epoch']
    if SB2==True:
        x2, dx2, y2, dy2, z2, dz2 = df_SLfit['cen2'], df_SLfit['cen2_er'], df_SLfit['amp2'], \
                                    df_SLfit['amp2_er'], df_SLfit['wid2'], df_SLfit['wid2_er'], \

    lines = sorted(list(unique_everseen(duplicates(lines))))
    print('lines from gfit table:', lines)
    if print_output==True:
        print( '*** Calculating % errors ***')
        print( '----------------------------')
    cen1_percer, cen2_percer = [[] for i in range(len(lines))], [[] for i in range(len(lines))]
    amp1_percer, amp2_percer = [[] for i in range(len(lines))], [[] for i in range(len(lines))]
    wid1_percer, wid2_percer = [[] for i in range(len(lines))], [[] for i in range(len(lines))]
    cen_stddev = [[] for i in range(len(lines))]
    amp_stddev = [[] for i in range(len(lines))]
    wid_stddev = [[] for i in range(len(lines))]
    for i in  range(len(lines)):                    # i: num of lines
        for j in range(len(df_SLfit)):                    # j: length of fit_states.csv (epochs x lines)
            if df_SLfit['line'][j]==lines[i]:
                if dx1[j]==0:
                    print(df_SLfit['line'][j], 'cen ', j, 'error = 0')
                    sys.exit()
                    cen1_percer[i].append(40)
                if dy1[j]==0:
                    print(df_SLfit['line'][j], 'amp ', j, 'error = 0')
                    sys.exit()
                    amp1_percer[i].append(40)
                if dz1[j]==0:
                    print(df_SLfit['line'][j], 'wid ', j, 'error = 0')
                    sys.exit()
                    wid1_percer[i].append(40)
                if dx1[j]!=0:
                    cen1_percer[i].append( abs(100*dx1[j]/x1[j])) # dim: n_lines x n_epochs
                if dy1[j]!=0:
                    amp1_percer[i].append( abs(100*dy1[j]/y1[j]))
                if dz1[j]!=0:
                    wid1_percer[i].append( abs(100*dz1[j]/z1[j]))
                cen_stddev[i].append(x1[j])
                amp_stddev[i].append(y1[j])
                wid_stddev[i].append(z1[j])
                if SB2==True:
                    if dx2[j]==0:
                        print(df_SLfit['line'][j], 'error = 0')
                        # sys.exit()
                        cen2_percer[i].append(40)
                    if dy2[j]==0:
                        print(df_SLfit['line'][j], 'error = 0')
                        # sys.exit()
                        amp2_percer[i].append(40)
                    if dz2[j]==0:
                        print(df_SLfit['line'][j], 'error = 0')
                        # sys.exit()
                        wid2_percer[i].append(40)
                    if dx2[j]!=0:
                        cen2_percer[i].append( abs(100*dx2[j]/x2[j])) # dim: n_lines x n_epochs
                    if dy2[j]!=0:
                        amp2_percer[i].append( abs(100*dy2[j]/y2[j]))
                    if dz2[j]!=0:
                        wid2_percer[i].append( abs(100*dz2[j]/z2[j]))
    if print_output==True:
        print( '\n')
        print( '*** Computing Radial velocities ***')
        print( '-----------------------------------')
    c = 2.998*pow(10,5) # km/s
    nlines = len(lines)

    lambda_shift1, lambda_shift2 = [[] for i in range(nlines)], [[] for i in range(nlines)]
    lambda_shift1_er, lambda_shift2_er = [[] for i in range(nlines)], [[] for i in range(nlines)]
    for i in  range(nlines):                    # i: num of best lines
        for j in range(len(df_SLfit)):                    # j: length of fit_states.csv
            if df_SLfit['line'][j]==lines[i]:
                lambda_shift1[i].append(x1[j])
                lambda_shift1_er[i].append(dx1[j])
                if SB2==True:
                    lambda_shift2[i].append(x2[j])
                    lambda_shift2_er[i].append(dx2[j])
    nepochs = len(lambda_shift1[0])
    # All wavelengths in air
    rv_dict =  {4009: [4009.2565, 0.00002], 4026: [4026.1914, 0.0010],
                4102: [4101.734, 0.006],    4121: [4120.8154, 0.0012],
                4128: [4128.07, 0.10],      4131: [4130.89, 0.10],
                4144: [4143.761, 0.010],    4267: [4267.258, 0.007], 
                4340: [4340.472, 0.006],    4388: [4387.9296, 0.0006],  
                4471: [4471.4802, 0.0015],  4481: [4481.130, 0.010],    
                4542: [4541.591, 0.010],    4553: [4552.62, 0.10], 
                4713: [4713.1457, 0.0006],      
                4861: [4861.35, 0.05],      4922: [4921.9313, 0.0005],
                5412: [5411.52, 0,10],      5876: [5875.621, 0.010],
                5890: [5889.951, 0.00003],  6562: [6562.79, 0.030],
                6678: [6678.151, 0.010],    7774: [7774.17, 0,10]
                }

    lambda_r, lambda_r_er = [], []
    for line in lines:
        # print(line)
        if line in rv_dict.keys():
            # print(line)
            lambda_r.append(rv_dict[line][0])
            lambda_r_er.append(rv_dict[line][1])

    #lambda_r = [4026.275, 4143.759, 4199.830, 4387.928, 4471.477, 4541.590, 4552.654]
    #lambda_rest  = [lambda_r[y] for y in best_lines_index]
    lambda_rest  = lambda_r
    #print(lambda_rest)
    print(len(lambda_rest))
    rvs1, rvs1_er  = [[] for i in range(nlines)], [[] for i in range(nlines)]
    rvs2, rvs2_er  = [[] for i in range(nlines)], [[] for i in range(nlines)]
    for i in range(nlines): # num of lines
        print('computing RV of line', lines[i])
        for j in range(nepochs): # num of epochs
            dlambda1 = lambda_shift1[i][j] - lambda_rest[i]
            dlambda1_er = np.sqrt( lambda_shift1_er[i][j]**2 + lambda_r_er[i]**2 )
            rvs1[i].append( dlambda1*c/lambda_rest[i]  )
            rvs1_er[i].append( np.sqrt( (dlambda1_er/dlambda1)**2 + (lambda_r_er[i]/lambda_rest[i])**2 )*rvs1[i][j] )
            if SB2==True:
                dlambda2 = abs(lambda_shift2[i][j] - lambda_rest[i])
                dlambda2_er = np.sqrt( lambda_shift2_er[i][j]**2 + lambda_r_er[i]**2 )
                rvs2[i].append( dlambda2*c/lambda_rest[i]  )
                rvs2_er[i].append( np.sqrt( (dlambda2_er/dlambda2)**2 + (lambda_r_er[i]/lambda_rest[i])**2 )*rvs2[i][j] )
    if print_output==True:
        print( '\n')
        print( '*** Choosing the best lines ***')
        print( '-------------------------------')

    # errors plots
    # bins=range(0,800,10)
    # if plots==True:
    #     if not os.path.exists(path+'errors'):
    #         os.makedirs(path+'errors')
    #     for i in range(nlines):
    #         fig, ax = plt.subplots(figsize=(8, 6))
    #         fig.subplots_adjust(left=0.11, right=0.91, top=0.96, bottom=0.11)
    #         y, x, _ = plt.hist(wid1_percer[i], bins=bins, histtype='stepfilled', density=False, linewidth=0.5)
    #         plt.axvline(np.nanmean(wid1_percer[i]), c='orange', ls='--', label='mean='+str(f'{np.nanmean(wid1_percer[i]):.1f}'))
    #         plt.axvline(np.nanmedian(wid1_percer[i]), c='blue', ls='--', label='median='+str(f'{np.nanmedian(wid1_percer[i]):.1f}'))
    #         plt.legend()
    #         plt.savefig(path+'errors/wid_histo_'+str(lines[i])+'.pdf')
    #         plt.close()
    #     for i in range(nlines):
    #         fig, ax = plt.subplots(figsize=(8, 6))
    #         fig.subplots_adjust(left=0.11, right=0.91, top=0.96, bottom=0.11)
    #         y, x, _ = plt.hist(amp1_percer[i], bins=bins, histtype='stepfilled', density=False, linewidth=0.5)
    #         plt.axvline(np.nanmean(amp1_percer[i]), c='orange', ls='--', label='mean='+str(f'{np.nanmean(amp1_percer[i]):.1f}'))
    #         plt.axvline(np.nanmedian(amp1_percer[i]), c='blue', ls='--', label='median='+str(f'{np.nanmedian(amp1_percer[i]):.1f}'))
    #         plt.legend()
    #         plt.savefig(path+'errors/amp_histo_'+str(lines[i])+'.pdf')
    #         plt.close()
    #     for i in range(nlines):
    #         fig, ax = plt.subplots(figsize=(8, 6))
    #         fig.subplots_adjust(left=0.11, right=0.91, top=0.96, bottom=0.11)
    #         y, x, _ = plt.hist(cen1_percer[i], bins=bins, histtype='stepfilled', density=False, linewidth=0.5)
    #         plt.axvline(np.nanmean(cen1_percer[i]), c='orange', ls='--', label='mean='+str(f'{np.nanmean(cen1_percer[i]):.1f}'))
    #         plt.axvline(np.nanmedian(cen1_percer[i]), c='blue', ls='--', label='median='+str(f'{np.nanmedian(cen1_percer[i]):.1f}'))
    #         plt.legend()
    #         plt.savefig(path+'errors/cen_histo_'+str(lines[i])+'.pdf')
    #         plt.close()

    # print table with means and medians
    if print_output==True:
        for i in range(nlines):
            if i == 0:
                print(' Primary:                ', str(lines[i]), end='')
            elif i<range(nlines)[-1]:
                print('     ', str(lines[i]), end='')
            else:
                print('     ', str(lines[i]))
        print('   mean(wid1_percer)  ', [str(f'{np.nanmean(x):6.3f}') for x in wid1_percer])
        print('   median(wid1_percer)', [str(f'{np.nanmedian(x):6.3f}') for x in wid1_percer])
        print('   mean(cen1_percer)  ', [str(f'{np.nanmean(x):6.3f}') for x in cen1_percer])
        print('   median(cen1_percer)', [str(f'{np.nanmedian(x):6.3f}') for x in cen1_percer])
        # print('   mean(amp1_percer)', [str(f'{np.nanmean(x):.3f}') for x in amp1_percer])
        # print('   median(amp1_percer)', [str(f'{np.nanmedian(x):.3f}') for x in amp1_percer])
        if SB2==True:
            print(' Secondary:')
            print('   mean(wid2_percer)  ', [str(f'{np.nanmean(x):6.3f}') for x in wid2_percer])
            print('   median(wid2_percer)', [str(f'{np.nanmedian(x):6.3f}') for x in wid2_percer])
            print('   mean(cen2_percer)  ', [str(f'{np.nanmean(x):6.3f}') for x in cen2_percer])
            print('   median(cen2_percer)', [str(f'{np.nanmedian(x):6.3f}') for x in cen2_percer])
            # print('   mean(amp2_percer)', [np.nanmean(x) for x in amp2_percer])
            # print('   median(amp2_percer)', [np.nanmedian(x) for x in amp2_percer])

    '''
    Writing stats to file rv_stats
    '''
    out = open(path+'rv_stats.txt', 'w')
    out.write(' ********************************\n')
    out.write('  Statistical analysis of errors \n')
    out.write(' ********************************\n')
    out.write('\n')
    out.write(' Mean of percentual errors'+'\n')
    out.write('   Lines '+'  amp1  '+'  wid1  '+' cen1 '+'  |  '+'  amp2  '+'  wid2  '+' cen2 '+'\n')
    out.write('   -------------------------------------------------------'+'\n')
    for i in range(nlines):
        out.write('   '+str(lines[i])+': '+str(f'{np.nanmean(amp1_percer[i]):7.3f}')+' '+str(f'{np.nanmean(wid1_percer[i]):7.3f}')+'  '+str(f'{np.nanmean(cen1_percer[i]):5.3f}')+'  |  '+\
                                            str(f'{np.nanmean(amp2_percer[i]):7.3f}')+' '+str(f'{np.nanmean(wid2_percer[i]):7.3f}')+'  '+str(f'{np.nanmean(cen2_percer[i]):5.3f}')+'\n')
    out.write('\n')
    out.write(' Median of percentual errors'+'\n')
    out.write('   Lines '+'  amp1  '+'  wid1  '+' cen1 '+'  |  '+'  amp2  '+'  wid2  '+' cen2 '+'\n')
    out.write('   -------------------------------------------------------'+'\n')
    for i in range(nlines):
        out.write('   '+str(lines[i])+': '+str(f'{np.nanmedian(amp1_percer[i]):7.3f}')+' '+str(f'{np.nanmedian(wid1_percer[i]):7.3f}')+'  '+str(f'{np.nanmedian(cen1_percer[i]):5.3f}')+'  |  '+\
                                            str(f'{np.nanmedian(amp2_percer[i]):7.3f}')+' '+str(f'{np.nanmedian(wid2_percer[i]):7.3f}')+'  '+str(f'{np.nanmedian(cen2_percer[i]):5.3f}')+'\n')
    '''
    Selecting the linestyle
    '''
    if not use_lines:
        if print_output==True:
            print('   --------------------------------------------')
            print('   Applying outlier_killer to remove bad lines:')
            # print('   # use_lines is empty')
        # print(len(wid1_percer), wid1_percer)
        best_lines_index, rm_lines_idx = outlier_killer([np.nanmedian(x) for x in wid1_percer], lines_ok_thrsld, print_output=print_output)
        # best_lines_index, rm_lines_idx = outlier_killer([np.nanmedian(x) for x in cen1_percer], lines_ok_thrsld, print_output=print_output)
        best_lines = [lines[x] for x in best_lines_index]
        if print_output==True:
            print('\n   these are the best lines: ', best_lines)
    elif use_lines=='from_table':
        # all_lines = [4009, 4026, 4102, 4144, 4200, 4340, 4388, 4471, 4481, 4542, 4553]
        if print_output==True:
            print('   -----------------------------------')
            print('   Selecting lines from table:')
        # for line in lambda_shift1:
        print(lines)
        table_lines = [x for x in lines if df_Bstars[str(x)][star_ind]=='x']
        best_lines_index = [i for i,x in enumerate(lambda_r) if any(round(x)==line for line in table_lines)]
        rm_lines_idx = [i for i,x in enumerate(lambda_r) if all(round(x)!=line for line in table_lines)]
        best_lines = [lines[x] for x in best_lines_index]
        if print_output==True:
            print('\n   these are the best lines: ', best_lines)
    else:
        if print_output==True:
            print('   -----------------------------------')
            print('   Selecting lines determined by user:')
            # print('   # use_lines is NOT empty')
        # print('lambda_r: ', lambda_r)
        best_lines_index = [i for i,x in enumerate(lambda_r) if any(round(x)==line for line in use_lines)]
        rm_lines_idx = [i for i,x in enumerate(lambda_r) if all(round(x)!=line for line in use_lines)]
        best_lines = [lines[x] for x in best_lines_index]
        if print_output==True:
            print('\n   these are the best lines: ', best_lines)

    nlines = len(best_lines)

    out.write('\n')
    out.write(' Lines with the best fitted profile according to the median wid1 criterion:\n')
    out.write(' --------------------------------------------------------------------------\n')
    for i in range(nlines):
        if i<range(nlines)[-1]:
            out.write('   '+str(best_lines_index[i])+': '+str(best_lines[i])+', ')
        else:
            out.write('   '+str(best_lines_index[i])+': '+str(best_lines[i])+'\n')
    out.write('\n')

    # for i in reversed(rm_lines_idx):
            # print(i)
        # del lambda_shift1[i]
        # del lambda_shift2[i]
        # del lambda_shift1_er[i]
        # del lambda_shift2_er[i]
        # del rvs1[i]
        # del rvs1_er[i]

    # sys.exit()
    if SB2==True:
        if print_output==True:
            print( '\n')
            print( '*** Choosing best epochs for SB2 analysis ***')
            print( '---------------------------------------------')

        # selecting epochs with larger separation between components
        delta_cen, best_OBs = [[] for i in range(nlines)], [[] for i in range(nlines)]

        if not min_sep:
            min_sep = df_Bstars['min_sep'][star_ind]
        # min_sep = 1.5 # higher = relaxed
        print('min_sep: ', min_sep)
        for i, k in zip(range(nlines), best_lines_index):
            for j in range(nepochs):
                # print(best_lines[i], 'epoch', j+1, abs(lambda_shift1[i][j]-lambda_shift2[i][j]))
                delta_cen[i].append(abs(lambda_shift1[k][j]-lambda_shift2[k][j]))
        for i, k in zip(range(nlines), best_lines_index):
            for j in range(nepochs):
                # print('lines =', best_lines[i])
                # print('lambda_shift1 =', lambda_shift1[k][j])
                # print('lambda_shift2 =', lambda_shift2[k][j])
                # print('abs(lambda_shift1[k][j]-lambda_shift2[k][j]) =', abs(lambda_shift1[k][j]-lambda_shift2[k][j]) )
                # print('mean(delta_cen) =', np.mean(delta_cen[i]) )
                # print('mean(delta_cen) - min_sep =', np.mean(delta_cen[i]) -min_sep)
                if abs(lambda_shift1[k][j]-lambda_shift2[k][j]) > np.mean(delta_cen[i])-min_sep:
                    best_OBs[i].append(j)
        # sys.exit()
        common_OB = set(best_OBs[0])
        for OB in best_OBs[1:]:
            common_OB.intersection_update(OB)
        common_OB = list(common_OB)
        wrong_OB = [x for x in range(nepochs) if not x in common_OB]
        if print_output==True:
            # print('   mean(Delta')
            print('   Epochs with components separation >', [f'{np.mean(x)-min_sep:.3f}' for x in delta_cen])
            print('  ', [x+1 for x in common_OB])
            print('   removed epochs: '+str([x+1 for x in wrong_OB]))
        out.write(' Epochs with components separation > '+ f'{np.mean(delta_cen[i])-min_sep:.3f}'+':'+'\n')
        out.write('   '+str([x+1 for x in common_OB])+'\n')
        out.write('   removed epochs: '+str([x+1 for x in wrong_OB])+'\n')
        out.write('\n')

        # removing epochs with inverted components
        # print('wrong_OB:', wrong_OB)
        for j in common_OB:
            temp_OB =[]
            for i in best_lines_index:
                # if j==20:
                #     print(lines[i], 'epoch', j+1, lambda_shift1[i][j], lambda_shift2[i][j], lambda_shift1[i][j]-lambda_shift2[i][j])
                temp_OB.append(lambda_shift1[i][j]-lambda_shift2[i][j])
            if not all(x>0 for x in temp_OB) and not all(x<0 for x in temp_OB):
                wrong_OB.append(j)
        # print('wrong_OB:', wrong_OB)
        oldOBs = copy.deepcopy(common_OB)
        for j in range(len(common_OB)-1, -1, -1):
            if common_OB[j] in wrong_OB:
                del common_OB[j]
        if print_output==True:
            print('   --------')
            print('   Removing epochs with inverted components:')
            print('   Best epochs:', [x+1 for x in common_OB])
            print('   removed epochs:'+str([x+1 for x in oldOBs if x not in common_OB]))
        out.write(' Removing epochs with inverted components:'+'\n')
        out.write('   removed epochs:'+str([x+1 for x in oldOBs if x not in common_OB])+'\n')
        out.write('   Best epochs:'+str([x+1 for x in common_OB])+'\n')
        out.write('\n')

        # removing bad/noisy epochs
        mean_amp1_er, mean_cen1_er = [], []
        if print_output==True:
            print('   --------')
            print('   Removing epochs with large errors:')
        out.write(' Removing epochs with large errors:'+'\n')
        for j in common_OB:
            temp_wid1, temp_cen1 = [], []
            for i in best_lines_index:
                #print(lines[i], 'epoch', j+1, wid1_percer[i][j])
            #    temp_wid1.append(wid1_percer[i][j])
            # if all(np.isnan(x)==False for x in temp_wid1):
            #    print('     epoch', j+1, 'mean(wid1_percer)', np.nanmean(temp_wid1))
            #    mean_amp1_er.append(np.nanmean(temp_wid1))
            # else:
            #    print('     epoch', j+1, 'mean(wid1_percer)', 'nan')
            #    mean_amp1_er.append(np.nan)
                temp_cen1.append(lambda_shift1_er[i][j])
            if all(np.isnan(x)==False for x in temp_cen1):
                # if print_output==True:
                #     print('     epoch', j+1, 'mean(cen1_percer)', np.nanmean(temp_cen1))
                out.write('   epoch '+str(f'{j+1:2}')+' mean(cen1_percer) = '+str(f'{np.nanmean(temp_cen1):.3f}')+'\n')
                mean_cen1_er.append(np.nanmean(temp_cen1))
            else:
                # if print_output==True:
                #     print('     epoch', j+1, 'mean(cen1_percer)', 'nan')
                out.write('   epoch '+str(f'{j+1:2}')+' mean(cen1_percer) = '+'nan'+'\n')
                mean_cen1_er.append(np.nan)

        if print_output==True:
            print('   Applying outlier_killer to remove epochs')
        # err_type_dic={'wid':mean_wid1_er, 'rvs':mean_rvs1_er, 'cen':mean_cen1_er, 'amp':mean_amp1_er}
        err_type_dic={'cen':mean_cen1_er}
        err_type='cen'
        # err_type = df_Bstars['ep_outlier_killer'][star_ind]
        # for key, val in err_type_dic.items():
        #     if val == err_type_dic[err_type]:
        #         err_type_key=str(key)
        rm_OBs_idx = outlier_killer(err_type_dic[err_type], epochs_ok_thrsld, print_output=False)[1]

        # print('rm_OBs_idx:', rm_OBs_idx)
        if print_output==True:
            print('   epochs removed: '+str([common_OB[x]+1 for x in rm_OBs_idx]))
        out.write('   epochs removed: '+str([common_OB[x]+1 for x in rm_OBs_idx])+'\n')
        out.write('\n')
        for i in reversed(rm_OBs_idx):
            wrong_OB.append(common_OB[i])
            del common_OB[i]
        best_OBs = common_OB
        # best_OBs0 = [2, 3, 4, 5, 6, 7, 8, 9, 11, 15, 16, 17, 18, 19, 25, 26, 28]
        # best_OBs = [x-1 for x in best_OBs0]

    else: # if SB2==False
        '''
        for SB1s
        '''
        if rm_epochs:
            if print_output==True:
                print( '\n')
                print( '*** Choosing best epochs for SB1 analysis ***')
                print( '---------------------------------------------')
            OBs_list = list(range(nepochs))
            print(OBs_list)
            if random_eps==True:
                import random
                if rndm_eps_exc:
                    OBs_list = [x for x in OBs_list if x not in rndm_eps_exc]
                # print(sorted(np.random.choice(OBs_list, size=20, replace=False)))
                # OBs_list = [x for x in OBs_list if x not in (9, 11, 12, 23)]
                best_OBs = sorted(np.random.choice(OBs_list, size=rndm_eps_n, replace=False))
                print(OBs_list, best_OBs)
                wrong_OB = [x for x in OBs_list if x not in best_OBs]+rndm_eps_exc
                print(wrong_OB)
            else:
                # removing bad/noisy epochs
                mean_amp1_er, mean_cen1_er, mean_wid1_er, mean_rvs1_er = [], [], [], []
                if print_output==True:
                    print('   --------')
                    print('   Removing epochs with large errors:')
                out.write(' Removing epochs with large errors:'+'\n')
                for j in OBs_list:
                    temp_wid1, temp_cen1, temp_amp1, temp_rvs1 = [], [], [], []
                    for i in best_lines_index:
                        # print(lines[i], 'epoch', j+1, wid1_percer[i][j])
                    #     temp_wid1.append(wid1_percer[i][j])
                    # print('epoch', j+1, 'mean(wid1_percer)', np.nanmean(temp_wid1))
                    # mean_amp1_er.append(np.nanmean(temp_wid1))
                        temp_wid1.append(wid1_percer[i][j])
                        temp_cen1.append(cen1_percer[i][j])
                        temp_amp1.append(amp1_percer[i][j])
                        temp_rvs1.append(100*rvs1_er[i][j]/rvs1[i][j])
                    if all(np.isnan(x)==False for x in temp_wid1):
                        mean_wid1_er.append(np.nanmean(temp_wid1))
                    else:
                        mean_wid1_er.append(np.nan)
                    if all(np.isnan(x)==False for x in temp_cen1):
                        mean_cen1_er.append(np.nanmean(temp_cen1))
                    else:
                        mean_cen1_er.append(np.nan)
                    if all(np.isnan(x)==False for x in temp_amp1):
                        mean_amp1_er.append(np.nanmean(temp_amp1))
                    else:
                        mean_amp1_er.append(np.nan)
                    if all(np.isnan(x)==False for x in temp_rvs1):
                        mean_rvs1_er.append(np.nanmean(temp_rvs1))
                    else:
                        mean_rvs1_er.append(np.nan)
                    if print_output==True:
                        print('     epoch', f'{j+1:2}', ' mean(cen1_%er) =', f'{np.nanmean(temp_cen1):6.4f}', ' mean(wid1_%er) =', f'{np.nanmean(temp_wid1):4.1f}', \
                                                        ' mean(amp1_%er) =', f'{np.nanmean(temp_amp1):4.1f}', ' mean(rvs1_%er) =', f'{np.nanmean(temp_rvs1):4.1f}')
                    out.write('     epoch'+ str(f'{j+1:2}')+ ' mean(cen1_%er) ='+ str(f'{np.nanmean(temp_cen1):6.4f}')+ \
                                                                ' mean(wid1_%er) ='+ str(f'{np.nanmean(temp_wid1):4.1f}')+ \
                                                                ' mean(amp1_%er) ='+ str(f'{np.nanmean(temp_amp1):4.1f}')+ \
                                                                ' mean(rvs1_%er) ='+ str(f'{np.nanmean(temp_rvs1):4.1f}')+'\n')
                if print_output==True:
                    print('   Applying outlier_killer to remove epochs')
                err_type_dic={'wid':mean_wid1_er, 'rvs':mean_rvs1_er, 'cen':mean_cen1_er, 'amp':mean_amp1_er}
                err_type = 'wid'
                for key, val in err_type_dic.items():
                    if val == err_type_dic[err_type]:
                        err_type_key=str(key)
                # print(err_type_dic[err_type])
                # print('\nerr_type =', err_type)
                rm_OBs_idx = outlier_killer(err_type_dic[err_type], epochs_ok_thrsld, print_output=False)[1]

                if print_output==True:
                    print('   epochs removed: '+str(len(rm_OBs_idx))+' '+str([OBs_list[x]+1 for x in rm_OBs_idx]))

                out.write('\n   epochs removed: '+str(len(rm_OBs_idx))+' '+str([OBs_list[x]+1 for x in rm_OBs_idx])+'\n')
                out.write('\n')
                for i in reversed(rm_OBs_idx):
                    del OBs_list[i]
                best_OBs = OBs_list
                wrong_OB = rm_OBs_idx
        else:
            OBs_list = list(range(nepochs))
            best_OBs = OBs_list
            wrong_OB = []

    for i in reversed(rm_lines_idx):
            # print(i)
            del lambda_shift1[i]
            del lambda_shift1_er[i]
            del rvs1[i]
            del rvs1_er[i]
            if SB2==True:
                del lambda_shift2[i]
                del lambda_shift2_er[i]
                del rvs2[i]
                del rvs2_er[i]

    # best_OBs0 = [1, 2, 3, 4, 5, 6, 8, 9, 11, 15, 17, 18, 19, 21, 22, 25, 27, 28]
    # best_OBs = [x-1 for x in best_OBs0]

    nepochs_0 = copy.deepcopy(nepochs)
    nepochs = len(best_OBs)
    rm_OBs_idx = sorted(wrong_OB)
    # rm_OBs_idx = [8, 9, 10, 11, 12, 13, 17, 21, 23, 26, 27]
    # rm_OBs_idx = [7, 10, 12, 13, 14, 16, 20, 23, 24, 26, 29]
    # rm_OBs_idx = [x-1 for x in rm_OBs_idx]
    # Printing/Writing final epochs
    # print('wrong_OB', rm_OBs_idx)
    if print_output==True:
        print('   ------------------')
        print('   Final best epochs:', [x+1 for x in best_OBs])
        print('   Number of epochs:', nepochs)
    out.write(' Final best epochs: '+str([x+1 for x in best_OBs])+'\n')
    out.write(' Number of epochs: '+str(nepochs)+'\n')
    out.write('\n')

    ########################################################################
    '''

    '''
    if print_output==True:
        print( '\n')
        print( '*** Calculating the RV weighted mean for each epoch  ***')
        print( '--------------------------------------------------------')
    # print(len(rvs1[0]))
    rv_list1 = list(map(list, zip(*rvs1))) # nlinesxnepochs (6x26) -> nepochsxnlines (26x6)
    rv_list1_er = list(map(list, zip(*rvs1_er)))
    if SB2==True:
        rv_list2 = list(map(list, zip(*rvs2))) # nlinesxnepochs (6x26) -> nepochsxnlines (26x6)
        rv_list2_er = list(map(list, zip(*rvs2_er)))
    # removing bad epochs from the list of radial velocities
    for j in range(nepochs_0-1, -1, -1):
        if not j in best_OBs:
            del rv_list1[j]
            del rv_list1_er[j]
            if SB2==True:
                del rv_list2[j]
                del rv_list2_er[j]
    # print(rv_list1)
    # print(rv_list2)
    # print(rv_list2_er)
    sigma_rv1, sigma_rv2 = [], []
    mean_rv1, wmean_rv1, mean_rv1_er, wmean_rv1_er = [], [], [], []
    mean_rv2, wmean_rv2, mean_rv2_er, wmean_rv2_er = [], [], [], []
    for i in range(nepochs): # i: 0 -> num of epochs
        sigma_rv1.append(np.std(rv_list1[i])) # std dev of the RVs for each epoch
        # arithmetic mean
        # mean_rv.append(np.mean(rv_list1[i])) # mean RV for each epoch
        # rv_list_sqr = [x**2 for x in rv_list1_er[i]]
        # mean_rv_er.append( np.sqrt( sum(rv_list_sqr) )/len(rv_list_sqr)) # error of the mean RV
        # weighted mean
        mean1, mean1_er = weighted_mean(rv_list1[i], rv_list1_er[i])
        wmean_rv1.append(mean1)
        wmean_rv1_er.append(mean1_er)
        if SB2==True:
            sigma_rv2.append(np.std(rv_list2[i])) # std dev of the RVs for each epoch
            mean2, mean2_er = weighted_mean(rv_list2[i], rv_list2_er[i])
            wmean_rv2.append(mean2)
            wmean_rv2_er.append(mean2_er)
    sigma_tot1 = np.std(sigma_rv1)
    sigma_er_tot1 = np.std(wmean_rv1_er)
    if SB2==True:
        sigma_tot2 = np.std(sigma_rv2)
        sigma_er_tot2 = np.std(wmean_rv2_er)

    # Coming back to nlines x nepochs without the removed epochs
    # to calculate a mean RV for each line
    rvs1 = list(map(list, zip(*rv_list1)))
    rvs1_er = list(map(list, zip(*rv_list1_er)))
    if SB2==True:
        rvs2 = list(map(list, zip(*rv_list2)))
        rvs2_er = list(map(list, zip(*rv_list2_er)))
    ep_avg1, ep_avg1_er = [], []
    ep_avg2, ep_avg2_er = [], []
    for i in range(nlines):
        # arithmetic mean
        ep_avg1.append(np.mean(rvs1[i])) # this is the mean RV of all epochs for each line
        ep_avg1_sqr = [x**2 for x in rvs1_er[i]]
        ep_avg1_er.append( np.sqrt( sum(ep_avg1_sqr) ) / len(ep_avg1_sqr) )
        #ep_avg1_er.append( np.sqrt( sum(ep_avg1_sqr)/(len(ep_avg1_sqr)-1) ) / np.sqrt(len(ep_avg1_sqr)) )
        if SB2==True:
            ep_avg2.append(np.mean(rvs2[i])) # this is the mean of all epochs for each line
            ep_avg2_sqr = [x**2 for x in rvs2_er[i]]
            ep_avg2_er.append( np.sqrt( sum(ep_avg2_sqr) ) / len(ep_avg2_sqr) )

    total_mean_rv1, total_mean_rv1_er = weighted_mean(wmean_rv1, wmean_rv1_er)
    if SB2==True:
        total_mean_rv2, total_mean_rv2_er = weighted_mean(wmean_rv2, wmean_rv2_er)
    # Printing/Writing
    if print_output==True:
        print('\n RV mean of the ',str(nepochs),' epochs for each line:')
        print(' -----------------------------------------')
        for i in range(nlines):                              # f'{a:.2f}
            #out.write('   - '+str(best_lines[i])+': '+str(f'{ep_avg1[i]:.3f}')+'\n')
            print('   - ',str(best_lines[i]),': ',str(f'{ep_avg1[i]:.3f}'),' +/- ',str(f'{ep_avg1_er[i]:.3f}'))
        print('\n')
        print(' Weighted mean RV of the ',str(nepochs),' epochs:')
        print(' -------------------------')
        print('   ', 'Primary  : ',str( f'{total_mean_rv1:.3f}' ),' +/- ',str( f'{total_mean_rv1_er:.3f}' ) ,\
            ', std dev = ', str(f'{sigma_tot1:.3f}'))
        if SB2==True:
            print('   ', 'Secondary: ',str( f'{total_mean_rv2:.3f}' ),' +/- ',str( f'{total_mean_rv2_er:.3f}' ) ,\
                ', std dev = ', str(f'{sigma_tot2:.3f}'))
        print('\n')
    out.write(' RV mean of the '+str(nepochs)+' epochs for each line: \n')
    out.write(' ---------------------------------------\n')
    for i in range(nlines):                              # f'{a:.2f}
        #out.write('   - '+str(best_lines[i])+': '+str(f'{ep_avg1[i]:.3f}')+'\n')
        out.write('   - '+str(best_lines[i])+': '+str(f'{ep_avg1[i]:.3f}')+' +/- '+str(f'{ep_avg1_er[i]:.3f}')+'\n')
    out.write('\n')
    out.write(' Weighted mean RV of the '+str(nepochs)+' epochs:\n')
    out.write('------------------------------------------\n')
    #out.write('   '+str(np.mean(mean_rv))+' +/- '+str(sigma_tot)+' (std dev)\n')
    out.write('   '+str( f'{total_mean_rv1:.3f}' )+' +/- '+ str( f'{total_mean_rv1_er:.3f}' ) +\
        ', std dev = '+ str(f'{sigma_tot1:.3f}')+'\n')
    out.write('\n')
    out.close()

    #####################################################################

    '''
    Writing RVs to file RVs.txt
    '''
    # JDfile = 'JDs.dat'
    # JDfile = 'JDs+vfts.dat'
    # df_rv = pd.read_csv(path+'JDs.dat', names = ['BBC_epoch', 'HJD'], sep='\t')
    df_rv = pd.read_csv(JDfile, names = ['epoch', 'JD'], sep='\s+')
    # print(df_rv)
    df_rv = df_rv.replace({'BBC_':''}, regex=True).replace({'.fits':''}, regex=True)
    if not len(rm_OBs_idx)==0:
        df_rv.drop(rm_OBs_idx, inplace=True)
        df_rv.reset_index(inplace=True)
    for i in range(nlines):
        df_rv['rv_'+str(best_lines[i])] = rvs1[i]
        df_rv['rv_'+str(best_lines[i])+'_er'] = rvs1_er[i]
    df_rv['mean_rv'] = wmean_rv1
    df_rv['mean_rv_er'] = wmean_rv1_er
    df_rv['sigma_rv'] = sigma_rv1
    #df_rv.to_csv(path+'RVs.dat', sep='\t', index=False)
    with open(path+'RVs1.txt', 'w') as fo:
        #fo.write(df_rv.to_string(formatters={'mean_rv':'{:.25f}'.format}, index=False))
        fo.write(df_rv.to_string(formatters={'HJD': '{:.8f}'.format}, index=False))
    if SB2==True:
        df_rv2 = pd.read_csv(path+'JDs.txt', names = ['epoch', 'JD'], sep='\t')
        df_rv2 = df_rv2.replace({'BBC_':''}, regex=True).replace({'.fits':''}, regex=True)
        if not len(rm_OBs_idx)==0:
            df_rv2.drop(rm_OBs_idx, inplace=True)
            df_rv2.reset_index(inplace=True)
        for i in range(nlines):
            df_rv2['rv_'+str(best_lines[i])] = rvs2[i]
            df_rv2['rv_'+str(best_lines[i])+'_er'] = rvs2_er[i]
        df_rv2['mean_rv'] = wmean_rv2
        df_rv2['mean_rv_er'] = wmean_rv2_er
        df_rv2['sigma_rv'] = sigma_rv2
        #df_rv.to_csv(path+'RVs.dat', sep='\t', index=False)
        with open(path+'RVs1.txt', 'a') as fo:
            #fo.write(df_rv.to_string(formatters={'mean_rv':'{:.25f}'.format}, index=False))
            fo.write('\n')
            fo.write('Secondary:\n')
            fo.write(df_rv2.to_string(formatters={'HJD': '{:.8f}'.format}, index=False))
    # print(f'SB2: {SB2}')
    if SB2==True:
        return df_rv2
    else:
        return df_rv

def lomb_scargle(df, path, probabilities = [0.5, 0.01, 0.001], SB2=False, rv2=False, rv2_err=False, print_output=False, plots=False, best_lines=False, min_sep=False):
    '''
    df: dataframe with the RVs, output from getrvs()
    '''
    if not os.path.exists(path+'LS'):
        os.makedirs(path+'LS')

    hjd, rv, rv_err = df['JD'], df['mean_rv'], df['mean_rv_er']
    starname = df['epoch'][0].split('_')[0]+'_'+df['epoch'][0].split('_')[1]
    print('starname:', starname)
    nepochs=len(hjd)

    lsout = open(path+'LS/ls_output.txt', 'w')
    lsout.write(' *************************\n')
    lsout.write('   LS output for star '+starname+'\n')
    lsout.write(' *************************\n')
    lsout.write('\n')

    # probabilities = [0.5, 0.01, 0.001]
    # print(rv)
    #ls = LombScargle(df_rv['HJD'], df_rv['mean_rv'], df_rv['sigma_rv'])
    # print(type(rv_err), rv_err)
    if rv_err.any() == False:
        ls1 = LombScargle(hjd, rv, normalization='model')
    else:
        ls1 = LombScargle(hjd, rv, rv_err, normalization='model')
    #method = 'baluev'
    method = 'bootstrap'
    fal1 = ls1.false_alarm_level(probabilities, method=method)
    if print_output==True:
        print('   these are the false alarm levels: ', [f'{x:.3f}' for x in fal1])
    lsout.write(' these are the false alarm levels: ')
    for fal in fal1:
        lsout.write(str(f'{fal:.3f}')+'  ')
    lsout.write('\n')
    # t0 = time.time()
    frequency1, power1 = ls1.autopower(method='fast', minimum_frequency=0.001, #0.001=1000
                                                maximum_frequency=2.5, # 0.9=1.1, 1.5=0.67, 2=0.5, 4=0.25
                                                samples_per_peak=2000) # 2000
    # t1 = time.time()
    # print('# LS calculation time: ', t1-t0)
    fap1 = ls1.false_alarm_probability(power1.max(), method=method)
    # t2 = time.time()
    # print('# FAP calculation time: ', t2-t1)
    fapper1 = fap1*100
    if print_output==True:
        print('   FAP of the highest peak         : ', f'{fap1:.5f}')
        print('   FAP of the highest peak x100    : ', f'{fapper1:.5f}')
    lsout.write(' FAP of the highest peak         : '+ f'{fap1:.5f}'+'\n')
    lsout.write(' FAP of the highest peak x100    : '+ f'{fapper1:.5f}'+'\n')

    max_freq1 = frequency1[np.argmax(power1)]
    # print('max_freq1 =', max_freq1)

    fal_50pc, fal_1pc, fal_01pc = fal1[0], fal1[1], fal1[2] 
    if power1.max() > fal_01pc:
        peaks_min_h = float(f'{fal_1pc:.3f}')
    elif power1.max() <= fal_01pc and power1.max() > fal_1pc:
        peaks_min_h = float(f'{0.6*fal_1pc:.3f}')
    else:
        peaks_min_h = float(f'{fal_50pc:.3f}')
    # print(peaks_min_h)
    
    freq_index1 = np.argmin(np.abs(frequency1 - 1/5))
    freq_index2 = np.argmin(np.abs(frequency1 - 1/1.1))
    peaks1a, _ = find_peaks(power1[:freq_index1], height=peaks_min_h, distance=1000)
    peaks1b, _ = find_peaks(power1[:freq_index2], height=peaks_min_h, distance=5000)
    # print(peaks1a)
    # print(peaks1b)
    peaks1 = np.unique(np.concatenate((peaks1a, peaks1b)))
    # print(peaks1)
    # peaks1, _ = find_peaks(power1, height=0.8*fal1[1], distance=20000) # height=0.8*fal1[1], distance=20000
    # peaks1, _ = find_peaks(power1, height=0.5*fal1[2], distance=1000) # height=0.8*fal1[1], distance=20000
    freq1 = frequency1[peaks1]
    peri1 = 1/frequency1[peaks1]

    if 1/max_freq1 < 1.2 and any(1/freq1 > 1.2):
        for i in range(len(1/freq1)):
            if 1/freq1[i]==max(1/freq1):
                best_freq1 = freq1[i]
                LS_power1 = power1[peaks1][i]
    else:
        best_freq1 = max_freq1
        LS_power1 = max(power1)
    best_period1 = 1/best_freq1
    # print('freq1 =', freq1)
    # print('peri1 =', peri1)
    # print('best_freq1 =', best_freq1)
    # print('best_period1 =', best_period1)

    if len(peri1)>0:
        max_peri1 = max(peri1)
        max_peri1_idx = peri1.tolist().index(max_peri1)
        # print('max_peri_idx', max_peri_idx)
        max_peri1_pow = power1[peaks1][max_peri1_idx]
        # print('max_peri_pow', max_peri_pow)
        # print('fal[1]/max_peri_pow', fal[1]/max_peri_pow)
        allP = sorted(peri1)
    if len(peri1)==0:
        peri1 = [best_period1]

    # periods.append(1/frequency1[peaks1])
    # frequencies.append(frequency1[peaks1])

    if SB2==True:
        if rv2_err == False:
            ls2 = LombScargle(hjd, rv2, normalization='model')
        else:
            ls2 = LombScargle(hjd, rv2, rv2_err, normalization='model')
        fal2 = ls2.false_alarm_level(probabilities, method=method)
        frequency2, power2 = ls2.autopower(method='fast', minimum_frequency=0.001, #0.001=1000
                                                    maximum_frequency=2.5, # 0.9=1.1, 1.5=0.67, 2=0.5
                                                    samples_per_peak=2000)
        fap2 = ls2.false_alarm_probability(power2.max(), method=method)
        fapper2 = fap2*100
        LS_power2 = max(power2)
        best_freq2 = frequency2[np.argmax(power2)]
        best_period2 = 1/best_freq2
        peaks2_min_h = float(f'{0.8*fal1[1]:.3f}')
        # print(peaks2_min_h)
        peaks2, _ = find_peaks(power2, height=peaks2_min_h, distance=20000) # distance=20000
        # peaks2, _ = find_peaks(power2, height=0.5*fal2[2], distance=1000) # distance=20000
        peri2 = 1/frequency2[peaks2]
        freq2 = frequency2[peaks2]
        if best_period2 < 1.1:
            if print_output==True:
                print('\n')
                print('   ### Warning: Secondary period possible alias, selecting most similar period to P$\_$1')
                print('\n')
            lsout.write(' ### Warning: Secondary period possible alias, selecting most similar period to P$\_$1 \n')
            dif = []
            for i in range(len(freq2)):
                dif.append(np.abs(best_freq1-freq2[i]))
            for i in range(len(freq2)):
                if np.abs(best_freq1-freq2[i]) == np.min(dif):
                    best_freq2   = freq2[i]
                    best_period2 = 1/freq2[i]
        # test |RV1-RV2|
        rv_abs = np.abs(rv-rv2)
        rv_abs_er = np.sqrt(rv_err**2+rv2_err**2)
        ls3 = LombScargle(hjd, rv_abs, rv_abs_er, normalization='model')
        fal3 = ls3.false_alarm_level(probabilities, method=method)
        frequency3, power3 = ls3.autopower(method='fast', minimum_frequency=0.001, #0.001=1000
                                                    maximum_frequency=2.5, # 0.9=1.1, 1.5=0.67, 2=0.5
                                                    samples_per_peak=2000)
        fap3 = ls3.false_alarm_probability(power3.max(), method=method)
        LS_power3 = max(power3)
        best_freq3 = frequency3[np.argmax(power3)]
        best_period3 = 1/best_freq3
        peaks3_min_h = float(f'{0.8*fal3[1]:.3f}')
        peaks3, _ = find_peaks(power3, height=peaks3_min_h, distance=20000) # distance=20000


    # t3 = time.time()
    # print('# Peaks calculation time and all before prints: ', t3-t2)
    # print("   number of periods               :  {0}".format(len(frequency1)))
    if print_output==True:
        print("   Best frequency                  :  {0:.3f}".format(best_freq1))
        print('   ***********************************************')
        print("   Best Period                     :  {0:.8f} days".format(best_period1))
        print('   ***********************************************')
        if SB2==True:
            print("   Best Period from secondary      :  {0:.8f} days".format(best_period2))
            print("   Period from |RV1-RV2|           :  {0:.8f} days".format(best_period3), 'correct period = P1 or ', best_period1/2)

        print('   Other periods:')
        print('     peaks                         : ', [f'{x:.3f}' for x in power1[peaks1]])
        print('     positions                     : ', [x for x in peaks1 ])
        # print('     positions                     : ', [f'{x:.3f}' for x in np.where(power1==power[peaks1]) ])
        # print('/////////////// these are the periods', periods)
        print('     frequencies                   : ', [f'{x:.5f}' for x in freq1])
        print('     periods                       : ', [f'{x:.3f}' for x in peri1])
        if SB2==True:
            print('   from secondary:')
            print('     peaks                         : ', [f'{x:.3f}' for x in power2[peaks2]])
            print('     frequencies                   : ', [f'{x:.3f}' for x in freq2])
            print('     periods                       : ', [f'{x:.3f}' for x in peri2])
    lsout.write("\n Best frequency                  :  {0:.3f}".format(best_freq1)+'\n')
    lsout.write(' ****************************************************'+'\n')
    lsout.write(" Best Period                     :  {0:.8f} days".format(best_period1)+'\n')
    lsout.write(' ****************************************************'+'\n')
    if SB2==True:
        lsout.write(" Best Period from secondary      :  {0:.8f} days".format(best_period2)+'\n')
        lsout.write(" Period from |RV1-RV2|           :  {0:.8f} days".format(best_period3)+ 'correct period = P1 or '+ str(best_period1/2)+'\n')
    lsout.write(' Other periods:'+'\n')
    lsout.write('   peaks                         : ')
    for peak in power1[peaks1]:
        lsout.write('     '+f'{peak:7.3f}')
    lsout.write('\n   frequencies                   : ')
    for freq in freq1:
        lsout.write('     '+f'{freq:7.3f}')
    lsout.write('\n   periods                       : ')
    for per in peri1:
        lsout.write('     '+f'{per:7.3f}')
    if SB2==True:
        lsout.write('\n from secondary:\n')
        lsout.write('   peaks                         : ')
        for peak in power2[peaks2]:
            lsout.write('     '+f'{peak:7.3f}')
        lsout.write('\n   frequencies                   : ')
        for freq in freq2:
            lsout.write('     '+f'{freq:7.3f}')
        lsout.write('\n   periods                       : ')
        for per in peri2:
            lsout.write('     '+f'{per:7.3f}')
    lsout.write('\n')
    # print(freq[2]-freq[1], freq[0])
    # print('beat period (f2-f1) = ',2*(freq[2]-freq[1]))
    # print('beat period (f1-f0) = ',(freq[1]-freq[0]))
    # print('beat period (f3-f2) = ',(freq[3]-freq[2]))
    # print('beat period 2/(f2-f1) = ',2/(freq[2]-freq[1]))
    # print("Best Period: {0:.8f} days".format(best_period))

   # print('allP: ', allP)
    #print('number of local maxima: '+str(len(mm[0])))
    #yy = [power[i] for i in mm]
    # t4 = time.time()
    # print('# Printing/writing time: ', t4-t3)

    indi = []
    for LS_pow, peri in zip(power1[peaks1], peri1):

        maxpower_maxfal = fal1[2]/LS_pow
        maxpower_maxfal2 = fal1[1]/LS_pow
        if print_output==True:
            print('   fal1/P                          : ', f'{maxpower_maxfal:.2f}')
            print('   fal2/P                          : ', f'{maxpower_maxfal2:.2f}')
            # print('fal1[2]/LS_pow :', fal1[2], '/', LS_pow, ' = ', fal1[2]/LS_pow)
        lsout.write(' fal1/P                          :  '+ f'{maxpower_maxfal:.2f}'+'\n')
        lsout.write(' fal2/P                          :  '+ f'{maxpower_maxfal2:.2f}'+'\n')
        # print('maxpower_maxfal =', maxpower_maxfal)
        # print('maxpower_maxfal2 =', maxpower_maxfal2)
        # print('peri =', peri)
        # print('peri1 =', peri1, ' len(peri1) =', len(peri1))
        # print('allP[-1] =', allP[-1])
        # print('fal1[1]/LS_power1', fal1[1]/LS_power1)
        # print('fal1[1]/max_peri1_pow', fal1[1]/max_peri1_pow)

        if len(peri1)>0:
            if maxpower_maxfal < 0.2 and peri >= 1.1:
                indi.append(0)
            elif maxpower_maxfal >= 0.2 and maxpower_maxfal < 0.6 and peri >= 1.1:
                indi.append(1)
            elif maxpower_maxfal >= 0.6 and maxpower_maxfal < 0.95 and peri >= 1.1:
                indi.append(2)
            elif peri >= 1.05 and maxpower_maxfal >= 0.95 and maxpower_maxfal2 < 0.95:
                indi.append(3)
            elif peri < 1.05 and sorted(peri1)[-1] > 1.05 and maxpower_maxfal < 1.0 and fal1[1]/LS_power1 < 1.0:
                indi.append(3)
            elif peri >= 1.05 and maxpower_maxfal >= 0.9 and maxpower_maxfal2 < 1.2:
                indi.append(4)
            elif peri < 1.05 and sorted(peri1)[-1] > 1.05 and maxpower_maxfal2 > 1.1 and len(peri1) > 1 and fal1[1]/LS_power1 < 1.25:
                indi.append(4)
            elif peri < 1.05 and sorted(peri1)[-1] > 1.05 and maxpower_maxfal2 < 1.1 and len(peri1) > 1 and fal1[1]/LS_power1 < 1.25:
                indi.append(4)
            elif peri >= 1.05 and maxpower_maxfal > 1.5 and maxpower_maxfal2 < 1.5:
                indi.append(5)
            elif peri < 1.05 and maxpower_maxfal2 < 1.1 and len(peri1) < 2:
                indi.append(5)
            elif peri < 1.05 and maxpower_maxfal2 < 1.1 and len(peri1) > 1 and sorted(peri1)[-1] < 1.05:
                indi.append(5)
            elif peri < 1.05 and maxpower_maxfal2 > 1.1 and len(peri1) > 1 and sorted(peri1)[-1] > 1.05 and fal1[1]/max_peri1_pow > 1.1:
                indi.append(5)
            else:
                indi.append(6)
           # for i in range(len(df_Bstars)):
           #     if df_Bstars['VFTS'][i] == star:
           #         if df_Bstars['SB2'][i] in ['SB2', 'SB2?']:
           #             ind = 7
        else:
            indi.append(6)
        if peri == best_period1:
            ind = indi[-1]
        # print('indi=',indi)
        # print('best_period1=',best_period1)
        # print('\n')
    if not indi:
        ind = 6
    elif all(x==6 for x in indi):
        ind = 6
    # print('indi =', indi)
    maxpower_maxfal = fal1[2]/LS_power1
    maxpower_maxfal2 = fal1[1]/LS_power1
    if print_output==True:
        # print(indi)
        print('\n   Classification index            : ', ind)
        print('   LS_power1                       : ', f'{LS_power1:.2f}')
        print('   fal1                            : ', f'{fal1[2]:.2f}')
        print('   maxpower_maxfal                 : ', f'{maxpower_maxfal:.2f}')
    lsout.write(' Classification index            :  '+str(ind)+'\n')
    # t5 = time.time()
    # print('# Classification index calculation time: ', t5-t4)
   # print('  fal1/power: ', maxpower_maxfal)
   # print('  fal2/power: ', maxpower_maxfal2)


    '''
    Plot periodogram
    '''
    # vs Period
    if plots==True:
        bins = [0.5, 1, 2, 5, 10, 20, 50, 100, 200, 500, 1000]
        if SB2==True:
            f_list, pow_list, comp_list, Per_list, fal_list, peak_list = [frequency1, frequency2, frequency3], \
                        [power1, power2, power3], ['primary', 'secondary', 'subtracted'], \
                        [best_period1, best_period2, best_period3], [fal1, fal2, fal3], [peaks1, peaks2, peaks3]
        else:
            f_list, pow_list, comp_list, Per_list, fal_list, peak_list = [frequency1], \
                [power1], ['primary'], [best_period1], [fal1], [peaks1]
        for frequency, power, comp, best_period, fal, peaks in \
                zip(f_list, pow_list, comp_list, Per_list, fal_list, peak_list):
            if not any(np.isnan(power)):
                fig, ax = plt.subplots(figsize=(8, 6))
                fig.subplots_adjust(left=0.12, right=0.97, top=0.93, bottom=0.12)
                ax.plot(1/frequency, power, 'k-', alpha=0.5)
                #ax.plot(1/frequency[peaks0], power[peaks0], "xr", label='distance=200')
                ax.plot(1/frequency[peaks], power[peaks], "ob", label='prominence=1')
                #ax.plot(1/frequency[peaks2], power[peaks2], "vg", label='width=2000')
                #ax.plot(1/frequency[peaks3], power[peaks3], "xk", label='threshold=2')
                #ax.plot(1/frequency[peaks4], power[peaks4], "xk", label='height=1')
                #ax.plot(1/frequency[mm], power[mm], 'ro')
                ax.yaxis.set_label_coords(-0.09, 0.5)
                ax.set(xlim=(0.3, 1300), ylim=(-0.03*power.max(), power.max()+0.1*power.max()), #x2=426
                # ax.set(xlim=(0.4, 600), ylim=(0, 35), #x2=426
                   xlabel='Period (days)',
                   ylabel='Lomb-Scargle Power');
                # ax.tick_params(which='both', width=0.6, labelsize=14)
                plt.xscale('log')
                #ax.xaxis.set_major_formatter(matplotlib.ticker.ScalarFormatter())
                tickLabels = map(str, bins)
                ax.set_xticks(bins)
                ax.set_xticklabels(tickLabels)
                # ax.set_yticks(range(0,35,5))
                ax.plot( (0.5, 800), (fal[0], fal[0]), '--r', lw=1.2)
                ax.plot( (0.5, 800), (fal[1], fal[1]), '--y', lw=1.2)
                ax.plot( (0.5, 800), (fal[2], fal[2]), '--g', lw=1.2)
                ax.text( 100, fal[2]+0.01, '0.1\% fap', fontsize=16)
                ax.text( 100, fal[1]+0.01, '1\% fap', fontsize=16)
                ax.text( 100, fal[0]+0.01, '50\% fap', fontsize=16)
                # ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
                if power.max()+0.1*power.max() >= 10:
                    ax.get_yaxis().set_major_formatter(StrMethodFormatter('{x:.0f}'))
                else:
                    ax.get_yaxis().set_major_formatter(StrMethodFormatter('{x:.1f}'))
                plt.title(starname+' Periodogram $-$ Best Period: {0:.4f}'.format(best_period)+' d')
                if best_lines and min_sep:
                    if SB2==True:
                        labels = ['lines ='+str(best_lines), 'min sep ='+str(min_sep)]
                    else:
                        labels = ['lines ='+str(best_lines), 'n epochs ='+str(nepochs)]
                    leg = plt.legend(labels, loc='best', markerscale=0, handletextpad=0, handlelength=0)
                    for item in leg.legendHandles:
                        item.set_visible(False)
                # plt.tight_layout()
                plt.savefig(path+'LS/LS-fap-BBB_'+starname+'_'+comp+'_periods_'+str(len(rv))+'_epochs.pdf')
                plt.close()

                # Plot for paper
                fig, ax = plt.subplots(figsize=(8, 6))
                fig.subplots_adjust(left=0.12, right=0.97, top=0.93, bottom=0.12)
                ax.plot(1/frequency, power, 'k-', alpha=0.5)
                ax.yaxis.set_label_coords(-0.09, 0.5)
                ax.set(xlim=(0.3, 1300), ylim=(-0.03*power.max(), power.max()+0.1*power.max()),
                   xlabel='Period (days)',
                   ylabel='Lomb-Scargle Power');
                plt.xscale('log')
                tickLabels = map(str, bins)
                ax.set_xticks(bins)
                ax.set_xticklabels(tickLabels)
                ax.plot( (0.5, 800), (fal[0], fal[0]), '-r', lw=1.2)
                ax.plot( (0.5, 800), (fal[1], fal[1]), '--y', lw=1.2)
                ax.plot( (0.5, 800), (fal[2], fal[2]), ':g', lw=1.2)
                # ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
                if power.max()+0.1*power.max() >= 10:
                    ax.get_yaxis().set_major_formatter(StrMethodFormatter('{x:.0f}'))
                else:
                    ax.get_yaxis().set_major_formatter(StrMethodFormatter('{x:.1f}'))
                # plt.title('VFTS '+starname)
                plt.title(starname+' '+comp)
                plt.savefig(path+'LS/BBC_'+starname+'_paper_LS_'+comp+'.png')
                # plt.show()
                plt.close()

        # t6 = time.time()
        # print('# Normal plot + paper plot time: ', t6-t5)

        # vs grid points
        fig, ax = plt.subplots(figsize=(8, 6))
        fig.subplots_adjust(left=0.08, right=0.97, top=0.94, bottom=0.10)
        ax.plot(power1)
        ax.plot(peaks1, power1[peaks1], "ob")
        #ax.set(xlim=(0.4, 600), ylim=(0, power.max()+0.1*power.max()), #x2=426
        #ax.set(xlim=(0.4, 600), ylim=(0, 15), #x2=426
        ax.set(ylim=(0, power1.max()+0.1*power1.max()),
           xlabel='number of points',
           ylabel='Lomb-Scargle Power');
        ax.xaxis.label.set_size(15)
        ax.yaxis.label.set_size(15)
        ax.tick_params(which='both', width=0.6, labelsize=14)
        ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
        ax.get_yaxis().set_major_formatter(StrMethodFormatter('{x:,.0f}'))
        plt.savefig(path+'LS/LS-fap-BBC_'+starname+'_points.pdf')
        plt.close()

        # vs frequency
        fig, ax = plt.subplots(figsize=(8, 6))
        fig.subplots_adjust(left=0.08, right=0.97, top=0.94, bottom=0.10)
        ax.plot(frequency1, power1)
        ax.plot(frequency1[peaks1], power1[peaks1], "ob")
        ax.vlines(np.abs(best_freq1-1), 0, power1.max()+0.1*power1.max(), colors='green', linestyles='dashed')
        ax.vlines(np.abs(best_freq1-2), 0, power1.max()+0.1*power1.max(), colors='green', linestyles='dashed')
        ax.vlines(np.abs(best_freq1+1), 0, power1.max()+0.1*power1.max(), colors='green', linestyles='dashed')
        ax.vlines(np.abs(best_freq1+2), 0, power1.max()+0.1*power1.max(), colors='green', linestyles='dashed')
        ax.text( best_freq1+0.03, power1.max(), r'$\mathrm{f}_0$', fontsize=14)
        ax.text( np.abs(best_freq1-1)-0.03, power1.max(), r'$\left| \mathrm{f}_0-1\right|$', fontsize=14, horizontalalignment='right')
        ax.text( np.abs(best_freq1-2)-0.03, power1.max(), r'$\left| \mathrm{f}_0-2\right|$', fontsize=14, horizontalalignment='right')
        ax.text( np.abs(best_freq1+1)+0.03, power1.max(), r'$\left| \mathrm{f}_0+1\right|$', fontsize=14, horizontalalignment='left')
        ax.text( np.abs(best_freq1+2)+0.03, power1.max(), r'$\left| \mathrm{f}_0+2\right|$', fontsize=14, horizontalalignment='left')
        # ax.set(xlim=(-0.1, 2.3), ylim=(0, power1.max()+0.1*power1.max()), #x2=426
        #ax.set(xlim=(0.4, 600), ylim=(0, 15), #x2=426
        ax.set(ylim=(0, power1.max()+0.1*power1.max()),
           xlabel='frequency (1/d)',
           ylabel='Lomb-Scargle Power');
        ax.xaxis.label.set_size(15)
        ax.yaxis.label.set_size(15)
        ax.tick_params(which='both', width=0.6, labelsize=14)
        ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
        ax.get_yaxis().set_major_formatter(StrMethodFormatter('{x:,.0f}'))
        plt.savefig(path+'LS/LS-fap-BBC_'+starname+'_frequencies.pdf')
        plt.close()

        # Stack the arrays in columns
        pow_spec = np.column_stack((frequency1, power1))
        # Save the data to a text file
        np.savetxt(path+'LS/power_spectrum.txt', pow_spec)

    if SB2==True:
        return [best_period1, best_period2, best_period3], [frequency1, frequency2, frequency3], [power1, power2, power3], [LS_power1, LS_power2, LS_power3], [fal1, fal2, fal3], [peaks1, peaks2, peaks3], sorted(peri1), peri2, indi, [fapper1, maxpower_maxfal, maxpower_maxfal2]
    else:
        return [best_period1], [frequency1], [power1], [LS_power1], [fal1], [peaks1], sorted(peri1), ind, [fapper1, maxpower_maxfal, maxpower_maxfal2]

def phase_rv_curve(df, periods, path, SB2=False, print_output=True, plots=True):

    '''
    Compute phases of the obsevations and models
    '''
    if print_output==True:
        print( '\n*** Computing phases ***')
        print( '------------------------')

    hjd, rv, rv_err = df['JD'], df['mean_rv'], df['mean_rv_er']
    name = df['epoch'][0].split('_')[0]+'_'+df['epoch'][0].split('_')[1]
    # print('starname:', starname)

    for Per in periods:
        phase = [ x/Per % 1 for x in hjd]
        ini = min(phase)
        fin = max(phase)

        df['phase'] = phase
        df_phase1 = df.sort_values('phase', ascending=True).reset_index(drop=True)
        df_phase2 = df_phase1.copy(deep=True)
        for i in range(len(df_phase2)):
            df_phase2.loc[i, 'phase']=df_phase1['phase'][i]+1
        df_phase = pd.concat([df_phase1, df_phase2], ignore_index=True)
        # Create a finer grid of phase values
        fine_phase = np.linspace(df_phase['phase'].min(), df_phase['phase'].max(), 1000)

        # Fitting a sinusoidal curve
        sinumodel = Model(sinu)

        phase_shift= [df_phase['phase'].iloc[i] for i,x in enumerate(df_phase['mean_rv']) if df_phase['mean_rv'].iloc[i]==max(df_phase['mean_rv'])][0]
        pars = Parameters()
        pars.add('A', value=(df_phase['mean_rv'].max()-df_phase['mean_rv'].min())/2 )
        pars.add('w', value=2*np.pi)
        pars.add('phi', value=(1-phase_shift+0.25)*2*np.pi)
        pars.add('h', value=(df_phase['mean_rv'].max()-df_phase['mean_rv'].min())/2 +df_phase['mean_rv'].min())

        res_sinu = sinumodel.fit(df_phase['mean_rv'], pars, x=df_phase['phase'], weights=1/df_phase['mean_rv_er'])
        # Calculate the 3-sigma uncertainty
        dely = 3 * res_sinu.params['A'].stderr
        # dely0 = res_sinu.eval_uncertainty(sigma=3)
        best_pars=res_sinu.best_values
        init_pars=res_sinu.init_values
        # Interpolate the best_fit values over the finer grid
        interp_func = interp1d(df_phase['phase'], res_sinu.best_fit, kind='cubic')
        fine_best_fit = interp_func(fine_phase)        
        # print('chi2 = ', res_sinu.chisqr)
        # print('Per = ', Per)
        out_sinu = open(path+'LS/'+name+f'_sinu_stats_{Per}.txt', 'w')
        out_sinu.write(name+'\n')
        out_sinu.write(res_sinu.fit_report()+'\n')
        out_sinu.close()

        sinu_chi2 = res_sinu.chisqr
        dely_dif = 2*dely
        mean_dely = np.mean(dely_dif)
        syst_RV = best_pars['h']
        syst_RV_er = res_sinu.params['h'].stderr
        # delta_RV = 2*best_pars['A']
        # delta_RV_er = 2*res_sinu.params['A'].stderr

        xx = np.linspace(df_phase['phase'].iloc[0], df_phase['phase'].iloc[-1], num=100, endpoint=True)
        #spl3 = UnivariateSpline(df_phase['phase'], df_phase['mean_rv'], df_phase['sigma_rv'], s=5e20)
        #diff = spl3(df_phase['phase']) - df_phase['mean_rv']
        # if SB2==True:
        #     for period2 in peri2:
        #         if (Per>1.1 and Per<80 and np.abs(Per-period2)<0.5) or (Per>=80 and np.abs(Per-period2)<2):
        #         # if f'{Per:.2f}'==f'{best_period2:.2f}':
        #             # print(Per, ' - ', period2, ' = ', np.abs(Per-period2))
        #             # phase = [ x/best_period2 % 1 for x in df_rv['HJD']]
        #             # ini = min(phase)
        #             # fin = max(phase)
        #             df_rv2['phase'] = phase
        #             df2_phase1 = df_rv2.sort_values('phase', ascending=True).reset_index(drop=True)
        #             df2_phase2 = df2_phase1.copy(deep=True)

        #             for i in range(len(df2_phase2)):
        #                 df2_phase2.loc[i, 'phase']=df2_phase1['phase'][i]+1
        #             df2_phase = pd.concat([df2_phase1, df2_phase2], ignore_index=True)
        #             phase_shift2= [df2_phase['phase'].iloc[i] for i,x in enumerate(df2_phase['mean_rv']) if df2_phase['mean_rv'].iloc[i]==max(df2_phase['mean_rv'])][0]
        #             pars = Parameters()
        #             pars.add('A', value=(df2_phase['mean_rv'].max()-df2_phase['mean_rv'].min())/2 )
        #             pars.add('w', value=2*np.pi)
        #             pars.add('phi', value=(1-phase_shift2+0.25)*2*np.pi)
        #             pars.add('h', value=(df2_phase['mean_rv'].max()-df2_phase['mean_rv'].min())/2 +df2_phase['mean_rv'].min())

        #             res_sinu2 = sinumodel.fit(df2_phase['mean_rv'], pars, x=df2_phase['phase'], weights=1/df2_phase['mean_rv_er'])
        #             #res_sinu = sinumodel.fit(df_phase['mean_rv'], x=df_phase['phase'], A=100, w=5, phi=0.2, h=250)
        #             dely2 = res_sinu2.eval_uncertainty(sigma=3)
        #             best_pars2=res_sinu2.best_values
        #             init_pars2=res_sinu2.init_values
        #             xx2 = np.linspace(df2_phase['phase'].iloc[0], df2_phase['phase'].iloc[-1], num=100, endpoint=True)
        #             if Per == LS_P[0]:
        #                 out_sinu2 = open(path+'LS/'+name_+'_sinu_stats2.dat', 'w')
        #                 out_sinu2.write(name+'\n')
        #                 out_sinu2.write(res_sinu2.fit_report()+'\n')
        #                 out_sinu2.close()

        '''
        Plot the phased data, model and residuals
        '''
        if plots==True:
            #print('ploting fit to phased data')
            fig, ax = plt.subplots(nrows=2, ncols=1, figsize=(8, 6), sharex=True, gridspec_kw = {'height_ratios':[3, 1]})
            fig.subplots_adjust(left=0.12, right=0.97, top=0.93, bottom=0.11, hspace=0.)

            #ax[0].plot(xx, spl3(xx), 'g-',lw=6, label='Spline weighted', alpha=0.5)
            ax[0].errorbar(df_phase['phase'], df_phase['mean_rv'], df_phase['mean_rv_er'], fmt='.k',ms=12, ecolor='gray', label='data')
            # ax[0].plot(df_phase['phase'], res_sinu.init_fit, '--', c='gray', label='init')
            #ax[0].plot(df_phase['phase'], res_sinu.best_fit, 'r-', lw=2, label='best-fit')
            # ax[0].plot(xx, sinu(xx, init_pars['A'], init_pars['w'], init_pars['phi'], init_pars['h']), '--', c='gray', label='init')
            ax[0].plot(xx, sinu(xx, best_pars['A'], best_pars['w'], best_pars['phi'], best_pars['h']), '-', c='darkorange', lw=2, label='best-fit')
            # ax[0].fill_between(df_phase['phase'], res_sinu.best_fit-dely, res_sinu.best_fit+dely, color='gray', alpha=0.3, label='$3\sigma$ uncert.')#color="#ABABAB"
            ax[0].fill_between(fine_phase, fine_best_fit-dely, fine_best_fit+dely, color='darkorange', alpha=0.2, label='$3\sigma$ uncert.')
            #res_sinu.plot_fit(ax=ax[0], datafmt='ko', fitfmt='r-', initfmt='--', show_init=True)
            res_sinu.plot_residuals(ax=ax[1], datafmt='ko')
            #plt.plot(phase,diff,'ok', alpha=0.5)
            # if SB2==True:
            #     delta_per = []
            #     for period2 in peri2:
            #         delta_per.append(np.abs(Per-period2))
            #     for k, dif in enumerate(delta_per):
            #         if dif == min(delta_per) and peri2[k]>1.1 and Per>1.1:
            #             print(f'{Per:.2f}', '=', f'{peri2[k]:.2f}')
            #             ax[0].errorbar(df2_phase['phase'], df2_phase['mean_rv'], df2_phase['mean_rv_er'], fmt='.g',ms=12, ecolor='green', label='data2')
            #             ax[0].plot(xx2, sinu(xx2, best_pars2['A'], best_pars2['w'], best_pars2['phi'], best_pars2['h']), 'b-', lw=2, label='best-fit2')
            #             ax[0].fill_between(df2_phase['phase'], res_sinu2.best_fit-dely2, res_sinu2.best_fit+dely2, color='green', alpha=0.3, label='$3\sigma$ uncert.')#color="#ABABAB"
            #             res_sinu2.plot_residuals(ax=ax[1], datafmt='go')

            ax[0].set_title('P = '+str(f'{Per:.4f}')+'(d)')
            ax[0].set(ylabel='Radial Velocity (km s$^{-1}$)')

            ax[1].set(xlabel='Phase')
            ax[1].set(ylabel='Residuals')
            ax[0].legend(loc='best',markerscale=1., frameon=False)
            plt.savefig(path+'LS/'+name+'_velcurve_P'+str(f'{Per:.4f}')+'.pdf')
            plt.close()

            # if Per == LS_P[0]:
            # Plot for paper
            fig, ax = plt.subplots(figsize=(8, 6))
            fig.subplots_adjust(left=0.12, right=0.97, top=0.93, bottom=0.11)
            ax.errorbar(df_phase['phase'], df_phase['mean_rv'], df_phase['mean_rv_er'], fmt='ok', ms=6, ecolor='black', label='data')
            ax.plot(xx, sinu(xx, best_pars['A'], best_pars['w'], best_pars['phi'], best_pars['h']), 'r-', lw=2, label='best-fit')
            # ax.plot(df_phase['phase'], res_sinu.init_fit, '--', c='gray', label='init')
            plt.axhline(syst_RV, color='gray', linestyle=':',linewidth=1.5, label='systemic RV = {0:.2f}'.format(syst_RV))
            # if SB2==True:
            #     delta_per = []
            #     for period2 in peri2:
            #         delta_per.append(np.abs(Per-period2))
            #     for k, dif in enumerate(delta_per):
            #         if dif == min(delta_per):
            #         # if Per == peri2[k]:
            #             print(dif, min(delta_per))
            #             print(f'{Per:.2f}', '=', f'{peri2[k]:.2f}')
            #             ax.errorbar(df2_phase['phase'], df2_phase['mean_rv'], df2_phase['mean_rv_er'], fmt='sg', ms=6, ecolor='green', label='data2') # mfc='none',
            #             ax.plot(xx2, sinu(xx2, best_pars2['A'], best_pars2['w'], best_pars2['phi'], best_pars2['h']), 'b--', lw=2, label='best-fit2')
            # ax.set_title('P = '+str(f'{Per:.4f}')+'(d)',fontsize=22)
            ax.set_title(name)

            x1, x2 = ax.get_xlim()
            y1, y2 = ax.get_ylim()
            x_min = x1-(x2-x1)*0.05
            x_max = x2+(x2-x1)*0.05
            y_min = y1-(y2-y1)*0.1
            y_max = y2+(y2-y1)*0.1

            ax.set(xlim=(x_min, x_max), ylim=(y_min, y_max),
                    ylabel='Radial Velocity (km s$^{-1}$)',
                    xlabel='Orbital Phase')
            # ax[0].yaxis.label.set_size(22)
            # ax.legend(loc='best',numpoints=1,markerscale=1.,fontsize=12, handletextpad=0.5,borderaxespad=0.3, borderpad=0.8, frameon=False)
            if SB2==True:
                plt.savefig(path+'LS/'+name+'_paper_RV_curve_SB2.pdf')
            else:
                plt.savefig(path+'LS/'+name+'_paper_RV_curve_SB1.pdf')
            plt.close()













































        # # sys.exit()
        # '''
        # Lomb-Scargle
        # '''
        # if print_output==True:
        #     print( '*** Computing L-S ***')
        #     print( '---------------------')
        # if SB2==True:
        #     LS_P, LS_f, LS_Pow, LS_peak, LS_FAL, LS_peaks, allP, peri2, indi, LS_faps = lomb_scargle(df_rv['HJD'], df_rv['mean_rv'], df_rv['mean_rv_er'], \
        #     path, star, SB2=SB2, rv2=df_rv2['mean_rv'], rv2_err=df_rv2['mean_rv_er'], print_output=print_output, plots=plots, \
        #     best_lines=best_lines, min_sep=min_sep)
        # else:
        #     LS_P, LS_f, LS_Pow, LS_peak, LS_FAL, LS_peaks, allP, indi, LS_faps = lomb_scargle(df_rv['HJD'], df_rv['mean_rv'], df_rv['mean_rv_er'], \
        #     path, star, SB2=SB2, print_output=print_output, plots=plots, \
        #     best_lines=best_lines, min_sep=min_sep)
        #     # t7 = time.time()
        #     # print('# points + freq plot time: ', t7-t6)
        # #####################################################################

        # '''
        # Fit to periodogram peak for period error
        # '''
        # Perr=False
        # if Perr==True:
        #     pars = Parameters()
        #     usemod = 'loren'
        #     if usemod == 'gauss':
        #         prefix = 'g_'
        #         gauss = Model(gaussian, prefix=prefix)
        #         pars.update(gauss.make_params())
        #         mod = gauss
        #     if usemod == 'loren':
        #         prefix = 'l_'
        #         loren = Model(lorentzian, prefix=prefix)
        #         pars.update(loren.make_params())
        #         mod = loren

        #     # cont = models.LinearModel(prefix='continuum_')
        #     # pars.update(cont.make_params())
        #     # pars['continuum_slope'].set(0, vary=False)
        #     # pars['continuum_intercept'].set(1, min=0.9)

        #     if peaks1.any():
        #         if print_output==True:
        #             print('   # peaks1 = ', peaks1)
        #         pars.add(prefix+'amp', value=-np.max(power1[peaks1]), max=-np.max(power1[peaks1])*0.8)
        #     else:
        #         if print_output==True:
        #             print('   # peaks1 is empty')
        #         pars.add(prefix+'amp', value=-np.max(power1), max=-np.max(power1)*0.8)
        #     pars.add(prefix+'wid', value=0.001, max=0.003)
        #     pars.add(prefix+'cen', value=best_freq1, vary=True)
        #     mod = mod
        #     # flim = np.logical_and(frequency1 > 0.0025, frequency1 < 0.0031)
        #     results = mod.fit(power1, pars, x=frequency1)
        #     # results = mod.fit(power1[flim], pars, x=frequency1[flim])
        #     out_file = open(path+'LS/freqfit_stats.dat', 'w')
        #     out_file.write(results.fit_report()+'\n')
        #     out_file.close()
        #     # print(results.fit_report())
        #     freq_fwhm = results.params[prefix+'wid'].value
        #     P_err = (1/best_freq1**2)*(freq_fwhm/2)
        #     if print_output==True:
        #         print('   width of the peak               : ', f'{freq_fwhm:.5f}')
        #         print('   Error on the period             : ', f'{P_err:.5f}')
        #         print('   Final period                    : ', f'{best_period1:.4f}', '+/-', f'{P_err:.4f}')
        #     lsout.write('\n')
        #     lsout.write(' Width of the peak               :  '+ f'{freq_fwhm:.5f}'+'\n')
        #     lsout.write(' Error on the period             :  '+ f'{P_err:.5f}'+'\n')
        #     lsout.write(' Final period                    :  '+ f'{best_period1:.4f}'+' +/- '+f'{P_err:.4f}'+'\n')

        #     if SB2==True:
        #         if peaks2.any():
        #             if print_output==True:
        #                 print('   # peaks2 = ', peaks2)
        #             pars.add(prefix+'amp', value=-np.max(power2[peaks2]), max=-np.max(power2[peaks2])*0.8)
        #         else:
        #             if print_output==True:
        #                 print('   # peaks2 is empty')
        #             pars.add(prefix+'amp', value=-np.max(power2), max=-np.max(power2)*0.8)
        #         # pars.add(prefix+'amp', value=-np.max(power2[peaks2]), max=-np.max(power2[peaks2])*0.8)
        #         pars.add(prefix+'wid', value=0.001, max=0.003)
        #         pars.add(prefix+'cen', value=best_freq2, vary=True)
        #         mod = mod
        #         # flim = np.logical_and(frequency1 > 0.0025, frequency1 < 0.0031)
        #         results2 = mod.fit(power2, pars, x=frequency2)
        #         # results = mod.fit(power1[flim], pars, x=frequency1[flim])
        #         out_file2 = open(path+'LS/freqfit_stats_SB2.dat', 'w')
        #         out_file2.write(results2.fit_report()+'\n')
        #         out_file2.close()
        #         # print(results.fit_report())
        #         freq_fwhm2 = results2.params[prefix+'wid'].value
        #         P_err2 = (1/best_freq2**2)*(freq_fwhm2/2)
        #         if print_output==True:
        #             print('   width of the peak               : ', f'{freq_fwhm2:.5f}')
        #             print('   Error on the period             : ', f'{P_err2:.5f}')
        #             print('   Final period                    : ', f'{best_period2:.4f}', '+/-', f'{P_err2:.4f}')
        #         lsout.write('\n # For secondary \n')
        #         lsout.write(' Width of the peak               :  '+ f'{freq_fwhm2:.5f}'+'\n')
        #         lsout.write(' Error on the period             :  '+ f'{P_err2:.5f}'+'\n')
        #         lsout.write(' Final period                    :  '+ f'{best_period2:.4f}'+' +/- '+f'{P_err2:.4f}'+'\n')

        #     if plots==True:
        #         if best_period1 > 100:
        #             x = 2
        #         else:
        #             x = 1.5
        #         fig, ax = plt.subplots(figsize=(8, 6))
        #         fig.subplots_adjust(left=0.12, right=0.97, top=0.93, bottom=0.12)
        #         ax.plot(frequency1, power1, 'k-', alpha=0.5)
        #         ax.plot(frequency1, results.init_fit, '--', c='grey')
        #         # ax.plot(frequency1[flim], results.init_fit, '--', c='grey')
        #         ax.plot(frequency1, results.best_fit, 'r-', lw=2)
        #         # ax.plot(frequency1[flim], results.best_fit, 'r-', lw=2)
        #         ax.set(xlim=(best_freq1-0.001*x, best_freq1+0.001*x),
        #             xlabel='Frequency (1/d)',
        #             ylabel='Lomb-Scargle Power');
        #         plt.savefig(path+'LS/period_fit'+name_+'_primary.pdf')
        #         plt.close()
        #         if SB2==True:
        #             fig, ax = plt.subplots(figsize=(8, 6))
        #             fig.subplots_adjust(left=0.12, right=0.97, top=0.93, bottom=0.12)
        #             ax.plot(frequency2, power2, 'k-', alpha=0.5)
        #             ax.plot(frequency2, results2.init_fit, '--', c='grey')
        #             # ax.plot(frequency1[flim], results.init_fit, '--', c='grey')
        #             ax.plot(frequency2, results2.best_fit, 'r-', lw=2)
        #             # ax.plot(frequency1[flim], results.best_fit, 'r-', lw=2)
        #             ax.set(xlim=(best_freq1-0.001*x, best_freq1+0.001*x),
        #                 xlabel='Frequency (1/d)',
        #                 ylabel='Lomb-Scargle Power');
        #             plt.savefig(path+'LS/period_fit_'+name_+'_secondary.pdf')
        #             plt.close()

        # # if print_output==True:
        # #     t8 = time.time()
        # #     print('# P_err + plot time: ', t8-t7)
        # #####################################################################

 

        # '''
        # Plot RVs vs HJD
        # '''
        # #df_rv = df_rv.drop([11, 12, 13, 15]).reset_index()
        # a = min(df_rv['HJD'])
        # b = max(df_rv['HJD'])
        # y1 = min(df_rv['mean_rv'])
        # y2 = max(df_rv['mean_rv'])
        # delta_RV=y2-y1
        # delta_RV_er=np.sqrt( df_rv['mean_rv_er'][df_rv['mean_rv'].idxmax()]**2 + df_rv['mean_rv_er'][df_rv['mean_rv'].idxmin()]**2 )
        # if SB2==True:
        #     y1sec = min(df_rv2['mean_rv'])
        #     y2sec = max(df_rv2['mean_rv'])
        #     delta_RVsec=y2sec-y1sec
        #     delta_RVsec_er=np.sqrt( df_rv2['mean_rv_er'][df_rv2['mean_rv'].idxmax()]**2 + df_rv2['mean_rv_er'][df_rv2['mean_rv'].idxmin()]**2 )
        # if plots==True:
        #     fig, ax = plt.subplots(figsize=(8,6))
        #     fig.subplots_adjust(left=0.12, right=0.97, top=0.93, bottom=0.12)
        #     plt.errorbar(df_rv['HJD'], df_rv['mean_rv'], df_rv['mean_rv_er'], fmt='.', mfc='orange', mec='orange',ms=12, ecolor='orange', capsize=5, label='RV per epoch')
        #     if SB2==True:
        #         plt.errorbar(df_rv2['HJD'], df_rv2['mean_rv'], df_rv2['mean_rv_er'], fmt='.g',ms=12, ecolor='green', capsize=5, alpha=0.6, label='RV per epoch (sec)')
        #     plt.hlines(syst_RV, a-150, b+150, color='blue', linestyles='--',linewidth=1., label='systemic RV = {0:.2f}'.format(syst_RV))
        #     plt.title(name+' $-$ '+str(len(df_rv))+' epochs',fontsize=22)
        #     # plt.tick_params(which='both', width=0.6, labelsize=14)
        #     #plt.axis([a-150, b+150, y1-10, y2+20])
        #     ax.set(xlim=(a-100, b+100))
        #     plt.ylabel(r'Radial Velocity (km s$^{-1}$)',fontsize=22)
        #     plt.xlabel(r'HJD (days)',fontsize=22)
        #     #plt.legend(loc='best',numpoints=1,markerscale=0.8,fontsize=9,fancybox='true',shadow='true',handletextpad=1,borderaxespad=2)
        #     plt.legend(loc='best',fontsize=18, handletextpad=0.4, borderaxespad=1, edgecolor='k')
        #     #ax.ticklabel_format(usecont=False)
        #     plt.savefig(path+'LS/'+name_+'_epochs.pdf')
        #     plt.close()
        # # if print_output==True:
        # #     t9 = time.time()
        # #     print('# compute phases and plot time: ', t9-t8)
        # #####################################################################
