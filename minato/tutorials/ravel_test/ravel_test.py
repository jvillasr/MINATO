import os
import glob
import time
import pandas as pd
# sys.path.append('/Users/villasenor/science/github/jvillasr/MINATO')
from minato import ravel, myRC

start_time = time.time()

results_file = 'RVs_results.csv'

# Load the results from the file if it exists
if os.path.exists(results_file):
    results_df = pd.read_csv(results_file, sep='\t')
else:
    # Initialize an empty DataFrame to store the results
    results_df = pd.DataFrame(columns=['Star_field_ID', 'nepochs', 'dRV', 'sigma_d', 'Bin', 'Period', 'LS_Pow', 'P_qflag'])
# print('results_df: ', results_df)


# path='sb1_data/'
# path='sb2_data_1/'
# path='sb2_data_5_080/'
# path='sb2_data_8_057/'
path='sb2_data_8_093/'
''
sb2=True
spectrum_files = sorted(glob.glob(f"{path}*Combined.fits"))
spectrum_files = spectrum_files#[0:4]

# print('spectrum_files: ', spectrum_files)
# lines = [4026, 4102, 4131, 4144, 4340, 4388, 4471, 4481, 4542, 4553]
# lines = [4026, 4102, 4144, 4340, 4388, 4471]
lines = [4026, 4144, 4388, 4471]
# lines = [4026, 4102, 4471]
star_path = ravel.SLfit(spectrum_files, path='', lines=lines, K=2, SB2=sb2, plots=True, shift=0, neblines=[])
# print('star_path from SLfit: ', star_path)
# star_path = 'BLOeM_1-055/SB2/'
# star_path = 'BLOeM_1-037/SB2/'
# star_path = 'BLOeM_5-080/SB2_2/'
# star_path = 'BLOeM_8-057/SB2_2/'
# star_path = 'BLOeM_8-093/SB2_2/'

# rv_analysis = ravel.GetRVs('fit_values.csv', star_path, star_path+'JDs.txt', rm_epochs=True, lines_ok_thrsld=2, 
#                        epochs_ok_thrsld=4, SB2=True)#, use_lines=[4340])
# re_results = rv_analysis.compute()


# Compute the orbital period with the Lomb-Scargle periodogram
rv_results = pd.read_csv(star_path+'fit_values.csv', sep=',', comment='#')
ls_results = ravel.lomb_scargle(rv_results, star_path, print_output=True, plots=True, SB2=True, best_lines=False)

# Compute the RV curve assuming sinusoidal model
# ravel.phase_rv_curve(re_results, periods=ls_results[6], path=star_path, SB2=True)


end_time = time.time()
execution_time = end_time - start_time

print('Execution time: ', execution_time)