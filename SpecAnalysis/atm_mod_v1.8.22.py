import sys
import os
import itertools
import myRC
import pprint
import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import scipy.interpolate as inter
from scipy.optimize import minimize
from glob import glob
from timeit import default_timer as timer
from datetime import timedelta, date
pp = pprint.PrettyPrinter(indent=4)
current_date = str(date.today())

# def rotin3_inp(inpfile, path, model, T, g, rot, vt):
def rotin3_inp(grid, T, g, rot):
    '''
    grid: 'earlyB', 'lateB', 'O'
    '''
    import csv
    import os
    import subprocess

    # lowT_models_path = '/home/jaime/science/KUL/atm_models/ATLAS9/Jaime/'
    # tlustyB_path = '/home/jaime/science/KUL/atm_models/TLUSTY/BLvispec_v2/'
    # tlustyO_path = '/home/jaime/science/KUL/atm_models/TLUSTY/OLvispec_v10/'

    lowT_models_path = '/Users/jaime/Science/KUL_postdoc/KUL_research/models/ATLAS9/Jaime/'
    tlustyB_path = '/Users/jaime/Science/KUL_postdoc/KUL_research/models/tlusty/BLvispec_v2/'
    tlustyO_path = '/Users/jaime/Science/KUL_postdoc/KUL_research/models/tlusty/OLvispec_v10/'
#
    H11_rotin = lowT_models_path+'r.dat'
    tlB_rotin = tlustyB_path+'291B.dat'
    tlO_rotin = tlustyO_path+'291B.dat'

    if grid == 'earlyB':
        inpfile = tlB_rotin
        path    = tlustyB_path
        model   = 'BL'
        vt      = '2'
        models_list = sorted(glob(path+'*v'+vt+'.vis.7'))
    elif grid == 'lateB':
        inpfile = H11_rotin
        path    = lowT_models_path
        model   = 't'
        models_list = sorted(glob(path+'*F.dat'))
    elif grid == 'O':
        inpfile = tlO_rotin
        path    = tlustyO_path
        model   = 'L'
        vt      = '10'
        models_list = sorted(glob(path+'*v'+vt+'.vis.7'))

    models_list = [x.replace(path, '') for x in models_list]

    # BL23000g300v2.vis.7        earlyB
    # t11000g26F.dat             lateB
    # L35000g450v10.vis.7        O

    df = pd.read_csv(inpfile, sep='\s+', header=None)
    # print(df)
    if grid=='earlyB' or grid=='O':
        df.loc[0, 0] = '\''+model+str(int(T*1000))+'g'+str(int(g*10))+'v'+str(vt)+'.vis.7\''
        # print(df.loc[0, 0].replace('\'', ''))
        # print(models_list[16])
        if df.loc[0, 0].replace('\'', '') in models_list:
            # print('Model', df.loc[0, 0], ' saved')
            df.loc[0, 1] = '\''+model+str(int(T*1000))+'g'+str(int(g*10))+'v'+str(vt)+'.vis.17\''
            df.loc[0, 2] = '\'T'+str(int(T*10))+'g'+str(int(g*10))+'v'+str(vt)+'r'+str(rot)+'fw05\''
            if grid=='O':
                df.loc[0, 2] = '\'T'+str(int(T*10))+'g'+str(int(g*10))+'v'+str(vt)+'r'+str(int(rot))+'fw05\''
            df.loc[1, 0] = rot
            # print(df)
            df.to_csv(inpfile, sep='\t', index=False, header=False)
            os.chdir(path)
            # print('path =', path)
            p=subprocess.run(['./rotin3.out < '+inpfile], text=True, check=True, shell=True)
            print('Model', df.loc[0, 2], ' saved')
    elif grid=='lateB':
        # print(inpfile)
        df.loc[0, 0] = '\''+model+str(int(T*1000))+'g'+str(int(g))+'F.dat\''
        # print(df.loc[0, 0].replace('\'', ''))
        if df.loc[0, 0].replace('\'', '') in models_list:
            df.loc[0, 1] = '\''+model+str(int(T*1000))+'g'+str(int(g))+'C.dat\''
            df.loc[0, 2] = '\'T'+str(T)+'g'+str(int(g))+'v2'+'r'+str(rot)+'fw05\''
            df.loc[1, 0] = rot
            # print(df)
            df.to_csv(inpfile, sep='\t', index=False, header=False)
            os.chdir(path)
            p=subprocess.run(['./rotin3.out < '+inpfile], text=True, check=True, shell=True)
            print('Model', df.loc[0, 0], ' saved')
        # else:
        #     print('model is not in models_list')

def fitplot(wA, fA, wM, fM, model, lr, dictionary, lines, figu='save', nrows=3, ncols=3, legend_ax=3,
            xlabel_ax=7, ylabel_ax=3, balmer_min_y=0.75):
    '''
    wA, fA : wavelength and flux of the observed spectrum.
    wM, fM : wavelength and flux of the mdoels (list).
    model  : name/identifier of the models. Used for labels in legend (list).
    lr     : light ratio contribution from the secondary. Used in figure title and name of the saved plot.
    dictionary : Python dictionary with name/identifier on the spectral lines, the region and title for each subplot.
    lines  : lines used in the dictionary (list).
    #savefig : default False. True will save the figure (bool).
    figu : default 'save'. Use 'show' to show the plot without saving it.
    nrows, ncols : number of rows and columns for subplots (int).
    legend_ax : number of the preferred subplot to display the legend (int).
    '''
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(8*ncols, 6*nrows), sharey=False)
    if type(axes)==np.ndarray:
        ax = axes.flatten()
    else:
        ax = [axes]
    for i, line in enumerate(dictionary):
        reg = dictionary[line]['region']
        cond = (wA > reg[0]) & (wA < reg[1])
        ax[i].plot(wA[cond], fA[cond],c='k',ls='-', linewidth=4, label='disent. spec')
        # ax[i].plot(wA[cond], fA[cond], 'ko', ms=8,ls='none', label='disent. spec')
        for f, w, mod in zip(fM, wM, model):
            cond = (w > reg[0]) & (w < reg[1])
            ax[i].plot(w[cond], f[cond],'--', c='orange', linewidth=4, label=mod)
            # ax[i].plot(w[cond], f[cond],'.', linewidth=2, label=mod)
        if line in [4102, 4340]:
            ax[i].set_ylim(balmer_min_y, 1.05)
        ax[i].set_title(dictionary[line]['title'], size=36)
        ax[i].tick_params(axis='both', which='major', labelsize=32)
    ax[legend_ax].legend(frameon=False, fontsize=20)
    fig.supxlabel(r'Wavelength (\AA)', size=48)
    fig.supylabel(r'Flux', x=0.01, size=48)
    fig.suptitle('Secondary light contribution = '+str(int(lr))+'\%'+' - fitted lines: '+str(lines), y=1, fontsize=36)
    plt.tight_layout()
    if figu=='save':
    # plt.savefig(model+'lr'+str(lr)+'_'+str(lines)+'.pdf')
        plt.savefig(str(model[0])+'_lr'+str(int(lr))+'.pdf')
    # plt.show()
    plt.close()

def rescale_flux(fluxA, fluxB, ratio0, ratio1):
    flux_new_A = (fluxA -1)*((1-ratio0)/(1-ratio1)) + 1
    flux_new_B = (fluxB -1)*(ratio0/ratio1) + 1
    return flux_new_A, flux_new_B

def crop_data(x_data, y_data, range1, range2):
    cond = (x_data < range1) | (x_data > range2)
    return x_data[cond], y_data[cond]

def chi2(obs, exp):
    return np.sum(((obs-exp)**2)/exp)

def slicedata(x_data, y_data, dictionary):
    x_data = pd.Series(x_data)
    y_data = pd.Series(y_data)
    x_data_sliced = []
    y_data_sliced = []
    for line in dictionary:
        reg = dictionary[line]['region']
        cond = (x_data > reg[0]) & (x_data < reg[1])
        x_data_sliced.append(np.array(x_data[cond]))
        y_data_sliced.append(np.array(y_data[cond]))
    return x_data_sliced, y_data_sliced

def He2H_ratio(wave, flux, ratio0, ratio1, dictionary, join=False):
    new_spetrum = []
    '''
    Modifies the He/H ratio
    It works on sliced data. Returns sliced data unless 'join' is True.
    '''
    for i,line in enumerate(dictionary):
        # print(line, len(wave[i]), len(flux[i]))
        # reg = dictionary[line]['region']
        # cond = (wave > reg[0]) & (wave < reg[1])
        if line in [4026, 4144, 4388]:
            new_spetrum.append( (flux[i] -1)*(ratio1/ratio0) + 1 )
        elif line==4121:
            reg_4121 = []
            cond1 = wave[i] < 4120
            cond2 = (wave[i] > 4120) & (wave[i] < 4122)
            cond3 = wave[i] > 4122
            # plt.plot(wave[i][cond1], flux[i][cond1])
            reg_4121.append( flux[i][cond1] )
            reg_4121.append( (flux[i][cond2] -1)*(ratio1/ratio0) + 1 )
            reg_4121.append( flux[i][cond3] )
            new_spetrum.append(np.array(list(itertools.chain.from_iterable(reg_4121))))
        elif line==4471:
            reg_4471 = []
            cond1 = wave[i] < 4468
            cond2 = (wave[i] > 4468) & (wave[i] < 4475)
            cond3 = wave[i] > 4475
            # plt.plot(wave[i][cond1], flux[i][cond1])
            reg_4471.append( flux[i][cond1] )
            reg_4471.append( (flux[i][cond2] -1)*(ratio1/ratio0) + 1 )
            reg_4471.append( flux[i][cond3] )
            new_spetrum.append(np.array(list(itertools.chain.from_iterable(reg_4471))))
        elif line in [4102, 4340]:
            new_spetrum.append(  (flux[i] -1)*((1-ratio1)/(1-ratio0)) + 1 )
        else:
            new_spetrum.append(flux[i])
    if join==True:
        new_spetrum = np.array(list(itertools.chain.from_iterable(new_spetrum)))
        new_wave = np.array(list(itertools.chain.from_iterable(wave)))
        return new_wave, new_spetrum
    else:
        return new_spetrum

def get_model(T, g, rot, source='tlusty'):
    '''
    source : Source of the models. Options are 'tlusty' and 'atlas'.
    '''

    # lowT_models_path = '/home/jaime/science/KUL/atm_models/ATLAS9/Jaime/'
    # tlustyB_path = '/home/jaime/science/KUL/atm_models/TLUSTY/BLvispec_v2/'
    # tlustyO_path = '/home/jaime/science/KUL/atm_models/TLUSTY/OLvispec_v10/'

    lowT_models_path = '/Users/jaime/Science/KUL_postdoc/KUL_research/models/ATLAS9/Jaime/'
    tlustyB_path = '/Users/jaime/Science/KUL_postdoc/KUL_research/models/tlusty/BLvispec_v2/'
    tlustyO_path = '/Users/jaime/Science/KUL_postdoc/KUL_research/models/tlusty/OLvispec_v10/'


    # lowT_models_path = '/Users/jaime/Science/KUL_postdoc/KUL_research/models/ATLAS9/Jaime/'
    lowT_models_list = sorted(glob(lowT_models_path+'*fw05'))
    lowT_models_list = [x.replace(lowT_models_path, '') for x in lowT_models_list]
    # tlustyB_path = '/Users/jaime/Science/KUL_postdoc/KUL_research/models/tlusty/BLvispec_v2/'
    tlustyB_list = sorted(glob(tlustyB_path+'*fw05'))
    tlustyB_list = [x.replace(tlustyB_path, '') for x in tlustyB_list]
    # tlustyO_path = '/Users/jaime/Science/KUL_postdoc/KUL_research/models/tlusty/OLvispec_v10/'
    tlustyO_list = sorted(glob(tlustyO_path+'*fw05'))
    tlustyO_list = [x.replace(tlustyO_path, '') for x in tlustyO_list]
    tlustyOB_list = tlustyB_list + tlustyO_list
    model = 'T'+str(int(T))+'g'+str(int(g*10))+'v2r'+str(int(rot))+'fw05'
    if T>30:
        model = 'T'+str(int(T*10))+'g'+str(int(g*10))+'v10r'+str(int(rot))+'fw05'
        # print('el modelo problema', model)

    if source=='tlusty':
        # if model in tlustyOB_list:
        try:
            if T>30 and g!=30:
                df = pd.read_csv(tlustyO_path+model,header=None, sep='\s+')
                # print(df)
            else:
                df = pd.read_csv(tlustyB_path+model,header=None, sep='\s+')
            return df[0].array, df[1].array, model
        except FileNotFoundError:
            pass
        # else:
        #     print('WARNING: No model named ', model, ' was found')
        #     if T>30 and g!=30:
        #         try:
        #             rotin3_inp('O', T, g, rot)
        #             df = pd.read_csv(tlustyO_path+model,header=None, sep='\s+')
        #             # print(df[0])
        #             return df[0].array, df[1].array, model
        #         except FileNotFoundError:
        #             raise ValueError('   WARNING: No model available for '+model)
        #     else:
        #         try:
        #             rotin3_inp('earlyB', T, g, rot)
        #             df = pd.read_csv(tlustyB_path+model,header=None, sep='\s+')
        #             return df[0].array, df[1].array, model
        #         except FileNotFoundError:
        #             raise ValueError('   WARNING: No model available for '+model)

    elif source=='atlas':
        model = 'T'+str(int(T))+'g'+str(int(g))+'v2r'+str(int(rot))+'fw05'
        # if model in lowT_models_list:
        try:
            df = pd.read_csv(lowT_models_path+model,header=None, sep='\s+')
            return df[0].array, df[1].array, model
        except FileNotFoundError:
            pass
        # else:
        #     try:
        #         print('WARNING: No model named ', model, ' was found')
        #         rotin3_inp('lateB', T, g, rot)
        #         df = pd.read_csv(lowT_models_path+model,header=None, sep='\s+')
        #         return df[0].array, df[1].array, model
        #     except FileNotFoundError:
        #         # print('WARNING: No model named '+model+' was found')
        #         raise ValueError('   WARNING: No model available for '+model)


def min_chi2(params, T1, g1, r1, T2, g2, r2, dicA, dicB):
    # dsnt_A = pd.read_csv('/home/jaime/science/KUL/VFTS291/ADIS_lguess_K1K2=0.3_94.0_15.0_renorm.txt', header=None, sep='\s+')
    # dsnt_B = pd.read_csv('/home/jaime/science/KUL/VFTS291/BDIS_lguess_K1K2=0.3_94.0_15.0.txt', header=None, sep='\s+')

    dsnt_A = pd.read_csv('/Users/jaime/Science/KUL_postdoc/BBC/291/tomer/ADIS_lguess_K1K2=0.3_94.0_15.0_renorm.txt', header=None, sep='\s+')
    dsnt_B = pd.read_csv('/Users/jaime/Science/KUL_postdoc/BBC/291/tomer/BDIS_lguess_K1K2=0.3_94.0_15.0.txt', header=None, sep='\s+')
    nparams = 7
    # print('Called with', params, T1, g1, T2, g2, r2)
    lr, he2h = params
    #
    dst_A_wave = dsnt_A[0]-0.2
    dst_B_wave = dsnt_B[0]-0.2

    dst_A_flux, dst_B_flux = rescale_flux(dsnt_A[1], dsnt_B[1], 0.3, lr)
    #
    dst_A_x, dst_A_y = slicedata(dst_A_wave, dst_A_flux, dicA)
    dst_B_x, dst_B_y = slicedata(dst_B_wave, dst_B_flux, dicB)

    try:
        if T1 < 16:
            modA_x, modA_y, modelA = get_model(T1, g1, r1, source='atlas')
        else:
            modA_x, modA_y, modelA = get_model(T1, g1, r1, source='tlusty')
        modB_x, modB_y, modelB = get_model(T2, g2, r2, source='tlusty')

    # print(modB_x)
    # print(modB_y[-1])
    # print(g1)
    # if g1 == 28:
    #     print(modelA, modA_y)
    #     sys.exit()

#     if not modA_y:
#         # print('\n ######### MODEL NOT FOUND', T1, g1, T2, g2, r2, '\n')
#         print('\n ######### MODEL NOT FOUND', T1, g1, '\n')
#         rotin3_inp('lateB', T1, g1, 35)
#         modA_x, modA_y, modelA = get_model(T1, g1, 35, 'atlas')
#     elif not modB_y:
#         if T2>30:
#             rotin3_inp('O', T2, g2, r2)
#         else:
#             rotin3_inp('earlyB', T2, g2, r2)
#         modB_x, modB_y, modelB = get_model(T2, g2, r2, 'tlusty')

#     if not modA_y:
#         print('\n ######### MODEL NOT FOUND', T1, g1, '\n')
#     if not modB_y:
#         print('\n ######### MODEL NOT FOUND', T2, g2, r2, '\n')

        if modA_y and modB_y:
            splA = inter.UnivariateSpline(modA_x, modA_y)
            splA.set_smoothing_factor(0.)
            spl_fluxA = [splA(x) for x in dst_A_x]
            #
            splB = inter.UnivariateSpline(modB_x, modB_y)
            splB.set_smoothing_factor(0.)
            spl_fluxB = [splB(x) for x in dst_B_x]

            spl_fluxA = He2H_ratio(dst_A_x, spl_fluxA, 0.075, he2h, dicA)
            # spl_fluxB = He2H_ratio(dst_B_x, spl_fluxB, 0.1, he2h, dicB)

            chi2A, chi2B, ndataA, ndataB = 0, 0, 0, 0
            for i,line in enumerate(dicB):
                if line == 4102:
                    dst_B_x_crop, dst_B_y_crop = crop_data(dst_B_x[i], dst_B_y[i], 4098, 4105)
                    spl_wavB_crop, spl_fluxB_crop = crop_data(dst_B_x[i], spl_fluxB[i], 4098, 4105)
                elif line == 4340:
                    dst_B_x_crop, dst_B_y_crop = crop_data(dst_B_x[i], dst_B_y[i], 4334, 4347)
                    spl_wavB_crop, spl_fluxB_crop = crop_data(dst_B_x[i], spl_fluxB[i], 4334, 4347)
                else:
                    dst_B_x_crop, dst_B_y_crop = dst_B_x[i], dst_B_y[i]
                    spl_wavB_crop, spl_fluxB_crop = dst_B_x[i], spl_fluxB[i]
                ndataB += len(spl_fluxB_crop)
                # if line==4553:
                #     print('wave =', spl_wavB_crop)
                #     print('flux model =', spl_fluxB[i])
                #     print('flux disen =', dst_B_y_crop)
                # print(line, chi2(dst_B_y_crop, spl_fluxB_crop))
                chi2B += chi2(dst_B_y_crop, spl_fluxB_crop)
            # print('number of data point of spectrum B', ndataB)
            for i,line in enumerate(dicA):
                # print(line, chi2(dst_A_y[i], spl_fluxA[i]))
                chi2A += chi2(dst_A_y[i], spl_fluxA[i])
                ndataA += len(spl_fluxA[i])
                # print(chi2A, chi2B)
                # chisqr = chi2A + chi2B
                # print(line, chisqr)
                # print(line, chisqr/len(dst_B_y_crop), len(dst_B_y_crop))
            chi2_tot = chi2A + chi2B
            # print('number of data point of spectrum A', ndataA)
            ndata = ndataA + ndataB
            chi2redA = chi2A/(ndataA-nparams)
            chi2redB = chi2B/(ndataB-nparams)
            chi2r_tot = chi2redA + chi2redB
            # print('total number of data point', ndata)
            # print('chi2 =', chi2_tot)
            # print('\n')
            # gridB.append([modelB, lr, T2, g2, rv, chi2_tot])
            return chi2_tot, chi2A, chi2B, chi2r_tot, chi2redA, chi2redB
    except TypeError:
        pass

# dsnt_A = pd.read_csv('/home/jaime/science/KUL/VFTS291/ADIS_lguess_K1K2=0.3_94.0_15.0_renorm.txt', header=None, sep='\s+')
# dsnt_B = pd.read_csv('/home/jaime/science/KUL/VFTS291/BDIS_lguess_K1K2=0.3_94.0_15.0.txt', header=None, sep='\s+')

dsnt_A = pd.read_csv('/Users/jaime/Science/KUL_postdoc/BBC/291/tomer/ADIS_lguess_K1K2=0.3_94.0_15.0_renorm.txt', header=None, sep='\s+')
dsnt_B = pd.read_csv('/Users/jaime/Science/KUL_postdoc/BBC/291/tomer/BDIS_lguess_K1K2=0.3_94.0_15.0.txt', header=None, sep='\s+')
# lines_dic = {
#                 4009: { 'region':[4005, 4015], 'title':'He I $\lambda$4009'},
#                 4026: { 'region':[4019, 4033], 'title':'He I $\lambda$4026'},
#                 4102: { 'region':[4084, 4117], 'title':'H$\delta$'},
#                 4121: { 'region':[4117, 4124], 'title':'He I $\lambda$4121'},
#                 4144: { 'region':[4137, 4151], 'title':'He I $\lambda$414'},
#                 4340: { 'region':[4323-10, 4359+10], 'title':'H$\gamma$'},
#                 4388: { 'region':[4380, 4396], 'title':'He I $\lambda$4388'},
#                 4471: { 'region':[4465, 4485], 'title':'He I $\lambda$4471, Mg II $\lambda$4481'},
#                 4553: { 'region':[4538, 4562], 'title':'Fe II, Si III'} }

lines_dic = {
                4026: { 'region':[4005, 4033], 'title':'He I $\lambda$4009/26'},
                4102: { 'region':[4084-20, 4117], 'title':'H$\delta$'},
                4121: { 'region':[4117, 4135], 'title':'He I $\lambda$4121, Si II $\lambda$4128/32'},
                4144: { 'region':[4137, 4151], 'title':'He I $\lambda$414'},
                4267: { 'region':[4260, 4275], 'title':'C II $\lambda$4267'},
                4340: { 'region':[4320, 4362], 'title':'H$\gamma$'},
                4388: { 'region':[4380, 4396], 'title':'He I $\lambda$4388'},
                4471: { 'region':[4465, 4485], 'title':'He I $\lambda$4471, Mg II $\lambda$4481'},
                4553: { 'region':[4536, 4560], 'title':'Fe II, Si III'} }

# select_lines = [4009, 4026, 4102, 4121, 4144, 4340, 4388, 4471, 4553]
select_linesA = [4026, 4102, 4121, 4144, 4267, 4340, 4388, 4471, 4553]
select_linesB = [4026, 4102, 4121, 4144, 4267, 4340, 4388, 4553]
# select_linesA = [4026, 4102, 4121, 4144, 4267, 4340]
# select_linesB = [4026, 4121, 4144, 4267, 4388, 4553]
user_dicA = { line: lines_dic[line] for line in select_linesA }
user_dicB = { line: lines_dic[line] for line in select_linesB }

test=False
if test==True:
    teffA = [11, 12, 13, 14, 15]
    # teffA = [16]
    loggA = [18, 20, 22, 24, 26, 28]
    # loggA = [20, 22.5, 25, 27.5, 30]
    rotA = [20, 60]
    teffB = [24, 26, 28, 30, 32.5, 35, 37.5]
    loggB = [25, 27.5, 30, 32.5, 35, 37.5, 40, 42.5, 45]
    rotB = range(100, 600, 100)
    lrat = np.linspace(0.25, 0.5, 6)
    # lrat = [0.25]
    he2hrat = [0.05, 0.065, 0.08, 0.1, 0.12, 0.14]
    # he2hrat = [0.05]

else:
    teffA = [11, 12, 13, 14, 15, 16, 17, 18]
    loggA = [18, 20, 22, 24, 26, 28, 30]
    rotA = range(30, 45, 5)
    # teffB = [20, 22, 24, 26, 28, 30, 32.5, 35, 37.5, 40]
    teffB = [16, 20, 24, 26, 28, 30, 32.5, 35, 37.5]
    # loggB = [25, 27.5, 30, 32.5, 35, 37.5, 40, 42.5, 45, 47.5]
    loggB = [25, 27.5, 30, 32.5, 35, 37.5, 40, 42.5, 45]
    rotB = range(100, 600, 100)
    lrat = np.linspace(0.25, 0.5, 6)
    # he2hrat = np.linspace(0.065, 0.1, 8)
    he2hrat = [0.05, 0.065, 0.08, 0.1, 0.12, 0.14]

print('number of models:', len(lrat)*len(he2hrat)*len(teffA)*len(loggA)*len(rotA)*len(teffB)*len(loggB)*len(rotB))

# sys.exit()

start = timer()

###############################################################################
# # Recomputing models with negative chi2
# tab_negative=pd.read_csv('/Users/jaime/Science/KUL_postdoc/BBC/291/atm_fitting/minchi2_result_he2honA_2022-08-01_0.csv')
# tab = tab_negative[tab_negative['chi2']<0]
# print('number of models =', len(tab))
#
# grid_res, result = [], []
# for i in range(len(tab)):
#     lr, heh, T1, g1, r1, T2, g2, r2 = tab['lrat'].iloc[i], tab['He2H'].iloc[i], tab['TeffA'].iloc[i], tab['loggA'].iloc[i], tab['rotA'].iloc[i] , \
#                                         tab['TeffB'].iloc[i], tab['loggB'].iloc[i], tab['rotB'].iloc[i]
#     print(len(tab)-i, lr, heh, T1, g1, r1, T2, g2, r2)
#     fix_pars = [lr, heh]
#     result_temp = min_chi2(fix_pars, T1, g1, r1, T2, g2, r2, user_dicA, user_dicB)
#     chi2_res, chi2A, chi2B, chi2red, chi2rA, chi2rB = result_temp
#     grid_res.append([lr, heh, T1, g1, r1, T2, g2, r2, chi2_res, chi2A, chi2B, chi2red, chi2rA, chi2rB])
###############################################################################

# fix_pars=[0.352178, 0.111729]
grid_res, result = [], []
for lr in lrat:
    for heh in he2hrat:
        for T1 in teffA:
            if T1 < 16:
                for g1 in loggA:
            else:
                for g1 in [20, 22.5, 25, 27.5, 30]
            for g1 in loggA:
                for r1 in rotA:
                    for T2 in teffB:
                        print('Computing chi2 for lrat =', lr, 'He/H =', heh, 'Teff_A =', T1, 'logg_A =', g1, 'rot_A =', r1, 'Teff_B =', T2)
                        for g2 in loggB:
                            # print('Computing chi2 for lrat =', lr, 'Teff_A =', T1, 'logg_A =', g1, 'Teff_B =', T2, 'logg_B =', g2)
                            for r2 in rotB:
                                try:
                                    fix_pars = [lr, heh]
                                    # result_temp = minimize(min_chi2, x0=(0.2), args=(T1, g1, T2, g2, r2), bounds=[(0.05, 0.7)], method='Nelder-Mead', options={'disp':False, 'return_all':False})
                                    # result_temp = minimize(min_chi2, x0=(0.3, 0.12), args=(T1, g1, T2, g2, r2), bounds=[(0.05, 0.7), (0.05, 0.2)], method='Nelder-Mead', options={'disp':False, 'return_all':False})
                                    result_temp = min_chi2(fix_pars, T1, g1, r1, T2, g2, r2, user_dicA, user_dicB)
                                    # print('result', result_temp)
                                    # result.append(result_temp)
                                    chi2_res, chi2A, chi2B, chi2red, chi2rA, chi2rB = result_temp
                                    # print('chi2 =', chi2_res)
                                    # lr = result_temp.x[0]
                                    # he2h = result_temp.x[1]
                                    grid_res.append([lr, heh, T1, g1, r1, T2, g2, r2, chi2_res, chi2A, chi2B, chi2red, chi2rA, chi2rB])
                                except (ValueError, TypeError):
                                    pass
            end_T1 = timer()
            print('    For lrat =', lr, 'He/H =', heh, 'Teff_A =', T1, 'Teff_B =', T2, 'completed in', timedelta(seconds=end_T1-start))
    end_lr = timer()
    print('        Light ratio', lr, 'completed in', timedelta(seconds=end_lr-start), '\n')


end = timer()
print(timedelta(seconds=end-start))
# print(os.getcwd())
os.chdir('/Users/jaime/Science/KUL_postdoc/BBC/291/atm_fitting/')
df_results = pd.DataFrame(grid_res, columns=['lrat', 'He2H', 'TeffA', 'loggA', 'rotA', 'TeffB', 'loggB', 'rotB', 'chi2', 'chi2A', 'chi2B', 'red_chi2', 'red_chi2A', 'red_chi2B'])
df_results_sort = df_results.sort_values(by=['chi2'], ignore_index=True).copy()
df_results_sort.to_csv('minchi2_result_he2honA_'+current_date+'_extra-rotA_2.csv', index=False)
print(df_results_sort[:30])
print('length of results:', len(df_results_sort))

# '''
# Disentangled spectrum rescaled
# '''
# T1_res, g1_res, T2_res, g2_res, r2_res, lr_res, chi2_res = df_results_sort.loc[0].to_list()
# f_A, f_B = rescale_flux(dsnt_A[1], dsnt_B[1], 0.3, lr_res)

# # plt.plot(dsnt_B[0], dsnt_B[1], 'k-', lw=3)
# # plt.plot(dsnt_A[0], fA120, '-')
# # plt.plot(dsnt_B[0], fB120, '-')

# modA_w, modA_f, modelA = get_model(T1_res, g1_res, 35, 'atlas')
# modB_w, modB_f, modelB = get_model(T2_res, g2_res, r2_res, 'tlusty')
# print(modelA, modelB)

# fitplot(dsnt_A[0]-0.2, f_A, [np.array(modA_w)], [np.array(modA_f)], [modelA], lr_res*100, lines_dic, select_linesA, savefig=True, legend_ax=7, balmer_min_y=0.4)
# fitplot(dsnt_B[0]-0.2, f_B, [np.array(modB_w)], [np.array(modB_f)], [modelB], lr_res*100, lines_dic, select_linesB, savefig=True, legend_ax=7)

'''
Disentangled spectrum rescaled
'''
# lr_res, he2h_res, T1_res, g1_res, rot1_res, T2_res, g2_res, rot2_res, chi2_res, \
#     chi2A, chi2B, chi2red, chi2rA, chi2rB = df_results_sort.loc[0].to_list()
#
# f_A, f_B = rescale_flux(dsnt_A[1], dsnt_B[1], 0.3, lr_res)
# print('\n', lr_res, he2h_res, T1_res, g1_res, rot1_res, T2_res, g2_res, rot2_res)
# modA_w, modA_f, modelA = get_model(T1_res, g1_res, rot1_res, 'atlas')
# modB_w, modB_f, modelB = get_model(T2_res, g2_res, rot2_res, 'tlusty')
# print(modelA)
#
# modslc_A_x, modslc_A_y = slicedata(modA_w, modA_f, user_dicA)
# modA_w_he, modA_f_he = He2H_ratio(modslc_A_x, modslc_A_y, 0.075, he2h_res, user_dicA, join=True)
#
# fitplot(dsnt_A[0]-0.2, f_A, [np.array(modA_w_he)], [np.array(modA_f_he)], [modelA], lr_res*100, lines_dic, select_linesA, figu='save', legend_ax=7, balmer_min_y=0.3)
# fitplot(dsnt_B[0]-0.2, f_B, [np.array(modB_w)], [np.array(modB_f)], [modelB], lr_res*100, lines_dic, select_linesB, figu='save', legend_ax=7)
