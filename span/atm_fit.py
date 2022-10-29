import time
import itertools
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.interpolate as inter
from glob import glob
from datetime import timedelta, date
current_date = str(date.today())

class atm_fitting:
    def __init__(self, grid, spectrumA, spectrumB):
        self.grid = grid
        self.spectrumA = spectrumA
        self.spectrumB = spectrumB
        # self.lrat = lrat
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
    def user_dic(self, lines):
        self.lines = lines
        usr_dic = { line: self.lines_dic[line] for line in self.lines }
        return usr_dic
    def compute_chi2(self, dic_lines_A, dic_lines_B):
        self.dic_lines_A = dic_lines_A
        self.dic_lines_B = dic_lines_B
        try:
            df = pd.read_feather(self.grid)
        except:
            df = pd.read_csv(self.grid)
        nparams = len(df.columns)
        # print(df[:5])

        spectra = Spectra(self.grid, self.spectrumA, self.spectrumB)
        wavA, wavB = spectra.get_wave()
        usr_dicA = self.user_dic(dic_lines_A)
        usr_dicB = self.user_dic(dic_lines_B)

        # result_dic = dict()
        result_dic = {'chi2_tot': [], 'chi2A': [], 'chi2B': [], 'chi2r_tot': [], 'chi2redA': [], 'chi2redB': []}
        # #Iterate over all values in pivot table
        # t0 = time.time()
        # rownum = 50000
        # v = df[:rownum].values
        # last_lr = None
        # for row in range(df[:rownum].shape[0]):
            
        #     if df['lrat'].loc[row] != last_lr:
        #         print(row, df['lrat'].loc[row])
        #         fluA, fluB = spectra.rescale_flux(df['lrat'].loc[row])
        #     for column in range(df[:rownum].shape[1]):
        #         test = v[row, column]
        #     # print(v[row])
        #     last_lr = df['lrat'].loc[row]
        # t1 = time.time()
        # print('Iteration over values takes: ' + str(t1-t0))

        #Iterate over all values in pivot table
        t0 = time.time()
        # rownum = 500000
        # rownum = 1460
        v = df.values
        last_lr, last_he2h = None, None
        # print(last_lr)
        for row in range(df.shape[0]):
            # print(v[row])
            # print(v[row, 0])
            if v[row, 0] != last_lr:
                # print(row, *v[row, 2:5])
                fluA, fluB = spectra.rescale_flux(v[row, 0])
                dst_A_w_slc, dst_A_f_slc = self.slicedata(wavA, fluA, usr_dicA)
                dst_B_w_slc, dst_B_f_slc = self.slicedata(wavB, fluB, usr_dicB)

            try:
                if v[row, 2] < 16:
                    modA_w, modA_f, modelA = self.get_model(*v[row, 2:5], source='atlas')
                else:
                    modA_w, modA_f, modelA = self.get_model(*v[row, 2:5], source='tlusty')
                modB_w, modB_f, modelB = self.get_model(*v[row, 5:], source='tlusty')


                if modA_f and modB_f:
                    # splA = inter.UnivariateSpline(modA_w, modA_f)
                    # splA.set_smoothing_factor(0.)
                    # modA_f_interp = [splA(x) for x in dst_A_w_slc]
                    modA_f_interp = [np.interp(x, modA_w, modA_f) for x in wavA]
                    #
                    # splB = inter.UnivariateSpline(modB_w, modB_f)
                    # splB.set_smoothing_factor(0.)
                    # modB_f_interp = [splB(x) for x in dst_B_w_slc]
                    modB_f_interp = [np.interp(x, modB_w, modB_f) for x in wavB]

                    if v[row, 1] != last_he2h:
                        # print('last He/H: ', last_he2h)
                        modA_f_interp = self.He2H_ratio(wavA, modA_f_interp, 0.075, v[row, 1], usr_dicA)
                        # modB_f_interp = He2H_ratio(dst_B_x, modB_f_interp, 0.1, he2h, dicB)



                    chi2A, chi2B, ndataA, ndataB = 0, 0, 0, 0
                    for i,line in enumerate(usr_dicB):
                        if line == 4102:
                            dst_B_x_crop, dst_B_y_crop    = self.crop_data(dst_B_w_slc[i], dst_B_f_slc[i], 4098, 4105)
                            spl_wavB_crop, modB_f_interp_crop = self.crop_data(dst_B_w_slc[i], modB_f_interp[i], 4098, 4105)
                        elif line == 4340:
                            dst_B_x_crop, dst_B_y_crop    = self.crop_data(dst_B_w_slc[i], dst_B_f_slc[i], 4334, 4347)
                            spl_wavB_crop, modB_f_interp_crop = self.crop_data(dst_B_w_slc[i], modB_f_interp[i], 4334, 4347)
                        else:
                            dst_B_x_crop, dst_B_y_crop    = dst_B_w_slc[i], dst_B_f_slc[i]
                            spl_wavB_crop, modB_f_interp_crop = dst_B_w_slc[i], modB_f_interp[i]
                        ndataB += len(modB_f_interp_crop)
                        chi2B += self.chi2(dst_B_y_crop, modB_f_interp_crop)
                    # print('number of data point of spectrum B', ndataB)
                    for i,line in enumerate(usr_dicA):
                        ndataA += len(modA_f_interp[i])
                        chi2A += self.chi2(dst_A_f_slc[i], modA_f_interp[i])
                        # print('chi2A =', chi2A, 'chi2B =', chi2B)
                        # chisqr = chi2A + chi2B
                        # print(line, chisqr)
                        # print(line, chisqr/len(dst_B_y_crop), len(dst_B_y_crop))
                    chi2_tot = chi2A + chi2B
                    result_dic['chi2_tot'].append(chi2_tot)
                    result_dic['chi2A'].append(chi2A)
                    result_dic['chi2B'].append(chi2B)
                    # print('number of data point of spectrum A', ndataA)
                    ndata = ndataA + ndataB
                    chi2redA = chi2A/(ndataA-nparams)
                    chi2redB = chi2B/(ndataB-nparams)
                    chi2r_tot = chi2redA + chi2redB
                    result_dic['chi2r_tot'].append(chi2r_tot)
                    result_dic['chi2redA'].append(chi2redA)
                    result_dic['chi2redB'].append(chi2redB)
                    # print('total number of data point', ndata)
                    # print('\n   chi2 =', chi2_tot)
                    # gridB.append([modelB, lr, T2, g2, rv, chi2_tot])
                    # return chi2_tot, chi2A, chi2B, chi2r_tot, chi2redA, chi2redB
            except TypeError:
                # pass
                # raise TypeError()
                result_dic['chi2_tot'].append(None)
                result_dic['chi2A'].append(None)
                result_dic['chi2B'].append(None)
                result_dic['chi2r_tot'].append(None)
                result_dic['chi2redA'].append(None)
                result_dic['chi2redB'].append(None)
            if v[row, 0] != last_lr and last_lr != None:
                t1 = time.time()
                print('\n Light ratio = ' + str(v[row, 0]) + ' completed in : ' + str(timedelta(seconds=t1-t0)) + ' [s] \n')
                print('\n')            
            if v[row, 1] != last_he2h and last_lr != None:
                t2 = time.time()
                print('   He/H ratio = ' + str(v[row, 1]) + ' completed in : ' + str(timedelta(seconds=t2-t0)) + ' [s] for l_rat = ' + str(v[row, 0]))
            last_lr = v[row, 0]
            last_he2h = v[row, 1]
        tf = time.time()
        print('\nComputation completed in: ' + str(timedelta(seconds=tf-t0)) + ' [s] \n')
        # print(result_dic)
        tf1 = time.time()
        output = pd.DataFrame.from_dict(result_dic)
        output = pd.concat([df, output], ignore_index=False, axis=1)
        print(output)
        tf2 = time.time()
        print('dataframe and created in: ' + str(timedelta(seconds=tf2-tf1)) + ' [s] \n')
        return output


    def rescale_flux(self, lrat):
        self.lrat = lrat
        ratio0 = 0.3
        ratio1 = lrat
        fluxA, fluxB = self.get_flux()
        flux_new_A = (fluxA -1)*((1-ratio0)/(1-ratio1)) + 1
        flux_new_B = (fluxB -1)*(ratio0/ratio1) + 1
        return flux_new_A, flux_new_B
    def slicedata(self, x_data, y_data, dictionary):
        self.x_data = x_data
        self.y_data = y_data
        self.dictionary = dictionary
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
    def get_model(self, *pars, source='tlusty'):
        '''
        source : Source of the models. Options are 'tlusty' and 'atlas'.
        '''
        T, g, rot = pars
        lowT_models_path = '~/Science/github/jvillasr/MINATO/span/models/ATLAS9/'
        tlustyB_path =     '~/Science/github/jvillasr/MINATO/span/models/TLUSTY/BLMC_v2/'
        tlustyO_path =     '~/Science/github/jvillasr/MINATO/span/models/TLUSTY/OLMC_v10/'

        lowT_models_list = sorted(glob(lowT_models_path+'*fw05'))
        lowT_models_list = [x.replace(lowT_models_path, '') for x in lowT_models_list]

        tlustyB_list = sorted(glob(tlustyB_path+'*fw05'))
        tlustyB_list = [x.replace(tlustyB_path, '') for x in tlustyB_list]

        tlustyO_list = sorted(glob(tlustyO_path+'*fw05'))
        tlustyO_list = [x.replace(tlustyO_path, '') for x in tlustyO_list]
        tlustyOB_list = tlustyB_list + tlustyO_list
        
        model = 'T'+str(int(T))+'g'+str(int(g*10))+'v2r'+str(int(rot))+'fw05'

        if source=='tlusty':
            try:
                if T>30:
                    model = 'T'+str(int(T*10))+'g'+str(int(g*10))+'v10r'+str(int(rot))+'fw05'
                    df = pd.read_csv(tlustyO_path+model,header=None, sep='\s+')

                else:
                    df = pd.read_csv(tlustyB_path+model,header=None, sep='\s+')
                return df[0].array, df[1].array, model
            except FileNotFoundError:
                # print('WARNING: No model named '+model+' was found')
                # raise ValueError('   WARNING: No model available for '+model)
                pass

        elif source=='atlas':
            model = 'T'+str(int(T))+'g'+str(int(g))+'v2r'+str(int(rot))+'fw05'
            try:
                df = pd.read_csv(lowT_models_path+model,header=None, sep='\s+')
                return df[0].array, df[1].array, model
            except FileNotFoundError:
                # print('WARNING: No model named '+model+' was found')
                # raise ValueError('   WARNING: No model available for '+model)
                pass
    def He2H_ratio(self, wave, flux, ratio0, ratio1, dictionary, join=False):
        new_spetrum = []
        '''
        Modifies the He/H ratio
        It works on sliced data. Returns sliced data unless 'join' is True.
        '''
        self.wave = wave
        self.flux = flux
        self.ratio0 = ratio0
        self.ratio1 = ratio1
        self.dictionary = dictionary
        self.join = join
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
    def chi2(self, obs, exp):
        self.obs = obs
        self.exp = exp
        return np.sum(((obs-exp)**2)/exp)
    def crop_data(self, x_data, y_data, range1, range2):
        self.x_data = x_data
        self.y_data = y_data
        self.range1 = range1
        self.range2 = range2
        cond = (x_data < range1) | (x_data > range2)
        return x_data[cond], y_data[cond]


class Spectra(atm_fitting):
    # def __init__(atm_fitting):

    def read_spec(self):
        dsnt_A = pd.read_csv(self.spectrumA, header=None, sep='\s+')        
        dsnt_B = pd.read_csv(self.spectrumB, header=None, sep='\s+')
        return dsnt_A, dsnt_B
    def get_wave(self):
        specA, specB = self.read_spec() 
        waveA = specA[0]-0.2
        waveB = specB[0]-0.2
        return waveA, waveB
    def get_flux(self):
        specA, specB = self.read_spec() 
        fluxA = specA[1]
        fluxB = specB[1]
        return fluxA, fluxB


# dsnt_A = '/Users/jaime/Science/KUL_postdoc/BBC/291/tomer/ADIS_lguess_K1K2=0.3_94.0_15.0_renorm.txt'
# dsnt_B = '/Users/jaime/Science/KUL_postdoc/BBC/291/tomer/BDIS_lguess_K1K2=0.3_94.0_15.0.txt'
# grid = '~/Science/github/jvillasr/MINATO/SpecAnalysis/data/grid.feather'

# select_linesA = [4026, 4102, 4121, 4144, 4267, 4340, 4388, 4471, 4553]
# select_linesB = [4026, 4102, 4121, 4144, 4267, 4340, 4388, 4553]

# res_df = atm_fitting(grid, dsnt_A, dsnt_B).compute_chi2(select_linesA, select_linesB)
# # print(res_dic)
# res_df.to_feather('./data/minchi2_result_'+current_date+'.feather')