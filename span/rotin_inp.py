def rotin3_inp(grid, T, g, rot):
    '''
    grid: 'earlyB', 'lateB', 'O'
    '''
    import csv
    import os
    import subprocess
    import pandas as pd
    from glob import glob

    lowT_models_path = '/home/jaime/science/KUL/atm_models/ATLAS9/Jaime/'
    tlustyB_path = '/home/jaime/science/KUL/atm_models/TLUSTY/BLvispec_v2/'
    tlustyO_path = '/home/jaime/science/KUL/atm_models/TLUSTY/OLvispec_v10/'
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
            df.loc[0, 2] = '\'T'+str(T)+'g'+str(int(g*10))+'v'+str(vt)+'r'+str(rot)+'fw05\''
            if grid=='O':
                df.loc[0, 2] = '\'T'+str(int(T*10))+'g'+str(int(g*10))+'v'+str(vt)+'r'+str(int(rot))+'fw05\''
            df.loc[1, 0] = rot
            # print(df)
            df.to_csv(inpfile, sep='\t', index=False, header=False)
            os.chdir(path)
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
        else:
            print('model is not in models_list')