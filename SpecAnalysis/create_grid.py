import timeit
from itertools import product
import numpy as np
import pandas as pd
from datetime import timedelta, date
current_date = str(date.today())

def create_grid(*params_lists):

    grid = list(product(*params_lists))
    return grid

teffA1 = np.linspace(10, 15, 6, dtype=int)
teffA2 = np.linspace(16, 18, 3, dtype=int)
loggA1 = [18, 20, 22, 24, 26, 28, 30]
loggA2 = [20, 22.5, 25, 27.5, 30]
rotA = np.linspace(0, 70, 8, dtype=int)
teffB = [16, 18, 20, 22, 24, 26, 28, 30, 32.5, 35, 37.5]
loggB = [20, 22.5, 25, 27.5, 30, 32.5, 35, 37.5, 40, 42.5, 45]
rotB = list(range(0, 600, 50))
lrat = np.linspace(0.1, 0.5, 9, dtype=float)
he2h = np.linspace(0.05, 0.12, 8, dtype=float)

pars_list1 = [lrat, he2h, teffA1, loggA1, rotA, teffB, loggB, rotB]
pars_list2 = [lrat, he2h, teffA2, loggA2, rotA, teffB, loggB, rotB]


grid1 = create_grid(*pars_list1)
grid2 = create_grid(*pars_list2)
grid = grid1 + grid2
print(grid[:2])
print(grid[-2:])

df = pd.DataFrame(grid, columns=['lrat','He2H','TeffA','loggA','rotA','TeffB','loggB','rotB'])

df.to_csv('grid.csv', index=False)

# df.to_feather('grid.feather')



