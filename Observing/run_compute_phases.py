from compute_phases import compute_phases

location = 'Roque de los Muchachos, La Palma'
target = '[NH52] 63'

time = '2022-09-07 23:06'
last_time = '2022-09-12 01:00'
ra, dec = '20 18 00.62', '36 39 03.06'

period = 6.84
compute_phases(time, location, period, 32, 0.1, last_time, ra, dec, target, twilight='nautical')