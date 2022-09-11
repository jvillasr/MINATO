from compute_phases import compute_phases

location = 'Roque de los Muchachos, La Palma'
# target = '[NH52] 63'
# time = '2022-09-07 23:06'
# last_time = '2022-09-12 01:00'
# ra, dec = '20 18 00.62', '36 39 03.06'
# period = 6.84
# compute_phases(time, location, period, 32, 0.1, last_time, ra, dec, target, twilight='nautical')


target = 'HD 165921' # Sergio's sample
time = '2022-09-03 22:10'
last_time = '2022-09-12 05:00'
ra, dec = '18 09 17.690', '-23:59:18.23'
period = 1.74
compute_phases(time, location, period, 16, 0.1, last_time, ra, dec, target, twilight='nautical')

# target = 'V420 Aur' # Soetkin's BeXRB
# time = '2022-09-11 03:30'
# last_time = '2022-09-13 05:00'
# ra, dec = '05 22 35.231', '37:40:33.640'
# period = 0.8
# compute_phases(time, location, period, 16, 0.1, last_time, ra, dec, target, twilight='nautical')