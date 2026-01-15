import openmc
import numpy as np

def get_flux_tally(sp_name):
    sp = sp = openmc.StatePoint(sp_name)
    flux_tally = sp.get_tally(scores=['flux'])
    flux_tally = flux_tally.get_slice(scores=['flux'])
    sp.close()
    return flux_tally

def get_runtime(sp_name):
    sp = openmc.StatePoint(sp_name)
    td = sp.runtime
    return td['simulation']

def get_pin_avg_error(tally):
    mean = tally.mean
    std_dev = tally.std_dev
    err = std_dev / (mean * np.sqrt(tally.num_realizations))
    return np.average(err)

tally = get_flux_tally('statepoint.300.h5')
tally_ww = get_flux_tally('statepoint.300_ww.h5')
error = get_pin_avg_error(tally)
error_ww = get_pin_avg_error(tally_ww)

rt = get_runtime('statepoint.300.h5')
rr_rt = 1.4943 + 1.5265
rt_ww = get_runtime('statepoint.300_ww.h5') + rr_rt

print(f'Mesh avg. error: {error}')
print(f'Mesh avg. error ww: {error_ww}')

print(f'sim runtime: {rt}')
print(f'sim runtime ww: {rt_ww}')

fom = 1/(rt * error)
fom_ww = 1/(rt_ww * error_ww)

print(f'fom: {fom}')
print(f'fom ww: {fom_ww}')




