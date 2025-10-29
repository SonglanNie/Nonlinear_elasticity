import matplotlib.pyplot as plt
import numpy

lam_vec = [1,10,10**2,10**3,10**4,10**5]
log_lam_vec = numpy.log10(lam_vec)

N_eval_Newt = [44,32,28,26,24,24]
N_eval_BTLS = [110,66,52,48,48,48]
N_eval_LC = [100,80,74,72,72,72]

plt.plot(log_lam_vec, N_eval_Newt, label='Direct Newton')
plt.plot(log_lam_vec, N_eval_BTLS, label='Backtracking LS')
plt.plot(log_lam_vec, N_eval_LC, label = 'Load continuation')

plt.legend()
plt.show()

N_solve_Newt = [19,13,11,10,9,9]
N_solve_BTLS = [23,15,12,11,11,11]
N_solve_LC = [41,31,28,27,27,27]

plt.plot(log_lam_vec, N_solve_Newt, label='Direct Newton')
plt.plot(log_lam_vec, N_solve_BTLS, label='Backtracking LS')
plt.plot(log_lam_vec, N_solve_LC, label = 'Load continuation')

plt.legend()
plt.show()
