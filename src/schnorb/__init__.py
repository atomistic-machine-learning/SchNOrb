from schnetpack import Properties


class SchNOrbProperties(Properties):
    ham_prop = 'hamiltonian'
    hamorth_prop = 'hamiltonian_orth'
    ov_trans_prop = 'ov_transform'
    ov_prop = 'overlap'
    en_prop = 'energy'
    f_prop = 'forces'
    psi_prop = 'psi'
    eps_prop = 'eps'


import schnorb.data
import schnorb.model
import schnorb.utils