import numpy as np
from scipy.interpolate import InterpolatedUnivariateSpline as ius
from dark_emulator import model_hod
    
class demu_pk_gx(model_hod.darkemu_x_hod):
    def __init__(self):
        super().__init__()

    def get_pk_gg(self, k, z):

        self._check_update_redshift(z)

        self._compute_p_1hcs(z)
        self._compute_p_1hss(z)
        self._compute_p_2hcc(z)
        self._compute_p_2hcs(z)
        self._compute_p_2hss(z)

        p_tot_1h = 2.*self.p_1hcs + self.p_1hss
        p_tot_2h = self.p_2hcc + 2.*self.p_2hcs + self.p_2hss
        pk_gg = ius( self.fftlog_1h.k, p_tot_1h )(k) + ius( self.fftlog_2h.k, p_tot_2h )(k)
        return pk_gg

    def get_pk_gm(self, k, z):

        self._check_update_redshift(z)

        self._compute_p_cen(z)
        self._compute_p_cen_off(z)
        self._compute_p_sat(z)

        p_tot = self.p_cen + self.p_cen_off + self.p_sat
        pk_gm = ius( self.fftlog_1h.k, p_tot )(k)
        return pk_gm