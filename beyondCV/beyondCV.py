#!/usr/bin/env python

# Simulation
def simulation():
    import numpy as np
    import matplotlib.pyplot as plt
    import fisher_dict
    import sys
    import utils
    import os

    def chisquare(Db_obs, Db_th, var):
        return np.sum((Db_obs-Db_th)**2/var), len(Db_obs)

    p = fisher_dict.flipperDict()
    p.read_from_file(sys.argv[1])
    totCL=utils.get_theory_cls(p)
    Dl=totCL[:,0]
    ls=np.arange(2,p['lmax'])
    Dl=Dl[2:len(ls)+2]

    lb,Db=utils.bin_spectrum(p,ls,Dl)
    freq_Planck,DNl_array_Planck=utils.get_noise(p,'Planck')
    freq_Planck=list(freq_Planck)
    freq_Planck.append('all')

    ns={}
    DNl={}

    plt.semilogy()
    plt.plot(ls,Dl)
    for freq in freq_Planck:
        ns['Planck_%s'%freq]=2.
        DNl['Planck_%s'%freq]=DNl_array_Planck[freq]*ns['Planck_%s'%freq]
        plt.plot(ls,DNl_array_Planck[freq],label='noise %s'%freq)
    plt.ylabel(r'$D_{\ell}$',fontsize=18)
    plt.xlabel(r'$\ell$',fontsize=18)
    plt.ylim(1,5*10**4)
    plt.xlim(1,3900)
    plt.legend()
    plt.show()

    covmat_PPPP=utils.cov('Planck_all','Planck_all','Planck_all','Planck_all',ns,ls,Dl,DNl,p)
    lb,covmat_PPPP_b=utils.bin_variance(p,ls,covmat_PPPP)

    Nsims=10
    for i in range(Nsims):
        epsilon_PPPP= np.sqrt(covmat_PPPP_b)*np.random.randn(len(covmat_PPPP_b))
        Db_obs= Db+ epsilon_PPPP

        if i==0:
            plt.figure()
            plt.subplot(2,1,1)
            plt.ylabel(r'$D_{\ell}$',fontsize=18)
            plt.xlabel(r'$\ell$',fontsize=18)
            plt.errorbar(lb,Db)
            plt.errorbar(lb,Db_obs,np.sqrt(covmat_PPPP_b),fmt='.')
            plt.subplot(2,1,2)
            plt.ylabel(r'$\Delta D_{\ell}$',fontsize=18)
            plt.xlabel(r'$\ell$',fontsize=18)
            plt.errorbar(lb,Db_obs-Db,np.sqrt(covmat_PPPP_b),fmt='.')
            plt.show()

        chi2,dof= chisquare(Db_obs, Db, covmat_PPPP_b)
        print('chi2',chi2,'dof',dof)

# Main function:
def main():
    simulation()

# script:
if __name__ == "__main__":
    main()

# end of qsurvey.py
