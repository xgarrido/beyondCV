#!/usr/bin/env python

# Global
import numpy as np

def simulation(setup):
    """
    Simulate CMB power spectrum given a set of cosmological parameters and noise level.
    """

    # Get experiment setup
    experiment = setup["experiment"]
    lmin, lmax = experiment["lmin"], experiment["lmax"]
    delta_l = experiment["delta_l"]

    from beyondCV import utils
    Dltt = utils.get_theory_cls(setup, lmax)
    ls = np.arange(lmin, lmax)
    Dl = Dltt[lmin:lmax]

    # Bin power spectrum
    lb, Db = utils.bin_spectrum(Dl, ls, lmin, lmax, delta_l)
    freq_Planck, DNl_array_Planck = utils.get_noise(experiment, "Planck")
    freq_Planck = list(freq_Planck)
    freq_Planck.append('all')

    ns = {}
    DNl = {}
    for freq in freq_Planck:
        key = "Planck_%s" % freq
        ns[key]=2.
        DNl[key]=DNl_array_Planck[freq]*ns[key]

    if setup.get("do_plot"):
        import matplotlib.pyplot as plt
        plt.semilogy()
        plt.plot(ls, Dl)
        for freq in freq_Planck:
            plt.plot(ls, DNl_array_Planck[freq], label="noise %s" % freq)
        plt.ylabel(r"$D_{\ell}$")
        plt.xlabel(r"$\ell$")
        plt.ylim(1, 5*10**4)
        plt.xlim(1, 3900)
        plt.legend()
        plt.show()

    covmat_PPPP = utils.cov("Planck_all","Planck_all","Planck_all","Planck_all", ns, ls, Dl, DNl, experiment["fsky"])
    lb, covmat_PPPP_b = utils.bin_variance(covmat_PPPP, ls, lmin, lmax, delta_l)
    epsilon_PPPP = np.sqrt(covmat_PPPP_b)*np.random.randn(len(covmat_PPPP_b))
    Db_obs = Db + epsilon_PPPP
    print("chi2(theo)/ndf = ", np.sum((Db_obs - Db)**2/covmat_PPPP_b)/len(lb))

    return Db_obs, covmat_PPPP_b

    # Nsims=10
    # for i in range(Nsims):
    #     epsilon_PPPP= np.sqrt(covmat_PPPP_b)*np.random.randn(len(covmat_PPPP_b))
    #     Db_obs= Db+ epsilon_PPPP

    #     if i==0 and setup.get("do_plot"):
    #         plt.figure()
    #         plt.subplot(2,1,1)
    #         plt.ylabel(r'$D_{\ell}$',fontsize=18)
    #         plt.errorbar(lb,Db)
    #         plt.errorbar(lb,Db_obs,np.sqrt(covmat_PPPP_b),fmt='.')
    #         plt.subplot(2,1,2)
    #         plt.ylabel(r'$\Delta D_{\ell}$',fontsize=18)
    #         plt.xlabel(r'$\ell$',fontsize=18)
    #         plt.errorbar(lb,Db_obs-Db,np.sqrt(covmat_PPPP_b),fmt='.')
    #         plt.show()

    #     chi2,dof = chisquare(Db_obs, Db, covmat_PPPP_b)
    #     print('chi2',chi2,'dof',dof)

def minimization(setup, Db, cov):
    """
    Minimize CMB power spectra over cosmo. parameters using `cobaya`.
    """

    # Get experiment setup
    experiment = setup["experiment"]
    lmin, lmax = experiment["lmin"], experiment["lmax"]
    delta_l = experiment["delta_l"]

    def chi2(_theory={"cl": {"tt": lmax}}):
        from beyondCV import utils
        ls = np.arange(lmin, lmax)
        Dl_theo = _theory.get_cl(ell_factor=True)["tt"][lmin:lmax]
        lb, Db_theo = utils.bin_spectrum(Dl_theo, ls, lmin, lmax, delta_l)
        chi2 = np.sum((Db - Db_theo)**2/cov)
        print("chi2/ndf = ", chi2/len(lb))
        return -chi2

    # Get cobaya setup
    info = setup["cobaya"]

    # Add likelihood function
    info["likelihood"] = {"chi2": chi2}

    from cobaya.run import run
    updated_info, products = run(info)

# Main function:
def main():
    import argparse
    parser = argparse.ArgumentParser(description = "A python to go beyond CMB cosmic variance")
    parser.add_argument("-y", "--yaml-file", help="Yaml file holding sim/minization setup",
                        default=None, required=True)
    parser.add_argument("-d", "--data-file", help="Data file holding simulated CMB spectrum and its covariance",
                        default=None, required=False)
    args = parser.parse_args()

    import yaml
    with open(args.yaml_file, "r") as stream:
        setup = yaml.load(stream)

    if args.data_file:
        Db, cov = np.loadtxt(args.data_file)
    else:
        Db, cov = simulation(setup)
    minimization(setup, Db, cov)

# script:
if __name__ == "__main__":
    main()

# end of beyondCV.py
