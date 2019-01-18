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

    from beyondCV import utils
    Dltt = utils.get_theory_cls(setup, lmax)
    ls = np.arange(lmin, lmax)
    Dl = Dltt[lmin:lmax]

    # Get experiment noise
    freq_Planck, DNl_array_Planck = utils.get_noise(experiment, "Planck")
    freq_Planck = list(freq_Planck)
    freq_Planck.append("all")

    ns = {}
    DNl = {}
    for freq in freq_Planck:
        key = "Planck_%s" % freq
        ns[key] = 2.
        DNl[key] = DNl_array_Planck[freq]*ns[key]

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
    epsilon_PPPP = np.sqrt(covmat_PPPP)*np.random.randn(len(covmat_PPPP))
    Dl_obs = Dl + epsilon_PPPP
    print("chi2(theo)/ndf = ", np.sum((Dl_obs - Dl)**2/covmat_PPPP)/len(ls))

    return Dl_obs, covmat_PPPP

def simulation_full(setup):
    """
    Simulate CMB power spectrum given a set of cosmological parameters and noise level.
    """

    # Get experiment setup
    experiment = setup["experiment"]
    lmin, lmax = experiment["lmin"], experiment["lmax"]

    from beyondCV import utils
    simu = setup["simulation"]
    cosmo = simu["cosmo. parameters"]

    Dltt = utils.get_theory_cls(setup, lmax)
    ls = np.arange(lmin, lmax)
    Dl = Dltt[lmin:lmax]

    freq_Planck, DNl_array_Planck = utils.get_noise(experiment, "Planck")
    freq_Planck = list(freq_Planck)
    freq_Planck.append("all")

    freq_SO, DNl_array_SO = utils.get_noise(experiment, "SO")
    freq_SO = list(freq_SO)
    freq_SO.append("all")

    ns = {}
    DNl = {}
    for freq in freq_Planck:
        key = "Planck_%s" % freq
        ns[key] = 2.
        DNl[key] = DNl_array_Planck[freq]*ns[key]

    for freq in freq_SO:
        key = "SO_%s" % freq
        ns[key] = 10.
        DNl[key] = DNl_array_SO[freq]*ns[key]

    fsky = experiment["fsky"]
    covmat_SSSS = utils.cov("SO_all", "SO_all", "SO_all", "SO_all", ns, ls, Dl, DNl, fsky)
    covmat_SSSP = utils.cov("SO_all", "SO_all", "SO_all", "Planck_all", ns, ls, Dl, DNl, fsky)
    covmat_SSPP = utils.cov("SO_all", "SO_all", "Planck_all", "Planck_all", ns, ls, Dl, DNl, fsky)
    covmat_SPSP = utils.cov("SO_all", "Planck_all", "SO_all", "Planck_all", ns, ls, Dl, DNl, fsky)
    covmat_SPPP = utils.cov("SO_all", "Planck_all", "Planck_all", "Planck_all", ns, ls, Dl, DNl, fsky)
    covmat_PPPP = utils.cov("Planck_all", "Planck_all", "Planck_all", "Planck_all", ns, ls, Dl, DNl, fsky)

    covmat_master = np.zeros((3,3,len(Dl)))
    Dl_obs = np.zeros((3,len(Dl)))

    covmat_master[0,0,:] = covmat_SSSS
    covmat_master[0,1,:] = covmat_SSSP
    covmat_master[0,2,:] = covmat_SSPP
    covmat_master[1,0,:] = covmat_SSSP
    covmat_master[1,1,:] = covmat_SPSP
    covmat_master[1,2,:] = covmat_SPPP
    covmat_master[2,0,:] = covmat_SSPP
    covmat_master[2,1,:] = covmat_SPPP
    covmat_master[2,2,:] = covmat_PPPP

    for i in range(len(Dl)):
        mat = utils.svd_pow(covmat_master[:,:,i],1./2)
        Dl_obs[:,i] = Dl[i] + np.dot(mat, np.random.randn(3))

    Dl_obs_SxS, Dl_obs_SxP, Dl_obs_PxP = Dl_obs[0,:], Dl_obs[1,:], Dl_obs[2,:]

    if setup.get("do_plot"):
        import matplotlib.pyplot as plt
        plt.semilogy()
        plt.plot(ls, covmat_SSSS, label="SSSS")
        plt.plot(ls, covmat_SSSP, label="SSSP")
        plt.plot(ls, covmat_SSPP, label="SSPP")
        plt.plot(ls, covmat_SPSP, label="SPSP")
        plt.plot(ls, covmat_SPPP, label="SPPP")
        plt.plot(ls, covmat_PPPP, label="PPPP")
        plt.legend()
        plt.show()

        grid = plt.GridSpec(4, 1, hspace=0, wspace=0)
        main = plt.subplot(grid[0:3], xticklabels=[])
        main.semilogy()
        main.plot(ls, Dl_obs_SxS, label="SOxSO",alpha=0.5)
        main.plot(ls, Dl_obs_SxP, label="SOxP",alpha=0.5)
        main.plot(ls, Dl_obs_PxP, label="PxP",alpha=0.5)
        main.legend()
        dev = plt.subplot(grid[3], ylim=[-5,5])
        dev.plot(ls, 100*(1 - Dl_obs_SxS/Dl_obs_SxS), alpha=0.5)
        dev.plot(ls, 100*(1 - Dl_obs_SxP/Dl_obs_SxS), alpha=0.5)
        dev.plot(ls, 100*(1 - Dl_obs_PxP/Dl_obs_SxS), alpha=0.5)
        dev.set_ylabel(r"$\Delta D_\ell\;[\sigma]$")
        dev.set_xlabel(r"$\ell$")
        plt.show()

    print("SOxSO chi2(theo)/ndf = ", np.sum((Dl_obs_SxS - Dl)**2/covmat_SSSS)/len(ls))
    print("SOxP chi2(theo)/ndf = ", np.sum((Dl_obs_SxP - Dl)**2/covmat_SPSP)/len(ls))
    print("PxP chi2(theo)/ndf = ", np.sum((Dl_obs_PxP - Dl)**2/covmat_PPPP)/len(ls))

    return Dl_obs_SxS, Dl_obs_SxP, Dl_obs_PxP, covmat_SSSS, covmat_SPSP, covmat_PPPP

def minimization(setup, Dl, cov):
    """
    Minimize CMB power spectra over cosmo. parameters using `cobaya`.
    """

    # Get experiment setup
    experiment = setup["experiment"]
    lmin, lmax = experiment["lmin"], experiment["lmax"]

    def chi2(_theory={"cl": {"tt": lmax}}):
        from beyondCV import utils
        Dl_theo = _theory.get_cl(ell_factor=True)["tt"][lmin:lmax]
        chi2 = np.sum((Dl - Dl_theo)**2/cov)
        print("chi2/ndf = ", chi2/len(Dl_theo))
        return -chi2

    # Get cobaya setup
    info = setup["cobaya"]

    # Add likelihood function
    info["likelihood"] = {"chi2": chi2}

    from cobaya.run import run
    return run(info)

# Main function:
def main():
    import argparse
    parser = argparse.ArgumentParser(description = "A python to go beyond CMB cosmic variance")
    parser.add_argument("-y", "--yaml-file", help="Yaml file holding sim/minization setup",
                        default=None, required=True)
    parser.add_argument("--survey", help="Set seed of random generator",
                        choices = ["SO", "SOxP", "P"], required=True)
    parser.add_argument("--data-file", help="Data file holding simulated CMB spectrum and its covariance",
                        default=None, required=False)
    parser.add_argument("--seed-simulation", help="Set seed for the simulation random generator",
                        default=None, required=False)
    parser.add_argument("--seed-minimization", help="Set seed for the minimization random generator",
                        default=None, required=False)
    args = parser.parse_args()

    if args.seed_simulation:
        print("WARNING: Seed for simulation set to {} value".format(args.seed_simulation))
        np.random.seed(int(args.seed_simulation))

    import yaml
    with open(args.yaml_file, "r") as stream:
        setup = yaml.load(stream)

    # Do the simulation
    if args.data_file:
        Dl, cov = np.loadtxt(args.data_file)
    else:
        sims = simulation_full(setup)

    survey = args.survey
    print("INFO: Doing minimization for '{}' survey".format(survey))
    if survey == "SO":
        Dl, cov = sims[0], sims[3]
    elif survey == "SOxP":
        Dl, cov = sims[1], sims[4]
    elif survey == "P":
        Dl, cov = sims[2], sims[5]

    # Do the minimization
    if args.seed_minimization:
        print("WARNING: Seed for minimization set to {} value".format(args.seed_minimization))
        np.random.seed(int(args.seed_minimization))
    updated_info, results = minimization(setup, Dl, cov)

    # Store MINUIT results (remove the 'minuit' object that can't be serialized)
    del results["OptimizeResult"]["minuit"]
    import pickle
    output_dir = setup.get("cobaya").get("output")
    if output_dir:
        pickle.dump(results, open(output_dir + "_results.pkl", "wb"))

# script:
if __name__ == "__main__":
    main()

# end of beyondCV.py
