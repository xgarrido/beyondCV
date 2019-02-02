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

    fsky = experiment["fsky"]
    survey = experiment["survey"]
    if survey == "P":
        covmat_PPPP = utils.cov("Planck_all","Planck_all","Planck_all","Planck_all", ns, ls, Dl, DNl, fsky)
        epsilon_PPPP = np.sqrt(covmat_PPPP)*np.random.randn(len(covmat_PPPP))
        Dl_obs = Dl + epsilon_PPPP
        print("chi2(theo)/ndf = ", np.sum((Dl_obs - Dl)**2/covmat_PPPP)/len(ls))
        return Dl_obs, covmat_PPPP

    else:
        for freq in freq_SO:
            key = "SO_%s" % freq
            ns[key] = 10.
            DNl[key] = DNl_array_SO[freq]*ns[key]

        covmat_SSSS = utils.cov("SO_all", "SO_all", "SO_all", "SO_all", ns, ls, Dl, DNl, fsky)
        covmat_SSSP = utils.cov("SO_all", "SO_all", "SO_all", "Planck_all", ns, ls, Dl, DNl, fsky)
        covmat_SSPP = utils.cov("SO_all", "SO_all", "Planck_all", "Planck_all", ns, ls, Dl, DNl, fsky)
        covmat_SPSP = utils.cov("SO_all", "Planck_all", "SO_all", "Planck_all", ns, ls, Dl, DNl, fsky)
        covmat_SPPP = utils.cov("SO_all", "Planck_all", "Planck_all", "Planck_all", ns, ls, Dl, DNl, fsky)
        covmat_PPPP = utils.cov("Planck_all", "Planck_all", "Planck_all", "Planck_all", ns, ls, Dl, DNl, fsky)

        if survey in ["SOxSO", "SOxP", "PxP"]:
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

            if survey == "SOxSO":
                Dl_obs, covmat = Dl_obs_SxS, covmat_SSSS
            elif survey == "SOxP":
                Dl_obs, covmat = Dl_obs_SxP, covmat_SPSP
            elif survey == "PxP":
                Cl_obs, covmat = Dl_obs_PxP, covmat_PPPP
            print("{} chi2(theo)/ndf = {}".format(survey, np.sum((Dl_obs - Dl)**2/covmat)/len(ls)))
            return Dl_obs, covmat
        elif survey in ["SOxSO-PxP", "SOxP-PxP", "SOxP-SOxSO", "SOxSO+PxP-2SOxP"] :
            if survey == "SOxSO-PxP":
                covmat = C1 = covmat_SSSS + covmat_PPPP - 2*covmat_SSPP
            elif survey == "SOxP-PxP":
                covmat = C2 = covmat_SPSP + covmat_PPPP - 2*covmat_SPPP
            elif survey == "SOxP-SOxSO":
                covmat = C3 = covmat_SPSP + covmat_SSSS - 2*covmat_SSSP
            elif survey == "SOxSO+PxP-2SOxP":
                covmat = C4 = covmat_SSSS + covmat_PPPP + 2*covmat_SSPP - 4*(covmat_SSSP+covmat_SPPP) + 4*covmat_SPSP

            Delta_Dl_obs = np.sqrt(covmat)*np.random.randn(len(ls))
            print("{} chi2(theo)/ndf = {}".format(survey, np.sum(Delta_Dl_obs**2/covmat)/len(ls)))
            return Delta_Dl_obs, covmat
        else:
            raise ValueError("Unknown survey '{}'!".format(survey))

def sampling(setup, Dl, cov):
    """
    Sample CMB power spectra over cosmo. parameters using `cobaya` using either
    minimization algorithms or MCMC methods.
    """

    # Get experiment setup
    experiment = setup["experiment"]
    lmin, lmax = experiment["lmin"], experiment["lmax"]

    # Chi2 for CMB spectra sampling
    def chi2(_theory={"cl": {"tt": lmax}}):
        Dl_theo = _theory.get_cl(ell_factor=True)["tt"][lmin:lmax]
        chi2 = np.sum((Dl - Dl_theo)**2/cov)
        return -chi2

    # Chi2 for CMB spectra residuals sampling
    from beyondCV import utils
    Dl_Planck = utils.get_theory_cls(setup, lmax)[lmin:lmax]
    def chi2_residuals(_theory={"cl": {"tt": lmax}}):
        Dl_theo = _theory.get_cl(ell_factor=True)["tt"][lmin:lmax]
        Delta_Dl_obs, Delta_Dl_theo = Dl, Dl_theo - Dl_Planck
        chi2 = np.sum((Delta_Dl_obs - Delta_Dl_theo)**2/cov)
        return -chi2

    # Get cobaya setup
    info = setup["cobaya"]

    # Add likelihood function
    survey = setup.get("experiment").get("survey")
    if survey in ["P", "SOxSO", "SOxP", "PxP"]:
        info["likelihood"] = {"chi2": chi2}
    else:
        info["likelihood"] = {"chi2": chi2_residuals}

    from cobaya.run import run
    return run(info)

def store_results(setup, results):
    # Store configuration and MINUIT results
    # Remove function pointer and cobaya results (issue with thread)
    del setup["cobaya"]["likelihood"]
    if results.get("OptimizeResult"):
        del results["OptimizeResult"]["minuit"]
    if results.get("maximum"):
        del results["maximum"]
    output_dir = setup.get("cobaya").get("output")
    if output_dir:
        import pickle
        pickle.dump({"setup": setup, "results": results},
                    open(output_dir + "_results.pkl", "wb"))


# Main function:
def main():
    import argparse
    parser = argparse.ArgumentParser(description = "A python to go beyond CMB cosmic variance")
    parser.add_argument("-y", "--yaml-file", help="Yaml file holding sim/minization setup",
                        default=None, required=True)
    parser.add_argument("--survey", help="Set seed of random generator",
                        choices = ["P", "SOxSO", "SOxP", "PxP",
                                   "SOxSO-PxP", "SOxP-PxP", "SOxP-SOxSO", "SOxSO+PxP-2SOxP"],
                        default=None, required=True)
    parser.add_argument("--seed-simulation", help="Set seed for the simulation random generator",
                        default=None, required=False)
    parser.add_argument("--seed-sampling", help="Set seed for the sampling random generator",
                        default=None, required=False)
    parser.add_argument("--do-mcmc", help="Do MCMC after minimization",
                        default=False, required=False, action="store_true")
    parser.add_argument("--use-covmat", help="Use covariance matrix from minimization",
                        default=False, required=False, action="store_true")
    parser.add_argument("--output-base-dir", help="Set the output base dir where to store results",
                        default=".", required=False)
    args = parser.parse_args()

    import yaml
    with open(args.yaml_file, "r") as stream:
        setup = yaml.load(stream)

    # Check survey
    survey = args.survey
    setup["experiment"]["survey"] = survey

    # Do the simulation
    print("INFO: Doing simulation for '{}' survey".format(survey))
    if args.seed_simulation:
        print("WARNING: Seed for simulation set to {} value".format(args.seed_simulation))
        setup["seed_simulation"] = args.seed_simulation
        np.random.seed(int(args.seed_simulation))
    Dl, cov = simulation(setup)

    # Do the minimization
    if args.seed_sampling:
        print("WARNING: Seed for sampling set to {} value".format(args.seed_sampling))
        setup["seed_sampling"] = args.seed_sampling
        np.random.seed(int(args.seed_sampling))
    setup["cobaya"]["output"] = args.output_base_dir + "/minimize"
    updated_info, results = sampling(setup, Dl, cov)
    store_results(setup, results)

    # Do the MCMC
    if args.do_mcmc:
        # Update cobaya setup
        params = setup.get("cobaya").get("params")
        if args.use_covmat:
            covmat = results.get("OptimizeResult").get("hess_inv")
            covmat_params = [k for k, v in params.items() if isinstance(v, dict) and "prior" in v.keys()]
            setup["cobaya"]["sampler"] = {"mcmc": {"covmat": covmat, "covmat_params": covmat_params}}
        else:
            for k, v in params.items():
                if isinstance(v, dict) and "prior" in v.keys():
                    proposal = (v.get("prior").get("max") - v.get("prior").get("min"))/2
                    params[k]["proposal"] = proposal
                    setup["cobaya"]["sampler"] = {"mcmc": None}

        setup["cobaya"]["output"] = args.output_base_dir + "/mcmc"
        updated_info, results = sampling(setup, Dl, cov)
        store_results(setup, results)

# script:
if __name__ == "__main__":
    main()

# end of beyondCV.py
