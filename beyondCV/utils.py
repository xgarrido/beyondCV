import numpy as np
from beyondCV import V3calc as V3

def get_noise(p, exp):
    l = np.arange(p['lmin'], p['lmax'])
    if p.get("use_external_file_%s"%exp) is None:
        sigma = np.array(p['noise_%s'%exp])
        beam_FWHM = np.array(p['beam_%s'%exp])
        freq = np.array(p['freq_%s'%exp])
        sigma_rad = np.deg2rad(sigma)/60
        beam_FWHM_rad = np.deg2rad(beam_FWHM)/60
        beam = beam_FWHM_rad/np.sqrt(8*np.log(2))
        DNl_array = {}
        DNl_all = 0
        count=0
        for f in freq:
            DNl_array[f] = sigma_rad[count]**2*np.exp(l*(l+1)*beam[count]**2)*l*(l+1)/(2*np.pi)
            DNl_all += 1/DNl_array[f]
            count += 1
        DNl_array['all'] = 1/DNl_all
    else:
        print('use external file')
        freq_all=np.array(p['freq_all_%s'%exp])
        ell,N_ell_T_LA,N_ell_P_LA,Map_white_noise_levels=V3.so_V3_LA_noise(2,p['fsky'],p['lmin'],p['lmax'],delta_ell=1,beam_corrected=True)
        freq=np.array(p['freq_%s'%exp])
        DNl_array={}
        DNl_all=0
        count=0
        for f1 in freq_all:
            for f2 in freq:
                if f1==f2:
                    DNl_array[f1]=N_ell_T_LA[count]*l*(l+1)/(2*np.pi)
                    DNl_all+=1/DNl_array[f1]
            count+=1
        DNl_array['all']=1/DNl_all
    return freq, DNl_array

# def get_theory_cls(p, lmax):
#     import camb
#     pars = camb.CAMBparams()
#     camb.lensing.ALens.value=p['Alens']
#     if p.get('H0'):
#         pars.set_cosmology(H0=p['H0'], ombh2=p['ombh2'], omch2=p['omch2'], mnu=p['mnu'], omk=p['omk'], tau=p['tau'])
#     else:
#         pars.set_cosmology(H0=None,cosmomc_theta=p['theta100']/100., ombh2=p['ombh2'], omch2=p['omch2'], mnu=p['mnu'], omk=p['omk'], tau=p['tau'])

#     pars.InitPower.set_params(As=np.exp(p['ln_ten_to_ten_As'])/1e10,ns=p['ns'], r=p['r'])
#     print('theta',p['theta100']/100.)
#     print('ombh2',p['ombh2'])
#     print('omch2',p['omch2'])
#     print('As',np.exp(p['ln_ten_to_ten_As'])/1e10)
#     print('ns',p['ns'])

#     pars.set_for_lmax(lmax, lens_potential_accuracy=0)
#     results = camb.get_results(pars)
#     powers = results.get_cmb_power_spectra(pars, CMB_unit='muK')
#     totCL=powers['total']
#     return(totCL)

def get_theory_cls(setup, lmax):
    # Get simulation parameters
    simu = setup["simulation"]
    cosmo = simu["cosmo. parameters"]
    # Get cobaya setup
    from copy import deepcopy
    info = deepcopy(setup["cobaya"])
    info["params"] = cosmo
    # Fake likelihood so far
    info["likelihood"] = {"one": None}
    from cobaya.model import get_model
    model = get_model(info)

    model.likelihood.theory.needs(cl={"tt": lmax})
    model.logposterior({}) # parameters are fixed
    Dls = model.likelihood.theory.get_cl(ell_factor=True)
    return Dls["tt"]

def bin_spectrum(dl, l, lmin, lmax, delta_l):
    Nbin = np.int(lmax/delta_l)
    db = np.zeros(Nbin)
    lb = np.zeros(Nbin)
    for i in range(Nbin):
        idx = np.where((l> i*delta_l) & (l< (i+1)*delta_l))
        db[i] = np.mean(dl[idx])
        lb[i] = np.mean(l[idx])
    idx = np.where(lb>lmin)
    lb,db = lb[idx],db[idx]
    return lb, db

def bin_variance(vl, l, lmin, lmax, delta_l):
    Nbin = np.int(lmax/delta_l)
    vb = np.zeros(Nbin)
    lb = np.zeros(Nbin)
    for i in range(Nbin):
        idx = np.where((l>i*delta_l) & (l<(i+1)*delta_l))
        vb[i] = np.sum(1/vl[idx])
        lb[i] = np.mean(l[idx])
    vb=1/vb

    idx = np.where(lb>lmin)
    lb, vb = lb[idx], vb[idx]
    return lb, vb

def writeSpectrum(l,Dl,fileName):

    g = open(fileName,mode="w")
    for k in range(len(l)):
        g.write("%f %e\n"%(l[k],Dl[k]))
    g.close()

def variance(p,l,Dl,DNl_all):
    var= 2./((2*l+1)*p['fsky'])*(Dl+DNl_all)**2
    return var

def generate_residu_from_variance(var):
    res=np.sqrt(var)*np.random.randn(len(var))
    return(res)

def generate_1d_grid(p,param,min,max,num):

    dir='grid_1d_%s'%param
    try:
        os.mkdir(dir)
    except:
        pass
    totCL=get_theory_cls(p)
    ls=np.arange(2,p['lmax'])
    Dl_fid=totCL[:,0]
    Dl_fid=Dl_fid[2:len(ls)+2]

    fid= p[param]
    values=fid+np.linspace(min,max,num)
    np.savetxt(dir+'/grid_values.dat',values)

    count=0
    for v in values:
        print(param,v)
        p[param]= v
        totCL=get_theory_cls(p)
        Dl=totCL[:,0][2:len(ls)+2]
        Diff=Dl-Dl_fid
        lb,Diff_b= bin_spectrum(p,ls,Diff)
        writeSpectrum(lb,Diff_b,dir+'/diff_spectrum_%03d.dat'%count)
        count+=1

def delta2(a,b):
    if a==b:
        return 1
    else:
        return 0

def delta3(a,b,c):
    if (a==b) & (b==c):
        return 1
    else:
        return 0

def delta4(a,b,c,d):
    if (a==b) & (b==c) & (c==d):
        return 1
    else:
        return 0

def f(a,b,c,d,ns):
    result= 1.*ns[a]*(ns[c]*ns[d]*delta2(a,b)-ns[c]*delta3(a,b,d)-ns[d]*delta3(a,b,c)+delta4(a,b,c,d))
    result/=(ns[c]*ns[d]*(ns[a]-delta2(a,c))*(ns[b]-delta2(b,d)))
    return result

def g(a,b,c,d,ns):
    result= 1.*ns[a]*(ns[c]*delta2(a,b)*delta2(c,d)-delta4(a,b,c,d))
    result/=(ns[a]*ns[b]*(ns[c]-delta2(a,c))*(ns[d]-delta2(b,d)))
    return result

def cov(a,b,c,d,ns,ls,Dl,DNl,fsky):
    fac= 1./((2*ls+1)*fsky)

    C=2*Dl**2+Dl*((f(a,c,b,d,ns)+f(a,d,b,c,ns))*DNl[a]+(f(b,d,a,c,ns)+f(b,c,a,d,ns))*DNl[b])+DNl[a]*DNl[b]*(g(a,c,b,d,ns)+g(a,d,b,c,ns))
    return fac*C

def svd_pow(A, exponent):
    E, V = np.linalg.eigh(A)
    return np.einsum("...ab,...b,...cb->...ac",V,E**exponent,V)


def fisher_planck(setup):
    experiment = setup["experiment"]
    lmin, lmax = experiment["lmin"], experiment["lmax"]
    ls = np.arange(lmin, lmax)

    from beyondCV import utils
    from copy import deepcopy

    params=["cosmomc_theta", "As", "omch2", "ns", "ombh2"]
    epsilon=0.01
    deriv={}
    for p in params:
        value=setup["simulation"]["cosmo. parameters"][p]
        setup_mod=deepcopy(setup)
        setup_mod["simulation"]["cosmo. parameters"][p]= (1-epsilon)*value
        Dltt_minus = utils.get_theory_cls(setup_mod, lmax)
        Dltt_minus = Dltt_minus[lmin:lmax]
        setup_mod=deepcopy(setup)
        setup_mod["simulation"]["cosmo. parameters"][p]= (1+epsilon)*value
        Dltt_plus = utils.get_theory_cls(setup_mod, lmax)
        Dltt_plus = Dltt_plus[lmin:lmax]
        deriv[p]= (Dltt_plus-Dltt_minus)/(2*epsilon*value)

    Dltt = utils.get_theory_cls(setup, lmax)
    Dltt = Dltt[lmin:lmax]

    freq_Planck, DNl_array_Planck = utils.get_noise(experiment, "Planck")
    freq_Planck = list(freq_Planck)
    freq_Planck.append('all')

    ns = {}
    DNl = {}
    for freq in freq_Planck:
        key = "Planck_%s" % freq
        ns[key]=2.
        DNl[key]=DNl_array_Planck[freq]*ns[key]

    covmat_PPPP = utils.cov("Planck_all","Planck_all","Planck_all","Planck_all", ns, ls, Dltt, DNl, experiment["fsky"])

    nparam=len(params)
    fisher=np.zeros((nparam,nparam))
    for count1, p1 in enumerate(params):
        for count2, p2 in enumerate(params):
            fisher[count1,count2]=np.sum(covmat_PPPP**-1*deriv[p1]*deriv[p2])
    cov=np.linalg.inv(fisher)
    for count, p in enumerate(params):
        print(p,setup_mod["simulation"]["cosmo. parameters"][p],np.sqrt(cov[count,count]))
