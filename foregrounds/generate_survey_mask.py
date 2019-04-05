import healpy as hp, pylab as plt,numpy as np
import matplotlib as mpl
import os

nside=2048
npix = hp.nside2npix(nside)
pix = np.arange(npix)
t,p = hp.pix2ang(nside,pix)

mapDir = 'SO_planck_mask'
try:
    os.makedirs(mapDir)
except:
    pass

degToRad=np.pi/180

lat=113
el=40
lat_min=lat-el
lat_max=lat+el
id=np.where((t>(lat_min)*degToRad) & (t<(lat_max)*degToRad))
survey_mask=np.zeros(12*nside**2)
survey_mask[id]=1


r = hp.Rotator(coord=['G','C'])
trot, prot = r(t,p)
id_in = hp.ang2pix(nside, trot, prot)
survey_mask_rot = survey_mask[id_in]

field_gal={}
field_gal[100]=4
field_gal[143]=3
field_gal[217]=2

field_pts={}
field_pts[100]=0
field_pts[143]=1
field_pts[217]=2

freqs=[100,143,217]

mask={}

for f in freqs:
    masks_gal=hp.fitsfunc.read_map('masks/HFI_Mask_GalPlane-apo0_2048_R2.00.fits',field=field_gal[f])
    masks_pts=hp.fitsfunc.read_map('masks/HFI_Mask_PointSrc_2048_R2.00.fits',field=field_pts[f])
    mask[f]=survey_mask_rot*masks_gal#*masks_pts
    hp.write_map('%s/mask_%s.fits'%(mapDir,f),mask[f],overwrite=True)
    
mtot=0
for f in freqs:
    mtot+=(1-mask[f])

cmap = plt.cm.Blues
cmaplist = [cmap(i) for i in range(cmap.N)]
cmaplist[0] = (.5,.5,.5,1.0)
cmap = cmap.from_list('Custom cmap', cmaplist, cmap.N)
cmap.set_under('w')
bounds = np.linspace(0,5,6)
norm = mpl.colors.BoundaryNorm(bounds, cmap.N)
ticks=[' 217-225 GHz mask',' 143-145 GHz mask',' 93-100 GHz mask','SO mask']
fig, ax = plt.subplots(1,1, figsize=(12,4))
hp.mollview(mtot,cmap=cmap,hold=True,cbar=False,notext=True,title='')
ax2 = fig.add_axes([0.80, 0.1, 0.03, 0.8])
cb = mpl.colorbar.ColorbarBase(ax2, cmap=cmap, norm=norm, spacing='proportional', ticks=bounds[1:]+0.5, boundaries=bounds,format='%s')
cb.ax.set_yticklabels(ticks)
plt.savefig('plots/masks.pdf',bbox_inches='tight')
plt.clf()
plt.close()
