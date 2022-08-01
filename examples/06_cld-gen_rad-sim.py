import os
import sys
import pickle
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import copy
from pyhdf.SD import SD, SDC
from scipy import interpolate

from er3t.util import mmr2vmr, cal_rho_air, downgrading
from er3t.pre.atm import atm_atmmod



__all__ = ['cld_gen']





class CLD2D_GEN:

    def __init__(self,
            cldGridSizeHalf = 1,       # number of grid space (squared shape)
            cldFracLimit    = 0.2,
            domXGrid0       = 200,
            domYGrid0       = 200,
            domGridSize     = 0.5,
            ):

        self.cldGridSizeHalf = cldGridSizeHalf
        self.marginSize      = domGridSize * cldGridSizeHalf
        self.cldSize         = domGridSize * cldGridSizeHalf * 2
        self.cldFracLimit    = cldFracLimit
        self.cldArea0        = self.cldSize ** 2
        self.cldArea         = 0.0
        self.cldNum          = 0

        self.domXGrid0    = domXGrid0
        self.domYGrid0    = domYGrid0
        self.domXGrid     = domXGrid0 + cldGridSizeHalf * 2
        self.domYGrid     = domYGrid0 + cldGridSizeHalf * 2

        self.domXSize     = domGridSize * self.domXGrid
        self.domYSize     = domGridSize * self.domYGrid
        self.domArea      = self.domXSize * self.domYSize

        self.cldFrac      = self.cldArea / self.domArea

        self.cld2d        = np.zeros((self.domXGrid, self.domYGrid), dtype=np.int16)

        self.cldCan       = np.zeros((self.domXGrid, self.domYGrid), dtype=np.int16)
        self.cldCan[cldGridSizeHalf:-cldGridSizeHalf, cldGridSizeHalf:-cldGridSizeHalf] = 1

    def C_CLD_RAND(self):
        cldIndX  = np.random.randint(self.cldGridSizeHalf, high=self.domXGrid0+self.cldGridSizeHalf, size=1)
        cldIndY  = np.random.randint(self.cldGridSizeHalf, high=self.domYGrid0+self.cldGridSizeHalf, size=1)
        print('Generating cloud...')
        while self.cldFrac <= self.cldFracLimit:
            while self.cldCan[cldIndX, cldIndY] != 1:
                cldIndX  = np.random.randint(self.cldGridSizeHalf, high=self.domXGrid0+self.cldGridSizeHalf, size=1)
                cldIndY  = np.random.randint(self.cldGridSizeHalf, high=self.domYGrid0+self.cldGridSizeHalf, size=1)
            cldIndXS1 = cldIndX-self.cldGridSizeHalf
            cldIndXE1 = cldIndX+self.cldGridSizeHalf
            cldIndYS1 = cldIndY-self.cldGridSizeHalf
            cldIndYE1 = cldIndY+self.cldGridSizeHalf
            self.cld2d[cldIndXS1:cldIndXE1, cldIndYS1:cldIndYE1]  = 1

            cldIndXS2 = cldIndX-self.cldGridSizeHalf*2
            if cldIndXS2 < 0:
                cldIndXS2 = 0
            cldIndXE2 = cldIndX+self.cldGridSizeHalf*2
            if cldIndXE2 > self.domXGrid:
                cldIndXE2 = self.domXGrid
            cldIndYS2 = cldIndY-self.cldGridSizeHalf*2
            if cldIndYS2 < 0:
                cldIndYS2 = 0
            cldIndYE2 = cldIndY+self.cldGridSizeHalf*2
            if cldIndYE2 > self.domYGrid:
                cldIndYE2 = self.domYGrid
            self.cldCan[cldIndXS2:cldIndXE2, cldIndYS2:cldIndYE2] = 0

            self.cldArea += self.cldArea0
            self.cldNum  += 1
            self.cldFrac = self.cldArea / self.domArea
            print('  Cloud #%d is created... Now cloud fraction is %f.' % (self.cldNum, self.cldFrac))
        print('2D random cloud field has been generated. Please check object.cld2d.')


def TEST():

    cld2d = CLD2D_GEN(cldGridSizeHalf=3, cldFracLimit=0.45)
    cld2d.C_CLD_RAND()
    import matplotlib.pyplot as plt
    fig = plt.figure(figsize=(6, 6))
    ax1 = fig.add_subplot(111)
    cs1 = plt.imshow(cld2d.cld2d*0.5, cmap='Greys', origin='lower')
    ax1.set_xlim([0, 100])
    ax1.set_ylim([0, 100])
    plt.show()

def CDATA():
    import h5py
    for cldFracLimit in [0.1, 0.45]:
        for cldGridSizeHalf in [1, 3]:
            cldSize = cldGridSizeHalf*0.5*2.0
            cld2d = CLD2D_GEN(cldGridSizeHalf=cldGridSizeHalf, cldFracLimit=cldFracLimit, domXGrid0=200-2*cldGridSizeHalf, domYGrid0=200-2*cldGridSizeHalf)
            cld2d.C_CLD_RAND()
            fname = 'cldfrac-%.2f_cldsize-%.2f.h5' % (cldFracLimit, cldSize)
            f = h5py.File(fname, 'w')
            f['cld_field'] = cld2d.cld2d
            f['cld_size']  = cldSize
            f['cld_frac']  = cld2d.cldFrac
            f['dom_grid_size'] = 0.5
            f['dom_size_x'] = cld2d.domXSize
            f['dom_size_y'] = cld2d.domYSize
            f.close()

def PLT_DATA():
    import h5py
    for cldFracLimit in [0.1, 0.45]:
        for cldGridSizeHalf in [1, 3]:
            #cld2d = CLD2D_GEN(cldGridSizeHalf=cldGridSizeHalf, cldFracLimit=cldFracLimit)
            #cld2d.C_CLD_RAND()
            cldSize = cldGridSizeHalf*0.5*2.0
            fname = 'cldfrac-%.2f_cldsize-%.2f.h5' % (cldFracLimit, cldSize)
            f = h5py.File(fname, 'r')
            cld_field = f['cld_field'][...]
            cld_size  = f['cld_size'][...]
            cld_frac  = f['cld_frac'][...]
            dom_size_x= f['dom_size_x'][...]
            dom_size_y= f['dom_size_y'][...]
            f.close()
            print(cld_size, cld_frac, dom_size_x, dom_size_y)

            import matplotlib.pyplot as plt
            fig = plt.figure(figsize=(6, 6.2))
            ax1 = fig.add_subplot(111)
            cs1 = plt.imshow(cld_field, cmap='Greys', origin='lower')
            ax1.set_xlim([0, cld_field.shape[0]-1])
            ax1.set_ylim([0, cld_field.shape[1]-1])
            plt.savefig('%s.png' % fname[:-3])

class CLD2D_GEN:

    def __init__(self,
            cldFracLimit    = 0.3,
            domGridNX       = 1000,
            domGridNY       = 1000,
            domGridSize     = 1.0
            ):

        self.cldFracLimit    = cldFracLimit
        self.cldAreaTotal    = 0.0
        self.cldNumTotal     = 0

        self.domGridSize  = domGridSize
        self.domGridNX    = domGridNX
        self.domGridNY    = domGridNY
        self.domSizeX     = domGridSize * self.domGridNX
        self.domSizeY     = domGridSize * self.domGridNY
        self.domArea      = self.domSizeX * self.domSizeY

        self.cldFrac      = self.cldAreaTotal / self.domArea

        self.cld2d        = np.zeros((self.domGridNX, self.domGridNY), dtype=np.int8)
        #self.cldIndPool   = np.zeros((self.domGridNX, self.domGridNY), dtype=np.int8)
        self.cldIndPool   = np.ones((self.domGridNX, self.domGridNY), dtype=np.int8)


    def C_CLD_FIELD(self,
            halfCldGridNX = 2,
            halfCldGridNY = 2,
            verbose       = False
            ):

        self.cldIndPool[halfCldGridNX:self.domGridNX-halfCldGridNX, halfCldGridNY:self.domGridNY-halfCldGridNY] = 0
        cldIndX  = np.random.randint(halfCldGridNX, high=self.domGridNX-halfCldGridNX+1, size=1)
        cldIndY  = np.random.randint(halfCldGridNY, high=self.domGridNY-halfCldGridNY+1, size=1)

        print('Generating cloud...')
        while self.cldFrac <= self.cldFracLimit:
            loopRecordNum = 0
            while self.cldIndPool[cldIndX-halfCldGridNX:cldIndX+halfCldGridNX, cldIndY-halfCldGridNY:cldIndY+halfCldGridNY].sum() != 0:
                cldIndX  = np.random.randint(halfCldGridNX, high=self.domGridNX-halfCldGridNX+1, size=1)
                cldIndY  = np.random.randint(halfCldGridNY, high=self.domGridNY-halfCldGridNY+1, size=1)
                loopRecordNum += 1
                if loopRecordNum >= 1e6:
                    print('Endless loop occurs... Exit.')
                    exit()

            cldIndXS = cldIndX-halfCldGridNX
            cldIndXE = cldIndX+halfCldGridNX
            cldIndYS = cldIndY-halfCldGridNY
            cldIndYE = cldIndY+halfCldGridNY

            self.cld2d[cldIndXS:cldIndXE, cldIndYS:cldIndYE]       = 1
            self.cldIndPool[cldIndXS:cldIndXE, cldIndYS:cldIndYE]  = 1

            #cldIndXS2 = cldIndX-self.cldGridSizeHalf*2
            #if cldIndXS2 < 0:
                #cldIndXS2 = 0
            #cldIndXE2 = cldIndX+self.cldGridSizeHalf*2
            #if cldIndXE2 > self.domXGrid-1:
                #cldIndXE2 = self.domXGrid-1
            #cldIndYS2 = cldIndY-self.cldGridSizeHalf*2
            #if cldIndYS2 < 0:
                #cldIndYS2 = 0
            #cldIndYE2 = cldIndY+self.cldGridSizeHalf*2
            #if cldIndYE2 > self.domYGrid-1:
                #cldIndXE2 = self.domYGrid-1

            #halfCldGridNX = halfCldGridNX + 1
            #halfCldGridNY = halfCldGridNY + 1

            self.cldAreaTotal += (halfCldGridNX * 2 * self.domGridSize) * (halfCldGridNY * 2 * self.domGridSize)
            self.cldNumTotal  += 1
            self.cldFrac = self.cldAreaTotal / self.domArea
            if verbose:
                print('  Cloud #%d is created... Now cloud fraction is %f.' % (self.cldNumTotal, self.cldFrac))
        print('2D random cloud field has been generated. Please check object.cld2d.')

def TEST(cldFrac, runNum):

    cld2d = CLD2D_GEN(cldFracLimit=cldFrac, domGridNX=1000, domGridNY=1000)
    #halfCldGridNX = 20
    #halfCldGridNY = 20
    halfCldGridNX = 50
    halfCldGridNY = 50
    cld2d.C_CLD_FIELD(halfCldGridNX=halfCldGridNX, halfCldGridNY=halfCldGridNY, verbose=False)
    cldSizeX = halfCldGridNX * 2 * cld2d.domGridSize
    cldSizeY = halfCldGridNY * 2 * cld2d.domGridSize

    print((cld2d.cld2d == 1).sum() * 1.0 / (cld2d.cld2d.ravel().size))

    import matplotlib.pyplot as plt
    fig = plt.figure(figsize=(6, 6))
    ax1 = fig.add_subplot(111)
    cs1 = plt.imshow(cld2d.cld2d, cmap='Greys', origin='lower')
    plt.xlabel('X Grid Index')
    plt.ylabel('Y Grid Index')
    fig.suptitle('2D Random Cloud Field', fontsize=20, y=0.98)
    plt.title('(Cloud Size: %.1fkm$\\times$%.1fkm; Cloud Fraction: %.1f)' % (cldSizeX/10.0, cldSizeY/10.0, cld2d.cldFracLimit), fontsize=12, y=1.015)
    plt.savefig('squared_clouds_%.1f_%2.2d.png' % (cldFrac, runNum))
    #plt.show()




class cld_gen:

    """
    Input:
        fname=    : keyword argument, default=None, the file path of the Python pickle file
        coarsing= : keyword argument, default=[1, 1, 1], the parameter to downgrade the data in [x, y, z]
        overwrite=: keyword argument, default=False, whether to overwrite or not
        verbose=  : keyword argument, default=False, verbose tag

    Output:
        self.lay
                ['x']
                ['y']
                ['altitude']
                ['pressure']
                ['temperature']   (x, y, z)
                ['extinction']    (x, y, z)

        self.lev
                ['altitude']
    """


    ID = 'Gnerate Cloud 3D'


    def __init__(self, \
                 fname     = None, \
                 extent    = None, \
                 overwrite = False, \
                 verbose   = False):

        self.verbose  = verbose     # verbose tag
        self.fname    = fname       # file name of the pickle file
        self.extent   = extent

        if ((self.fname is not None) and (os.path.exists(self.fname)) and (not overwrite)):

            self.load(self.fname)

        elif ((self.fname is not None) and (os.path.exists(self.fname)) and (overwrite)) or \
             ((self.fname is not None) and (not os.path.exists(self.fname))):

            self.dump(self.fname)

        elif ((self.fname is None)):

            self.run()

        else:

            sys.exit('Error   [cld_gen]: Please check if \'%s\' exists.' % self.fname)


    def load(self, fname):

        with open(fname, 'rb') as f:
            obj = pickle.load(f)
            if hasattr(obj, 'lev') and hasattr(obj, 'lay'):
                if self.verbose:
                    print('Message [cld_gen]: loading %s ...' % fname)
                self.fname = obj.fname
                self.lay   = obj.lay
                self.lev   = obj.lev
            else:
                sys.exit('Error   [cld_gen]: %s is not the correct \'pickle\' file to load.' % fname)


    def run(self):

        if self.verbose:
            print("Message [cld_gen]: Processing %s ..." % fname_h4)

        # pre process
        self.pre_gen()

        # downgrade data if needed
        # if any([i!=1 for i in self.coarsing]):
        #     self.downgrade(self.coarsing)

        # post process
        # self.post_gen()


    def pre_gen(self, earth_radius=6378.0, cloud_thickness=1.0):

        self.lay = {}
        self.lev = {}

        f     = SD(self.fname_h4, SDC.READ)

        # lon lat
        lon0       = f.select('Longitude')
        lat0       = f.select('Latitude')
        cot0       = f.select('Cloud_Optical_Thickness_16')
        cer0       = f.select('Cloud_Effective_Radius_16')
        cot_pcl0   = f.select('Cloud_Optical_Thickness_16_PCL')
        cer_pcl0   = f.select('Cloud_Effective_Radius_16_PCL')
        cth0       = f.select('Cloud_Top_Height')


        if 'actual_range' in lon0.attributes().keys():
            lon_range = lon0.attributes()['actual_range']
            lat_range = lat0.attributes()['actual_range']
        else:
            lon_range = [-180.0, 180.0]
            lat_range = [-90.0 , 90.0]

        lon       = lon0[:]
        lat       = lat0[:]
        cot       = np.float_(cot0[:])
        cer       = np.float_(cer0[:])
        cot_pcl   = np.float_(cot_pcl0[:])
        cer_pcl   = np.float_(cer_pcl0[:])
        cth       = np.float_(cth0[:])

        logic     = (lon>=lon_range[0]) & (lon<=lon_range[1]) & (lat>=lat_range[0]) & (lat<=lat_range[1])
        lon       = lon[logic]
        lat       = lat[logic]
        cot       = cot[logic]
        cer       = cer[logic]
        cot_pcl   = cot_pcl[logic]
        cer_pcl   = cer_pcl[logic]
        cth       = cth[logic]

        if self.extent is not None:
            logic     = (lon>=self.extent[0])&(lon<=self.extent[1])&(lat>=self.extent[2])&(lat<=self.extent[3])
            lon       = lon[logic]
            lat       = lat[logic]
            cot       = cot[logic]
            cer       = cer[logic]
            cot_pcl   = cot_pcl[logic]
            cer_pcl   = cer_pcl[logic]
            cth       = cth[logic]

        xy = (self.extent[1]-self.extent[0])*(self.extent[3]-self.extent[2])
        N0 = np.sqrt(lon.size/xy)

        Nx = int(N0*(self.extent[1]-self.extent[0]))
        if Nx%2 == 1:
            Nx += 1

        Ny = int(N0*(self.extent[3]-self.extent[2]))
        if Ny%2 == 1:
            Ny += 1

        lon_1d0 = np.linspace(self.extent[0], self.extent[1], Nx+1)
        lat_1d0 = np.linspace(self.extent[2], self.extent[3], Ny+1)

        lon_1d = (lon_1d0[1:]+lon_1d0[:-1])/2.0
        lat_1d = (lat_1d0[1:]+lat_1d0[:-1])/2.0

        dx = (lon_1d[1]-lon_1d[0])/180.0 * (np.pi*earth_radius)
        dy = (lat_1d[1]-lat_1d[0])/180.0 * (np.pi*earth_radius)

        x_1d = (lon_1d-lon_1d[0])*dx
        y_1d = (lat_1d-lat_1d[0])*dy


        # lon, lat
        lat_2d, lon_2d = np.meshgrid(lat_1d, lon_1d)
        # lon_2d, lat_2d = np.meshgrid(lon_1d, lat_1d)

        # cot
        cot_range     = cot0.attributes()['valid_range']
        cer_range     = cer0.attributes()['valid_range']
        cot_pcl_range = cot_pcl0.attributes()['valid_range']
        cer_pcl_range = cer_pcl0.attributes()['valid_range']
        cth_range     = cth0.attributes()['valid_range']

        # +
        # create cot_all/cer_all that contains both cot/cer and cot_pcl/cer_pcl
        cot_all = np.zeros(cot.size, dtype=np.float64)
        cer_all = np.zeros(cer.size, dtype=np.float64)
        cth_all = np.zeros(cth.size, dtype=np.float64); cth_all[...] = np.nan

        logic = (cot>=cot_range[0]) & (cot<=cot_range[1]) & (cer>=cer_range[0]) & (cer<=cer_range[1]) & (cth>=cth_range[0]) & (cth<=cth_range[1])
        cot_all[logic] = cot[logic]*cot0.attributes()['scale_factor'] + cot0.attributes()['add_offset']
        cer_all[logic] = cer[logic]*cer0.attributes()['scale_factor'] + cer0.attributes()['add_offset']
        cth_all[logic] = cth[logic]*cth0.attributes()['scale_factor'] + cth0.attributes()['add_offset']

        logic_pcl = np.logical_not(logic) & (cot_pcl>=cot_pcl_range[0]) & (cot_pcl<=cot_pcl_range[1]) & (cer_pcl>=cer_pcl_range[0]) & (cer_pcl<=cer_pcl_range[1]) & (cth>=cth_range[0]) & (cth<=cth_range[1])
        cot_all[logic_pcl] = cot_pcl[logic_pcl]*cot_pcl0.attributes()['scale_factor'] + cot_pcl0.attributes()['add_offset']
        cer_all[logic_pcl] = cer_pcl[logic_pcl]*cer_pcl0.attributes()['scale_factor'] + cer_pcl0.attributes()['add_offset']
        cth_all[logic_pcl] = cth[logic_pcl]*cth0.attributes()['scale_factor'] + cth0.attributes()['add_offset']
        cth_all /= 1000.0

        logic_all = logic | logic_pcl
        # -

        cot = cot_all
        cer = cer_all
        cth = cth_all

        cot[np.logical_not(logic_all)] = 0.0
        cer[np.logical_not(logic_all)] = 0.0
        cth[np.logical_not(logic_all)] = np.nan

        points = np.transpose(np.vstack((lon, lat)))

        cot_2d = interpolate.griddata(points, cot, (lon_2d, lat_2d), method='nearest')
        cer_2d = interpolate.griddata(points, cer, (lon_2d, lat_2d), method='nearest')
        cth_2d = interpolate.griddata(points, cth, (lon_2d, lat_2d), method='nearest')

        f.end()

        self.atm = atm_atmmod(np.arange(int(np.nanmax(cth_2d))+2))
        self.lay['x']  = {'data':x_1d     , 'name':'X'          , 'units':'km'}
        self.lay['y']  = {'data':y_1d     , 'name':'Y'          , 'units':'km'}
        self.lay['nx'] = {'data':Nx       , 'name':'Nx'         , 'units':'N/A'}
        self.lay['ny'] = {'data':Ny       , 'name':'Ny'         , 'units':'N/A'}
        self.lay['dx'] = {'data':dx       , 'name':'dx'         , 'units':'km'}
        self.lay['dy'] = {'data':dy       , 'name':'dy'         , 'units':'km'}
        self.lay['altitude'] = copy.deepcopy(self.atm.lay['altitude'])
        self.lay['cot']= {'data':cot_2d   , 'name':'Cloud optical thickness', 'units':'N/A'}
        self.lay['cer']= {'data':cer_2d   , 'name':'Cloud effective radius' , 'units':'micron'}
        self.lay['cth']= {'data':cth_2d   , 'name':'Cloud top height'       , 'units':'km'}
        self.lay['lon']= {'data':lon_2d   , 'name':'Longitude'              , 'units':'degree'}
        self.lay['lat']= {'data':lat_2d   , 'name':'Latitude'               , 'units':'degree'}


        # temperature 3d
        t_1d = self.atm.lay['temperature']['data']
        Nz   = t_1d.size
        t_3d      = np.empty((Nx, Ny, Nz), dtype=t_1d.dtype)
        t_3d[...] = t_1d[None, None, :]

        self.lay['temperature'] = {'data':t_3d, 'name':'Temperature', 'units':'K'}


        # extinction 3d
        ext_3d      = np.zeros((Nx, Ny, Nz), dtype=np.float64)

        alt = self.atm.lay['altitude']['data']
        for i in range(Nx):
            for j in range(Ny):

                cld_top  = cth_2d[i, j]

                if not np.isnan(cld_top):
                    lwp  = 5.0/9.0 * 1.0 * cot_2d[i, j] * cer_2d[i, j] / 10.0
                    ext0 = 0.75 * 2.0 * lwp / cer_2d[i, j] / 100.0

                    # index = np.argmin(np.abs(cld_top-alt))
                    index = 0
                    ext_3d[i, j, index] = ext0

        self.lay['extinction'] = {'data':ext_3d, 'name':'Extinction coefficients'}

        self.Nx = Nx
        self.Ny = Ny
        self.Nz = Nz


    def pre_gen(self,
            cldGridSizeHalf = 1,       # number of grid space (squared shape)
            cldFracLimit    = 0.2,
            domXGrid0       = 200,
            domYGrid0       = 200,
            domGridSize     = 0.5,
            ):

        print('haha')
        exit()

        self.cldGridSizeHalf = cldGridSizeHalf
        self.marginSize      = domGridSize * cldGridSizeHalf
        self.cldSize         = domGridSize * cldGridSizeHalf * 2
        self.cldFracLimit    = cldFracLimit
        self.cldArea0        = self.cldSize ** 2
        self.cldArea         = 0.0
        self.cldNum          = 0

        self.domXGrid0    = domXGrid0
        self.domYGrid0    = domYGrid0
        self.domXGrid     = domXGrid0 + cldGridSizeHalf * 2
        self.domYGrid     = domYGrid0 + cldGridSizeHalf * 2

        self.domXSize     = domGridSize * self.domXGrid
        self.domYSize     = domGridSize * self.domYGrid
        self.domArea      = self.domXSize * self.domYSize

        self.cldFrac      = self.cldArea / self.domArea

        self.cld2d        = np.zeros((self.domXGrid, self.domYGrid), dtype=np.int16)

        self.cldCan       = np.zeros((self.domXGrid, self.domYGrid), dtype=np.int16)
        self.cldCan[cldGridSizeHalf:-cldGridSizeHalf, cldGridSizeHalf:-cldGridSizeHalf] = 1


    def dump(self, fname):

        self.fname = fname
        with open(fname, 'wb') as f:
            if self.verbose:
                print('Message [cld_gen]: saving object into %s ...' % fname)
            pickle.dump(self, f)



    def downgrade(self, coarsing):

        dnx, dny, dnz = coarsing

        if (self.Nx%dnx != 0) or (self.Ny%dny != 0) or \
           (self.Nz%dnz != 0):
            sys.exit('Error   [cld_gen]: the original dimension %s is not divisible with %s, please check input (dnx, dny, dnz).' % (str(self.lay['Temperature']['data'].shape), str(coarsing)))
        else:
            new_shape = (self.Nx//dnx, self.Ny//dny, self.Nz//dnz)

            if self.verbose:
                print('Message [cld_gen]: Downgrading data from dimension %s to %s ...' % (str(self.lay['Temperature']['data'].shape), str(new_shape)))

            self.lay['x']['data']        = downgrading(self.lay['x']['data']       , (self.Nx//dnx,))
            self.lay['y']['data']        = downgrading(self.lay['y']['data']       , (self.Ny//dny,))
            self.lay['altitude']['data'] = downgrading(self.lay['altitude']['data'], (self.Nz//dnz,))

            for key in self.lay.keys():
                if isinstance(self.lay[key]['data'], np.ndarray):
                    if self.lay[key]['data'].ndim == len(coarsing):
                        self.lay[key]['data']  = downgrading(self.lay[key]['data'], new_shape)


    def post_gen(self):

        dz  = self.lay['altitude']['data'][1:]-self.lay['altitude']['data'][:-1]
        dz0 = dz[0]
        diff = np.abs(dz-dz0)
        if any([i>0.001 for i in diff]):
            print(dz0, dz)
            sys.exit('Error   [cld_gen]: Non-equidistant intervals found in \'dz\'.')
        else:
            dz  = np.append(dz, dz0)
            alt = np.append(self.lay['altitude']['data']-dz0/2.0, self.lay['altitude']['data'][-1]+dz0/2.0)

        self.lev['altitude']    = {'data':alt, 'name':'Altitude'       , 'units':'km'}

        self.lay['thickness']   = {'data':dz , 'name':'Layer thickness', 'units':'km'}


class cld_gen:

    """
    Generate 3D cloud field (hemispherical clouds)
    """

    def __init__(self, Nx=100, Ny=100, dpi=100, cloudR=10):

        self.Nx = Nx
        self.Ny = Ny

        self.clouds = []

        self.fig = plt.figure(frameon=False, figsize=(Nx/dpi, Ny/dpi), dpi=dpi)
        self.ax  = self.fig.add_axes([0.0, 0.0, 1.0, 1.0])

        self.ax.set_facecolor('white')
        self.ax.axis('off')
        self.ax.set_xlim([0, Nx])
        self.ax.set_ylim([0, Ny])

        # c1 = mpatches.CirclePolygon((50, 50), 50, color='k', lw=0.0)
        c1 = mpatches.Ellipse((50, 50), 100, 100, angle=45, color='k', lw=0.0)
        self.ax.add_artist(c1)


        self.fig.canvas.draw()
        buff = self.fig.canvas.tostring_rgb()
        ncols, nrows = self.fig.canvas.get_width_height()
        data = np.frombuffer(buff, dtype=np.uint8).reshape(nrows, ncols, 3)[:, :, 0]

        print(data[:, 99])

        self.fig.canvas.print_png('haha.png')

    def _add_a_cloud():

        pass

    def _update_2d(self):

        self.fig.canvas.draw()
        buff = self.fig.canvas.tostring_rgb()
        data = np.frombuffer(buff, dtype=np.uint8).reshape(nrows, ncols, 3)[:, :, 0]

    def _cloud_records(self, x, y, width, height, angle):

        cloud0 = {
                'ID': len(self.clouds),
                'x' : x,
                'y' : y,
                'width' : width,
                'height': height,
                'angle' : angle
                }

        self.clouds.append(cloud0)


if __name__ == '__main__':

    cld = cld_gen()

    pass
