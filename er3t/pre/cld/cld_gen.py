import os
import sys
import pickle
import numpy as np


__all__ = ['cld_gen_hem']



class cld_gen_hem:

    """
    Purpose: generate 3D cloud field (hemispherical clouds)

    Input:
        Nx=: keyword argument, default=400, number of pixel in x of 3D space
        Ny=: keyword argument, default=400, number of pixel in y of 3D space
        Nz=: keyword argument, default=20, number of pixel in z of 3D space
        dx=: keyword argument, default=100, delta length in x per pixel (units: meter)
        dy=: keyword argument, default=100, delta length in y per pixel (units: meter)
        dz=: keyword argument, default=500, delta length in z per pixel (units: meter)
        radii=      : keyword argument, default=[5000], a pool of radius of clouds that will be radomly picked from (units: meter)
        weights=    : keyword argument, default=None (evenly pick from radii), possibilities for picking the size of clouds specified in radii
        w2h_ratio=  : keyword argmument, default=1.0, width (x) to height (z) ratio, smaller the number, taller the clouds
        min_dist=   : keyword argument, default=0, minimum distance between each two clouds, the larger the number, the more sparse of the cloud fields
        cloud_frac= : keyword argument, default=0.2, target cloud fraction for the generated cloud field
        trial_limit=: keyword argument, default=100, number of trials if the cloud scene is too full to add new clouds
        overlap=    : keyword argument, default=False, whether different clouds can overlap

    Output:
        self.lay
                ['x']
                ['y']
                ['altitude']
                ['pressure']
                ['temperature']   (x, y, z)
                ['extinction']    (x, y, z)
                ['cot']           (x, y, z)
                ['cer']           (x, y, z)

        self.lev
                ['altitude']

    """

    ID = 'Hemispherical Cloud 3D'

    def __init__(
            self,
            Nx=400,
            Ny=400,
            Nz=20,
            dx=100,
            dy=100,
            dz=500,
            radii=[5000],
            weights=None,
            w2h_ratio=1.0,
            min_dist=0,
            cloud_frac=0.2,
            trial_limit=100,
            overlap=False
            ):

        self.Nx = Nx
        self.Ny = Ny
        self.Nz = Nz

        self.dx = dx
        self.dy = dy
        self.dz = dz

        self.w2h_ratio = w2h_ratio
        self.min_dist = min_dist

        self.trial = 0
        self.trial_limit = trial_limit
        self.overlap = overlap

        self.x = np.arange(Nx)
        self.y = np.arange(Ny)
        self.z = np.arange(Nz)

        self.x_2d, self.y_2d = np.meshgrid(self.x, self.y, indexing='ij')
        self.x_3d, self.y_3d, self.z_3d= np.meshgrid(self.x, self.y, self.z, indexing='ij')

        self.clouds   = []
        self.space_3d = np.zeros_like(self.x_3d)
        self.where_2d = np.ones((Nx, Ny))
        self.can_add_more = True
        self.cloud_frac = 0.0

        radii = np.array(radii)
        if weights is not None:
            weights = np.array(weights)

        while (self.cloud_frac<cloud_frac) and (self.can_add_more):

            self._add_a_cloud(np.random.choice(radii, p=weights), min_dist=self.min_dist, w2h_ratio=self.w2h_ratio, limit=1)

        self._cloud_optical_prop()

    def _add_a_cloud(self, radius, min_dist=0, w2h_ratio=1.0, limit=1):

        """
        radius: radius of the hemispherical cloud, units in meter
        min_dist: minimum distance between clouds - the larger the value, the more distant away from cloud to cloud
        limit: when to stop adding more clouds to avoid overlap
        """

        if not self.overlap:
            self.where_2d = np.ones_like(self.where_2d)
            for cloud0 in self.clouds:
                logic_no = (((self.x_2d-cloud0['x'])*self.dx)**2 + ((self.y_2d-cloud0['y'])*self.dy)**2) <= (cloud0['radius']+radius+min_dist)**2
                self.where_2d[logic_no] = 0

        indices = np.where(self.where_2d==1)
        N_avail = indices[0].size

        if N_avail > limit:

            index = np.random.randint(0, N_avail-1)
            loc_x = indices[0][index]
            loc_y = indices[1][index]

            ndx = radius//self.dx
            index_x_s = max((0, loc_x-ndx-1))
            index_x_e = min((self.x.size-1, loc_x+ndx+1))

            ndy = radius//self.dy
            index_y_s = max((0, loc_y-ndy-1))
            index_y_e = min((self.y.size-1, loc_y+ndy+1))

            logic_cloud0 = (((self.x_3d[index_x_s:index_x_e, index_y_s:index_y_e, :]-loc_x)*self.dx)**2 + \
                            ((self.y_3d[index_x_s:index_x_e, index_y_s:index_y_e, :]-loc_y)*self.dy)**2 + \
                             (self.z_3d[index_x_s:index_x_e, index_y_s:index_y_e, :]*self.dz*w2h_ratio)**2) <= radius**2
            self.space_3d[index_x_s:index_x_e, index_y_s:index_y_e, :][logic_cloud0] = 1

            # add this newly created cloud into self.clouds
            # =============================================================================
            cloud0 = {
                    'ID': len(self.clouds),
                    'x' : loc_x,
                    'y' : loc_y,
                    'radius': radius
                    }

            self.clouds.append(cloud0)
            # =============================================================================

            self.cloud_frac = self.space_3d[:, :, 0].sum()/(self.Nx*self.Ny)

            self.trial = 0

        else:

            self.trial += 1
            if self.trial >= self.trial_limit:
                self.can_add_more = False

    def _cloud_optical_prop(self, extinction0=1.5e-6):

        """
        Assign cloud optical properties, e.g., cloud optical thickness, cloud effective radius

        extinction0: volume extinction per m^3
        """

        extinction_grid = extinction0 * self.dx*self.dy*self.dz

        self.ext_3d = extinction_grid * self.space_3d

        self.cot_2d = np.sum(self.ext_3d, axis=-1)



if __name__ == '__main__':

    pass
