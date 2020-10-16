"""
author: Sebastian Schmidt, 12/15/2017

Create "cloud ids" - hence the name "CID"

This code package identifies a contiguous group of pixels such as an individual
cloud from a collection of 2D pixels in a pool. It starts with a pixel in the
pool and looks for neighbors, which it moves into the pool.

The "grow" method is called recursively to incorporate neighbors, their
neighbors, and so on, until no more neighbors can be found.

From the remaining pixel pool, new cloud objects can be created as instances
of the SIMPLE_CLOUD class until the pool is exhausted.

"SIMPLE_CLOUD" can only be used for fairly small objects because python has
a limit how far it can step down in a recursion. To overcome this problem,
two other classes are included: "CLOUD" and "CLOUDS" (a collection of clouds).

The "CLOUD" class is a version of "CLOUD_SIMPLE" where any neighbors exceeding
the recursion depth are returbed as object attributes.

The "CLOUDS" class converts all pixels in the initial pool into clouds, consolidating
individual CLOUD objects when they are adjacent.

"""

import sys
import numpy as np



class cloud_one:

    def __init__(self,pool,start=0,ID=None,nmax=900):
        self.pool   = pool        # assign pool of available pixels
        self.pix    = []          # pixels in object
        self.id     = ID          # ID of cloud (optional)
        self.n      = 0           # number of pixels
        self.nmax   = nmax        # maximum recursion depth
        self.nbr    = []          # valid but unmatched neighbors after exceeding maximum number of pixels
        verb        = False       # print out changes to cloud
        self.grow(idx=start,verb=verb)
        self.slim()               # check for neighbors that are actually already in object

    def grow(self,idx=0,verb=True): # idx specifies where in the pool to start the search
        i=self.pool[idx][0]
        j=self.pool[idx][1]
        if [i,j] in self.pool:
            self.n=self.n+1
            self.pool.remove([i,j])      # Remove pixel from pool
            self.pix.append([i,j])       # ...and add it to cloud object
            if verb: print('ID',self.id,'append',i,j)

        Di=[-1,0,1] # search pattern x direction
        Dj=[-1,0,1] # search pattern y direction
        if self.n<self.nmax: # Find neighbors
            for di in Di:
                for dj in Dj:
                    if [i+di,j+dj] in self.pool:
                        self.grow(idx=self.pool.index([i+di,j+dj]),verb=verb)
        else:
            idxs=[]     # neighbor indeces
            for di in Di:
                for dj in Dj:
                    if [i+di,j+dj] in self.pool:
                        idxs.append(self.pool.index([i+di,j+dj]))
            if len(idxs)>0:
                nbr = self.getpix(idxs)
                if verb: print('ID',self.id,'new neighbor',nbr)
                self.addnbr(nbr)
                if verb: print('ID',self.id,'neighbors',self.nbr)

    # get actual pixels (x,y) for index array within pool
    def getpix(self,idxs):
        xy=[]
        for idx in idxs:
            xy.append(self.pool[idx])
        return(xy)

    # add new neighbors to neighbor list
    def addnbr(self,nbrs):
        for nbr in nbrs:
            if len(self.nbr) == 0:
                self.nbr.append(nbr)
            else:
                if not nbr in self.nbr:
                    self.nbr.append(nbr)

    # get rid of neighbors that are actually already in object
    def slim(self):
        noneighbors=[]
        for nbr in self.nbr: # identify neighbors that are part of object
            if nbr in self.pix: noneighbors.append(nbr)
        for noneighbor in noneighbors:
            self.nbr.remove(noneighbor)

    def mergewith(self,mate):
        verb=False
        if verb: print('initiating mating:')
        if verb: print(self.id,mate.id)
        for matepix in mate.pix:
            if verb: print("mate's pixels",matepix)
            self.pix.append(matepix)
        if verb: print('Mates of mate:',mate.mates)
        for matesmate in mate.mates:
            if verb: print('adding matesmate',matesmate)
            if matesmate not in self.mates: self.mates.append(matesmate)
        self.n=len(self.pix)

    def geometry(self):
        verb = False
        if verb: print("Find edges for ID:",self.id)
        pix=self.pix
        xx=np.array(pix)[:,0]
        yy=np.array(pix)[:,1]
        x0=np.min(xx); x1=np.max(xx); xc=np.mean(xx)
        y0=np.min(yy); y1=np.max(yy); yc=np.mean(yy)
        self.xmin=x0-1
        self.xmax=x1+1
        self.ymin=y0-1
        self.ymax=y1+1
        self.xc=xc
        self.yc=yc
        self.area=len(pix)
        edge=[]
        for p in pix:
            no=0
            if [p[0],p[1]] in pix: no=no+1
            if [p[0]+1,p[1]] in pix: no=no+1
            if [p[0]-1,p[1]] in pix: no=no+1
            if [p[0],p[1]+1] in pix: no=no+1
            if [p[0],p[1]-1] in pix: no=no+1
            if no<5: edge.append(p)
        self.edge=edge
        self.perimeter=len(edge)
        self.radius=2*self.area/self.perimeter
        self.cea_radius=np.sqrt(self.area/np.pi) # circle effective area radius
        self.aspect=self.cea_radius/self.radius  # =1 (circle), =1.12 (square) =1.7 (rectangle with aspect ratio 2)
        self.orientation=(self.ymax-self.ymin)/(self.xmax-self.xmin) # > 1 E-W, < 1 N-S
        self.u_radius=0.5*self.perimeter/np.pi

    def printgeometry(self):
        print('\nGeometry parameters of ID ',self.id)
        print('   aspect ratio             :',self.aspect)
        print('   orientation EW           :',self.orientation)
        print('   orientation NS           :',1./self.orientation)
        print('   perimeter                :',self.perimeter)
        print('   area                     :',self.area)
        print('   equiv. circle radius     :',self.radius)
        print('   equiv. area radius       :',self.cea_radius)
        print('   equiv. perimeter radius  :',self.u_radius)
        print('   main PCA direction (deg) :',self.direction)

    def size(self):
        return(len(self.pix))

    def pca(self):
        #draw = True if (ax in vars()) else False
        verb = False

        if verb: print("\nPCA for ID=",self.id)

        #1 define X and Y data vectors
        Xpos=np.array(self.pix)[:,0]
        Ypos=np.array(self.pix)[:,1]
        n=len(Xpos)
        X = np.reshape(np.array([Xpos,Ypos],dtype=float),(2,n)).T

        #2 Standardize data - not done in this case
        X_std=X
        #3 Covariance matrix
        mean_vec = np.mean(X_std, axis=0)
        cov_mat = (X_std - mean_vec).T.dot((X_std - mean_vec)) / (X_std.shape[0]-1)
        if verb: print(mean_vec)
        #4 Eigenvalue decomposition
        eig_vals, eig_vecs = np.linalg.eig(cov_mat)

        #5 Zip eigenvalues and eigenvectors, then sort them
        # Make a list of (eigenvalue, eigenvector) tuples
        eig_pairs = [(np.abs(eig_vals[i]), eig_vecs[:,i]) for i in range(len(eig_vals))]
        # Sort the (eigenvalue, eigenvector) tuples from high to low
        eig_pairs.sort()
        eig_pairs.reverse()

        #6 convert eigenvalues/vectors into primary directions
        ev_area = eig_pairs[0][0]*eig_pairs[1][0]
        scale=self.area/ev_area
        pca=[]
        for i in range(2):
            sc   = eig_pairs[i][0]*scale
            xd   = eig_pairs[i][1][0]
            yd   = eig_pairs[i][1][1]
            pca.append([sc*xd,sc*yd])
        self.pca=pca

        # determine NS direction in degrees (first PCA only)
        N_vs_S=self.pca[0][0]/self.pca[0][1]
        self.direction=np.rad2deg(np.arctan(N_vs_S))

    def drawbox(self,ax,color='white',linewidth=0.5):
        lw=linewidth
        ax.autoscale(False)
        ax.plot([self.ymin,self.ymin],[self.xmin,self.xmax],color=color,linewidth=lw)
        ax.plot([self.ymax,self.ymax],[self.xmin,self.xmax],color=color,linewidth=lw)
        ax.plot([self.ymin,self.ymax],[self.xmin,self.xmin],color=color,linewidth=lw)
        ax.plot([self.ymin,self.ymax],[self.xmax,self.xmax],color=color,linewidth=lw)
        #ax.scatter(self.yc,self.xc,color=color)
        return ax

    def drawedge(self,ax,color='white',linewidth=0.2,nocirc=False):
        import matplotlib.pyplot as plt
        lw=linewidth
        edge=np.array(self.edge)
        xx  =edge[:,0]
        yy  =edge[:,1]
        ax.scatter(yy,xx,s=lw,c=color)
        if not nocirc:
            circle1 = plt.Circle((self.yc,self.xc),self.radius, color=color,fill=False)
            circle2 = plt.Circle((self.yc,self.xc),self.cea_radius, color=color,fill=False,ls='--')
            circle3 = plt.Circle((self.yc,self.xc),self.u_radius, color=color,fill=False,ls=':')
            ax.add_artist(circle1)
            ax.add_artist(circle2)
            ax.add_artist(circle3)
        return ax

    def drawpca(self,ax,color='red',linewidth=1,chunkmin=0):
        for i in range(2):
            if self.area > chunkmin:
                xd=self.pca[i][0]
                yd=self.pca[i][1]
                ax.plot([self.yc-yd,self.yc+yd],[self.xc-xd,self.xc+xd],'--',color=color,linewidth=linewidth)
        return ax



def mergesets(l):

    """
    from https://stackoverflow.com/questions/4842613/merge-lists-that-share-common-elements
         L = [['a','b','c'],['b','d','e'],['k'],['o','p'],['e','f'],['p','a'],['d','g']]
     --> L = [['a','b','c','d','e','f','g','o','p'],['k']]
    """

    out = []
    while len(l)>0:

        first, *rest = l
        first = set(first)

        lf = -1
        while len(first)>lf:
            lf = len(first)

            rest2 = []
            for r in rest:
                if len(first.intersection(set(r)))>0:
                    first |= set(r)
                else:
                    rest2.append(r)
            rest = rest2

        out.append(first)
        l = rest
    return(out)



class clouds:


    def __init__(self,pool,nmax=900,verb=False):
        n       = len(pool)                      # number of pixels in pool
        chunks  = []                             # this list will hold the contiguous chunks that are made from the pool of pixels
        ctr     = 0                              # counts chunks (used as IDs for individual chunks as well)
        while n>0:                               # runs until pool is emptied
            chunk = cloud_one(pool,nmax=nmax,ID=ctr) # make a new chunk at the expense of the pool
            chunks.append(chunk)                 # insert that chunk into a list of chunks
            ctr   = ctr+1                        # count number of objects
            if verb: print("Found chunk, assigned ID=",chunk.id)
            n     = len(pool)                    # size of remaining pool
        nc=len(chunks)
        if verb: print("Identified ",nc," contiguous chunks.")

        # Step through chunks and find their unmatched neighbors (called "mates")
        # ...with the goal of ultimately joining neighboring chunks (mates) later in the code
        for chunk in chunks:
            chunk.mates=[chunk.id] # initialize list of neighbors
            nn=len(chunk.nbr)      # get number of neighbors for this chunk
            if nn>0:               # proceed only if there are neighbors
                if verb: print('\nchunk ID',chunk.id,' neighbors',chunk.nbr)
                for nbr in chunk.nbr:             # step through all the neighbor pixels of this chunk...
                    for otherchunk in chunks:     # ...and check whether any of the other chunks...
                        if nbr in otherchunk.pix: # ...contain this pixel.
                            if verb: print('ID',chunk.id,'nbr',nbr,'is contained in ID',otherchunk.id)
                            if otherchunk.id not in chunk.mates:   # If that other chunk is not yet a mate, ...
                                chunk.mates.append(otherchunk.id)  # ...add it to the chunk's mates.
                                if verb: print(' -> current mates:',chunk.mates)

        # Combine adjacent chunk IDs (mates) in one list of mate sets.
        # Relationships between mates are generally not mutual.
        # In [{1,2},{2,3,5},{5,6}], for example, 2 is a mate of 1, but not vice versa.
        relationships=[]
        for chunk in chunks:
            relationships.append(chunk.mates)

        # Relationships are transitive. I.e., if 3 is a mate of 2 and 2 is a mate of 1,
        # then 3 is also a mate of 1: [{1,2},{2,8},{5,6}] --> [{1,2,8},{5,6}]
        # Below, relationships are consolidated into groups with the mergesets function.
        groups=mergesets(relationships)
        ng=len(groups) # number of non-contiguous mate groups
        if verb: print("\nThrough relationships, the number of chunks was reduced to",ng,'.')

        appendedmates=[]      # This list will contain the chunks that have been appended to (joined with) their mates.
        for group in groups:  # Go through each relationship with...
            group=list(group)
            np=len(group)     # ...np mates per relationship
            if np > 1:               # If a relationship has at least two mates...
                merge=group[0]       # ...choose the first member of a relationship...
                if chunks[merge].id != merge: sys.exit("ID mismatch")
                for other in group[1:]:                    # ...go through its mates...
                    chunks[merge].mergewith(chunks[other]) # ...merge with them one by one...
                    appendedmates.append(chunks[other])    # ...add them to the "appended" list...

        for chunk in appendedmates:
            chunks.remove(chunk)                           # ...and terminate these chunks.

        if len(chunks) != ng: sys.exit("Did not get expected number of chunks.")
        self.list  = chunks
        self.n     = len(chunks)
        self.edges = None


    def discard(self,chunkmin=400,chunkmax=10000):
        chunks=self.list
        discardablechunks=[]
        for chunk in chunks:
            if (len(chunk.pix) < chunkmin) or (len(chunk.pix) > chunkmax):
                discardablechunks.append(chunk)
        for discard in discardablechunks: chunks.remove(discard)
        self.n=len(chunks)
        # Reassign IDs (consolidate)
        for i,chunk in enumerate(chunks):
            chunk.id=i
        self.list=chunks



#    def sort_by_area(self):
#        print('I got here')
#        chunks=self.list
#        size  =[]
#        for chunk in chunks:
#            size.append(len(chunk.pix))
#        s=np.argsort(size)
#        self.list=chunks[s]

class SIMPLE_CLOUD:

    def __init__(self,pool):
        nmax        = 900      # maximum recursion depth
        self.pool   = pool     # assign pool of available pixels
        self.pix    = []       # pixels in object
        self.n      = 0        # size of cloud
        self.nmax   = nmax     # maximum size of cloud
        self.grow()

    def grow(self,idx=0): # idx specifies where in the pool to start the search
        i=self.pool[idx][0]
        j=self.pool[idx][1]
        if [i,j] in self.pool:
            self.pool.remove([i,j])      # Remove pixel from pool
            self.pix.append([i,j])       # ...and add it to cloud object
            self.n=self.n+1
        else:
            print("Error, elements should still be in pool.")

        if self.n<self.nmax: # Find neighbors
            Di=[-1,0,1] # search pattern x direction
            Dj=[-1,0,1] # search pattern y direction
            for di in Di:
                for dj in Dj:
                    if [i+di,j+dj] in self.pool:
                        self.grow(idx=self.pool.index([i+di,j+dj]))
        else:
           sys.exit("Recursion depth exceeded, use different method.")



if __name__ == "__main__":

    pass
