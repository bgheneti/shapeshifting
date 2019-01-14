import numpy as np
import shapely.ops as ops
import shapely.geometry as geometry
import matplotlib.pyplot as plt
import shapely.affinity as affinity
from pypolypart import pypolypart
from scipy.signal import convolve2d

from pyclipper import MinkowskiSum, scale_from_clipper as from_clipper, scale_to_clipper as to_clipper

class Shape:
    def __init__(self, shapes):
        self.blocks = shapes
        self.polygon = ops.cascaded_union([s.polygon for s in shapes]) 
        self.subblocks = np.concatenate([s.subblocks for s in shapes])
        
    @staticmethod
    def rectangle(x, y, dx, dy):
        return geometry.Polygon([(x+i*dx,y+j*dy) for i,j in [(1,1),(-1,1),(-1,-1),(1,-1)]])
    
    @staticmethod    
    def plot_polygons(polygons):
        plt.figure(figsize=(12,12))
        plt.axis('equal')
        
        for p in polygons:
            plt.plot(*p.boundary.xy, linewidth=4)
            
        plt.show()

    def plot(self,union=False):
        self.plot_polygons([self.polygon] if union else [c.polygon for c in self.blocks])

    def radius(self):
        corners = np.array(self.polygon.convex_hull.boundary.xy).T
        return max([np.linalg.norm(c1-c2) for c1 in corners for c2 in corners])     

    def partition_around(self, shape_b=None, buffered=True, simplify=True):
            
        r = self.radius() if shape_b is None else shape_b.radius()
        dx = dy = 4*r
            
        polygon = self.polygon.buffer(r) if buffered else self.polygon
        polygon = polygon.simplify(0.05) if simplify else polygon
        (x),(y) = self.polygon.centroid.xy

        area = zip(*self.rectangle(x,y,dx,dy).boundary.xy)[:-1]
        hole = zip(*polygon.boundary.coords.xy)[:-1]
                
        return [geometry.Polygon(p) for p in pypolypart.polys_to_tris_and_hulls([area], [hole])["hulls"]]
    
    
    ############################################################
    # Manipulating Shapes as a discrete grid / Convolving Shapes 
    ############################################################
    
    def grid_coords(self):
        return np.array(self.subblocks*2-0.5, dtype='int')
    
    def grid_coords_to_shape(self, coords, buffered=True, margin=0):
        side = 1. if buffered else 0.5
        side += 2*margin
        return Shape([Block(c[0],c[1], width=side, height=side) for c in np.array(coords/2.0)])
    
    def convolve_grid_coords(self, shape_b):
        coords_a = self.grid_coords()
        coords_b = shape_b.grid_coords()
        
        min_a = np.min(coords_a, axis=0)
        min_b = np.min(coords_b, axis=0)
        
        max_a = np.max(coords_a, axis=0)
        max_b = np.max(coords_b, axis=0)

        grid_a = np.zeros(tuple(max_a-min_a+1))
        grid_b = np.zeros(tuple(max_b-min_b+1))
        
        grid_a[tuple((coords_a-min_a).T)] = 1
        grid_b[tuple((coords_b-min_b).T)] = 1
        
        coords = np.array(np.where(convolve2d(grid_a, grid_b)>0)).T
        
        coords += min_a + (min_b - min_a)

        return coords
    
    def on_grid(self):
        return np.all(self.subblocks%0.5==0.25)
    
    def c_space(self, shape_b, margin=0):              
        assert self.on_grid() and shape_b.on_grid()        
        return self.grid_coords_to_shape(self.convolve_grid_coords(shape_b),margin=margin,buffered=True)
    
    ########################
    #Calc Hulls for Rotation
    ########################
    
    def bounds(self, polygon):
            return np.array(list(reversed((zip(*polygon.boundary.xy)))))
        
    def polygon_rotate(self, polygon, theta):
            return affinity.rotate(polygon,theta,geometry.Point(0,0))
        
    def polygon_rotate_range(self, polygon, theta1, theta2, num):
        return ops.cascaded_union([self.polygon_rotate(polygon, t) for t in np.linspace(theta1, theta2, num)])
    
    def bounds_rotate(self, polygon, theta1, theta2):
        return self.bounds(self.polygon_rotate_range(polygon, theta1, theta2, 1 if theta1==theta2 else 200))
        
    def loop(self, x):
        return np.vstack((x, x[0]))
    
    def split_poly_boundaries(self, p):
        b = p.boundary
        
        if b.type == 'MultiLineString':
            return [zip(*l.coords.xy)[:-1] for l in b]
        else:
            return [zip(*b.coords.xy)[:-1]]
        
    def calc_hulls(self, msum=None, area=None, buffer=0.2):
        if area is None:
            area = self.rectangle(0,0,8,8)
        polys = self.split_poly_boundaries(area)

        if msum is None:
            holes = []
            polys = [p[::-1] for p in polys]
        else:
            holes = self.split_poly_boundaries(msum)
            
        return [geometry.Polygon(p) for p in pypolypart.polys_to_tris_and_hulls(polys, holes)["hulls"]]
    
    def msum(self, bounds_a, bounds_b):
        return geometry.Polygon(from_clipper(MinkowskiSum(to_clipper(bounds_b*-1,100), to_clipper(bounds_a,100), True),100)[0])

    def buffered_msum(self, msum, buffer=0.2):
        return msum.simplify(0.01).buffer(buffer,cap_style=3).simplify(0.1)
    
    def free_rectangle(self, msum, x, dx=0.25, dy=.25):
        return self.rectangle(x[0], x[1], dx, dy).difference(msum)
        
    def c_space_rotate(self, shape_b, x0=None, xN=None):
        angle_ranges = [[0,0],[90,90],[180,180], [270,270], [0,360]]
        bounds_a = self.bounds(self.polygon)
        bounds_B = {(a,b): self.bounds_rotate(shape_b.polygon, a, b) for a,b in angle_ranges}

        msums = {angles: self.msum(bounds_a,bounds_b) for angles,bounds_b in bounds_B.items()}
        buffered_msums = {angles: self.buffered_msum(msum) for angles,msum in msums.items()}
        
        angles0 = (x0[2], x0[2])
        anglesN = (xN[2], xN[2])
        
        if x0 is not None:
            buffered_msums[angles0] = buffered_msums[angles0].difference(self.free_rectangle(msums[angles0], x0))
        if xN is not None:
            buffered_msums[anglesN] = buffered_msums[anglesN].difference(self.free_rectangle(msums[anglesN], xN))
            print buffered_msums[anglesN]
            
        hulls = {angle_range: self.calc_hulls(msum) for angle_range,msum in buffered_msums.items()}

        #plt.plot(*zip(*bounds_B.values()[0]))
        #plt.plot(*zip(*bounds_a))
        return hulls
    
    def c_space_rotate_latch(self, shape_b, x, y, theta):
        msum1 = self.msum(self.bounds(self.polygon), self.bounds_rotate(shape_b.polygon, theta, theta))
        msum2 = self.buffered_msum(msum1)
        p = self.rectangle(x, y, 0.25, 0.25).difference(msum1).intersection(msum2)
        
        if p.area==0:
            return []
        else:
            print self.calc_hulls(area=p, buffer=0)
            return self.calc_hulls(area=p, buffer=0)

    def plot_msums(self, msums):
        plt.figure()
        for msum in msums.values():
            plt.plot(*zip(*self.loop(msum)))
        plt.show()
        
class Block(Shape):
    def __init__(self, x, y, theta=0, width=1, height=0.5):
        self.x      = float(x)
        self.y      = float(y)
        self.height = float(height)
        self.width  = float(width)
        self.theta  = float(theta)

        self.blocks = [self]
        self.polygon = self.rectangle(x, y, self.width/2, self.height/2)
        self.polygon = affinity.rotate(self.polygon, float(theta))
        
        center = geometry.point.Point(x,y)
        
        self.subblocks = [ self.rectangle(self.x-self.width/4, self.y, self.width/4, self.height/4),
                           self.rectangle(self.x+self.width/4, self.y, self.width/4, self.height/4) ]
        
        self.subblocks = np.array([affinity.rotate(b, self.theta,origin=center).centroid.xy for b in self.subblocks])\
                           .reshape(2,2)

def plot_hulls(hulls, path=None, text=True, figure=True, color='black'):
    if figure:
        plt.figure(figsize=(10,10))#,facecolor='black')

    plt.axis('equal')
    for i, hull in enumerate(hulls):
        X,Y = hull.boundary.xy
        [x],[y] = hull.centroid.xy
        plt.plot(X,Y,'-',linewidth=4)
        ax = plt.gca()
        #ax.set_facecolor((0,0,0))
        if text:
            plt.text(x,y, str(i), fontsize=20)#, color='white')
        
    if path is not None:
        plt.plot(*path, linewidth=3, color=color)
    
    if figure:
        plt.show()
