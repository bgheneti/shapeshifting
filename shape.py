import numpy as np
import shapely.ops as ops
import shapely.geometry as geometry
import matplotlib.pyplot as plt
import shapely.affinity as affinity
from pypolypart import pypolypart

from pyclipper import MinkowskiSum, scale_from_clipper as from_clipper, scale_to_clipper as to_clipper

class Shape:
    def __init__(self, shapes):
        self.blocks = np.concatenate([s.blocks for s in shapes])
        self.polygon = ops.cascaded_union([s.polygon for s in shapes]) 
        self.subblocks = np.concatenate([s.subblocks for s in shapes])
        
    @staticmethod
    def rectangle(x, y, dx, dy):
        return geometry.Polygon([(x+i*dx,y+j*dy) for i,j in [(1,1),(-1,1),(-1,-1),(1,-1)]])
    
    @staticmethod    
    def plot_polygons(polygons):
        plt.figure(figsize=(7,7))
        plt.axis('equal')
        
        for p in polygons:
            plt.plot(*p.boundary.xy, linewidth=4)
            
        plt.show()

    def plot(self,union=False):
        self.plot_polygons([self.polygon] if union else [c.polygon for c in self.blocks])

    def radius(self):
        corners = np.array(self.polygon.convex_hull.boundary.xy).T
        return max([np.linalg.norm(c1-c2) for c1 in corners for c2 in corners])     
    
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
        
    def calc_hulls(self, msum=None, area=None, buffer=0.25, radius=4.):
        if area is None:
            area = self.rectangle(0,0,radius,radius)
        polys = self.split_poly_boundaries(area)

        if msum is None:
            holes = []
            polys = [p[::-1] for p in polys]
        else:
            holes = self.split_poly_boundaries(msum)
            
        return [geometry.Polygon(p) for p in pypolypart.polys_to_tris_and_hulls(polys, holes)["hulls"]]
    
    def msum(self, bounds_a, bounds_b):
        return geometry.Polygon(from_clipper(MinkowskiSum(to_clipper(bounds_b*-1,100), to_clipper(bounds_a,100), True),100)[0])

    def buffered_msum(self, msum, buffer=0.3):
        return msum.simplify(0.01).buffer(buffer,cap_style=2,join_style=2).simplify(0.1)
    
    def free_rectangle(self, msum, x, dx, dy):
        return self.rectangle(x[0], x[1], dx, dy).difference(msum)
    
    def trim_buffer(self, msums, buffered_msums, x, dx=.3, dy=.3):
        if x is not None:
            angles = (x[2], x[2])
            buffered_msums[(angles)] = buffered_msums[angles].difference(self.free_rectangle(msums[angles], x, dx, dy))
        
    def c_space_rotate(self, shape_b, x0=None, xN=None):
        angle_ranges = [[0,0],[90,90],[180,180], [270,270], [0,360]]
        bounds_a = self.bounds(self.polygon)
        bounds_B = {(a,b): self.bounds_rotate(shape_b.polygon, a, b) for a,b in angle_ranges}
        radius = self.radius()+shape_b.radius()

        msums = {angles: self.msum(bounds_a,bounds_b) for angles,bounds_b in bounds_B.items()}
        buffered_msums = {angles: self.buffered_msum(msum) for angles,msum in msums.items()}
        
        self.trim_buffer(msums, buffered_msums, x0)
        self.trim_buffer(msums, buffered_msums, xN)

        hulls = {angle_range: self.calc_hulls(msum, radius=radius) for angle_range,msum in buffered_msums.items()}

        return msums, hulls

    def plot_msums(self, msums):
        plt.figure()
        for msum in msums.values():
            plt.plot(*zip(*self.loop(msum.boundary)))
        plt.show()
        
class Block(Shape):
    def __init__(self, x, y, theta=0, width=1, height=0.5, scale=1):
        x*= scale
        y*= scale
        width *= scale
        height *= scale
        
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
        plt.figure(figsize=(10,10))

    plt.axis('equal')
    for i, hull in enumerate(hulls):
        X,Y = hull.boundary.xy
        [x],[y] = hull.centroid.xy
        plt.plot(X,Y,'-',linewidth=4)
        ax = plt.gca()
        #ax.set_facecolor((0,0,0))
        if text:
            plt.text(x,y, str(i), fontsize=20)
        
    if path is not None:
        plt.plot(*path, linewidth=3, color=color)
    
    if figure:
        plt.show()
