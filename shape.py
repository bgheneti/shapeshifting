import numpy as np
import shapely.ops as ops
import shapely.geometry as geometry
import matplotlib.pyplot as plt
import shapely.affinity as affinity
from pypolypart import pypolypart
from scipy.signal import convolve2d

class Shape:
    def __init__(self, shapes):
        self.blocks = shapes
        self.polygon = ops.cascaded_union([s.polygon for s in shapes]) 
        self.subblocks = np.concatenate([s.subblocks for s in shapes])
        
    def rectangle(self, x, y, dx, dy):
        return geometry.Polygon([(x+i*dx,y+j*dy) for i,j in [(1,1),(-1,1),(-1,-1),(1,-1)]])
    
    def plot_polygons(self, polygons):
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
        return self.grid_coords_to_shape(self.convolve_grid_coords(shape_b),margin=margin)      
        
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
