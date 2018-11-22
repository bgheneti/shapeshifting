import collections
import numpy as np
from graph_tool import Graph, search, generation, draw

class Visitor(search.BFSVisitor):
    def __init__(self, pred, dist):
        self.pred = pred
        self.dist = dist

    def tree_edge(self, e):
        self.pred[e.target()] = int(e.source())
        self.dist[e.target()] = self.dist[e.source()] + 1

class HullGraph(Graph):
    def __init__(g, hulls):
        def sorted_edge(edge):
            return tuple(sorted(sorted(edge, key=lambda x: x[1]), key=lambda x: x[0]))
        
        def edge_equation(x1, x2):
            normal = np.array([-(x2[1]-x1[1]),(x2[0]-x1[0])])
            return (x1.dot(normal)/np.linalg.norm(normal), normal/np.linalg.norm(normal))

        def extract_edges(hulls):
            e2h = collections.defaultdict(set)
            
            v_poly = g.new_vertex_property("object")
            v_poly_eq = g.new_vertex_property("object")
            
            for h, hull in enumerate(hulls):
                vertex = g.add_vertex()
                
                points = zip(*hull.boundary.xy)
                b = np.zeros((len(points)-1))
                normal = np.zeros((len(points)-1, 2))
                
                v_poly[vertex] = hull
                v_poly_eq[vertex] = {'b': b, 'A': normal}                
                
                for i, edge in enumerate(zip(points[:-1],points[1:])):
                    e2h[sorted_edge(edge)].add(h)
                    b[i], normal[i] = edge_equation(*np.array(edge))
                    
            return e2h, v_poly, v_poly_eq
        
        def add_edges_from_hull_dict(e2h):
            i2e = {}
            e_points = g.new_edge_property("object")
            for i, ((h1,h2),e) in enumerate([(tuple(v),e) for e,v in e2h.items() if len(v)==2]):
                i2e[i] = g.add_edge(g.vertex(h1),g.vertex(h2))
                e_points[i2e[i]] = np.array(e)
                
            return i2e, e_points
        
        super(HullGraph, g).__init__(directed=False)
                
        e2h, g.vertex_properties["polygon"], g.vertex_properties["polygon_eq"] = extract_edges(hulls)
        
        g.edge_dict, g.edge_properties["points"] = add_edges_from_hull_dict(e2h)
            
       
    def line_graph(g):
        lg, lg_vmap = generation.line_graph(g)

        g_vmap = g.new_edge_property("int")
        for v in lg_vmap:
            g_vmap[g.edge_dict[lg_vmap[v]]] = v
        g.edge_properties["vmap"] = g_vmap
        return lg
    
    def point_vertex(g, point):
        for v, hull in enumerate(g.vertices()):
            if g.vertex_properties['polygon'][hull].contains(point):
                return v
                return None

    def bfs_search(g, v_ind):
        pred = g.vertex_properties['pred'] = g.new_vertex_property("int", -1)
        dist = g.vertex_properties['dist'] = g.new_vertex_property("int64_t")
        
        bfs = search.bfs_search(g, g.vertex(v_ind), Visitor(pred, dist))
        
    def vertex_path(g, v0, vN):
        g.bfs_search(v0)
        path = [vN]
        v = vN
        
        while v!=v0:
            v = g.vertex_properties['pred'][v]
            path.append(v)
            
        return list(reversed(path))
    
    def point_path(g, p0, pN):
        v0 = g.point_vertex(p0)
        vN = g.point_vertex(pN)
        assert(v0 is not None and vN is not None)
        
        return g.vertex_path(v0, vN)
                    
    def draw_graph(g):
        draw.graph_draw(g, vertex_text=g.vertex_index, edge_text=g.edge_index)
