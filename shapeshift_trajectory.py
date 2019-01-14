from shape_boat import ShapeBoat_spline
from multiboat_trajectory_optimization.trajectory_planner import BoatConfigurationPlanning
import numpy as np

def shapeshift_trajectory(boat_shape, obstacle_shape, xy0, xyN, N=10, margin=0.0, boat_type=ShapeBoat_spline, opt_angle=False):
    x0 = np.zeros((1, boat_type.num_states))
    xN = np.zeros((1, boat_type.num_states))
    x0[0, :3] = xy0
    xN[0, :3] = xyN
    boat = boat_type(boat_shape, obstacle_shape, margin=margin)
    boat.set_end_points(x0, xN)
    planner = BoatConfigurationPlanning(boat)
    boats_S, boats_U, in_hull, on_edge, mp, result, solve_time = planner.compute_spline_trajectory(0., 10, x0, xN, input_position_cost=True, slack=0, N=N)

    if opt_angle:
        boats_S, boats_U, in_hull, on_edge, mp, result, solve_time = planner.compute_spline_trajectory(0., 10, x0, xN, input_angle_cost=True, slack=0, N=N, in_hull=in_hull, on_edge=on_edge, states_initialization=boats_S, fix_initialization_inds=[0,1])
        
    boats_S_new = knots_to_trajectory(boats_S, 10)
    
    return {'boat': boat, 'S_knots': boats_S, 'S' : boats_S_new, 'U': boats_U, 'mp': mp}

def knots_to_trajectory(boats_S, dN, order=3):
    boats_S_sample = np.zeros((boats_S.shape[0],boats_S.shape[1]+2*(order-1),boats_S.shape[2]))
    boats_S_sample[:,order:-order] = boats_S[:,1:-1]
    boats_S_sample[:,:order] = boats_S[:,0,:]
    boats_S_sample[:,-order:] = boats_S[:,-1,:]
        
    shape = boats_S_sample.shape
    num_knots = shape[1]
    
    #number of knots
    N = dN*(num_knots-3)+1
    
    new_boats_S = np.zeros((shape[0],N,shape[2]))
    M = 0.5 * np.array([[1, 1, 0],[-2, 2, 0],[1, -2, 1]])
                    
    for b in range(shape[0]):
        for x in range(0,N): 
            knot_ind = int(x/dN)
            knot_fraction = x/float(dN)-knot_ind
            p = boats_S_sample[b,knot_ind:knot_ind+3,:2]
            B = np.array([1, knot_fraction, knot_fraction**2]).dot(M)
            #print B
            dB_dt = np.array([0, 1, 2*knot_fraction]).dot(M)/dN
            
            new_boats_S[b,x,:2] = B.dot(p)
            new_boats_S[b,x,3:5] =dB_dt.dot(p)
            
    return new_boats_S
