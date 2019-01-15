from shape_boat import ShapeBoat_spline
from pydrake.all import SolutionResult
from multiboat_trajectory_optimization.trajectory_planner import BoatConfigurationPlanning
import numpy as np

def shapeshift_trajectory(boat_shape, obstacle_shape, xy0, xyN, N=15, margin=0.0, boat_type=ShapeBoat_spline, opt_angle=True):
    result2 = None
    
    x0 = np.zeros((1, boat_type.num_states))
    xN = np.zeros((1, boat_type.num_states))
    x0[0, :3] = xy0
    xN[0, :3] = xyN
    boat = boat_type(boat_shape, obstacle_shape, margin=margin)
    boat.set_end_points(x0, xN)
    
    planner = BoatConfigurationPlanning(boat)
    S_knots, U, in_hull, on_edge, mp, result1, solve_time = planner.compute_spline_trajectory(0., 10, x0, xN, input_position_cost=True, slack=0, N=N)
    success = result1==0
        
    if opt_angle and success:
        S_knots, U_knots, in_hull, on_edge, mp, result2, solve_time = planner.compute_spline_trajectory(0., 10, x0, xN, input_angle_cost=True, slack=0, N=N, in_hull=in_hull, on_edge=on_edge, states_initialization=S_knots, fix_initialization_inds=[0,1])
        success = result2==0
               
    S, U = knots_to_trajectory(S_knots, U_knots, 10) if success else None
    
    return {'boat':          boat, 
            'S_knots':       S_knots,
            'U_knots':       U_knots,
            'S' :            S,
            'U':             U, 
            'result_spline': result1,
            'result_angle':  result2,
            'success':       success,
            'metrics':       metrics(U) if success else None
            }


def knots_to_trajectory(S, U, dN, order=3):
    S_sample = np.zeros((S.shape[0],S.shape[1]+(order-1)+(order-2),S.shape[2]))
    U_sample = np.zeros((U.shape[0],U.shape[1]+2,U.shape[2]))
    S_sample[:,order-1:-order] = S[:,1:-1]
    S_sample[:,:order-1] = S[:,0,:]
    S_sample[:,-order:] = S[:,-1,:]
    U_sample[:,:-2] = U
        
    shape = S_sample.shape
    num_knots = shape[1]
    
    #number of knots
    N = dN*(num_knots-3)+1
    
    S_new = np.zeros((shape[0],N,shape[2]))
    U_new = np.zeros((shape[0],N,3))

    M = 0.5 * np.array([[1, 1, 0],[-2, 2, 0],[1, -2, 1]])
                    
    for b in range(shape[0]):
        for x in range(N): 
            knot_ind = int(x/dN)
            knot_fraction = x/float(dN)-knot_ind
            p = S_sample[b,knot_ind:knot_ind+3,:2]
            B = np.array([1, knot_fraction, knot_fraction**2]).dot(M)
            dB_dt = np.array([0, 1, 2*knot_fraction]).dot(M)/dN
            d2B_dt2 = np.array([0, 0, 2]).dot(M)/dN
                        
            S_new[b,x,:2]  = B.dot(p)
            S_new[b,x,3:5] = dB_dt.dot(p)
            U_new[b,x,:2]  = d2B_dt2.dot(p)           
            U_new[b,x,2]   = U_sample[b,knot_ind,2] if knot_fraction<0.5 else U_sample[b,knot_ind,3]
            
    return S_new, U_new

def metrics(U):
    return analytics({'U_position_cost': U_position_cost(U), 'U_angle_cost': U_angle_cost(U)})

def analytics(costs):
    analytic_ops = [('_sum',np.sum),('_std',np.std),('_avg',np.average), ('',lambda x: x)]
    return {k+fk:fv(v) for k,v in costs.items() for fk,fv in analytic_ops}
     
def U_position_cost(U):
    return np.sum(U[0,:,:2]**2,axis=1)

def U_angle_cost(U):
    return U[0,:,2]**2