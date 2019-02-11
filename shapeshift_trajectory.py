from shape_boat import ShapeBoat_spline, ShapeBoat
from pydrake.all import SolutionResult
from tabulate import tabulate
from multiboat_trajectory_optimization.trajectory_planner import BoatConfigurationPlanning
import numpy as np
import pickle
import copy

def shapeshift_trajectory(boat_shape, obstacle_shape, xy0, xyN, N=15, boat_type_init=None, boat_type=ShapeBoat_spline, plot=True, feasible_U=False):
    result=result_init=boat=boat_init=S=S_init=S_knots=U=U_init=U_knots=in_hull=solve_time=solve_time_init=None
    success=True
    opt_angle=True
    
    x0 = np.zeros((1, boat_type.num_states))
    xN = np.zeros((1, boat_type.num_states))
    x0[0, :3] = xy0
    xN[0, :3] = xyN
    
    if boat_type_init is not None:
        boat_init = boat_type_init(boat_shape, obstacle_shape)
        planner = BoatConfigurationPlanning(boat_init)
        boat_init.set_end_points(x0, xN)
        
        S_init, U_init, in_hull, mp, result_init, solve_time_init = planner.compute_mip_trajectory(x0, xN, N=N)
        success = result_init==0
        opt_angle=False
    
    if success:
        boat = boat_type(boat_shape, obstacle_shape, search=True)
        planner = BoatConfigurationPlanning(boat)
        boat.set_end_points(x0, xN)
        
        S, U, in_hull, mp, result, solve_time_final = planner.compute_mip_trajectory(x0, xN, S_init, U_init, in_hull, N=N, opt_angle=opt_angle)
        success = result==0
        
        if boat_type is ShapeBoat_spline:
            S_knots = S
            U_knots = U

            if feasible_U and success:
                S, U = boat_type.knots_to_feasible_trajectory(S_knots, U_knots)
            elif success:
                S, U = boat_type.knots_to_trajectory(S_knots, U_knots, dT_target=1)
            else:
                S, U = (None, None)
            
    if success and plot:  
        boat.plot_hulls(S, S_knots, all_hulls=True, text=False)
        boat.plot_hulls(S, S_knots, in_hull)
                
    return {'boat_init':        boat_init,
            'boat':             boat,
            'S_knots':          S_knots,
            'U_knots':          U_knots,
            'S_init':           S_init,
            'S' :               S,
            'U':                U, 
            'in_hull':          in_hull,
            'result_init':      result_init,
            'result':           result,
            'success':          success,
            'metrics':          metrics(U, solve_time_init, solve_time_final) if success else None    
            }


def metrics(U, solve_time_init, solve_time_final):
    x = analytics({'pos_cost': U_position_cost(U), 'ang_cost': U_angle_cost(U)}) 
    x.update({'solve_time_init':solve_time_init, 'solve_time_final': solve_time_final, 'solve_time': total_solve_time(solve_time_init, solve_time_final)})
    return x

def total_solve_time(solve_time_init, solve_time_final):
    return solve_time_final + (solve_time_init if solve_time_init is not None else 0.)

def analytics(costs):
    analytic_ops = [('_avg', np.average)]
    return {k+fk:fv(v) for k,v in costs.items() for fk,fv in analytic_ops}
     
def U_position_cost(U):
    return 100*np.sum(U[0,:,:2]**2,axis=1)

def U_angle_cost(U):
    return U[0,:,2]**2
        
def experiment(test, boat_init, boat):
    return shapeshift_trajectory(*test,boat_type=boat, boat_type_init=boat_init, N=10)

def experiments(tests, boats):
    return {boat:{test: experiment(tests[test], *boats[boat]) for test in tests} for boat in boats}

def write_results(results, label):
    with open('results/MIQP_'+label+'.pickle', 'wb') as f:
        pickle.dump(results, f)
        
    with open('results/MIQP_'+label+'_basic.pickle', 'wb') as f:
        pickle.dump({x1:{y1:{'S':y2['S'], 'U':y2['U']} for y1,y2 in x2.items()} for x1,x2 in results.items()}, f)

    for k,v in results.items():
        with open('results/MIQP_'+label+'_'+k+'.csv', 'wb') as f:
            f.write(results_table(v))
        
def bold_mins(results):
    results = {x1:{y1:{'metrics': dict(y2['metrics'])} for y1,y2 in x2.items()} for x1,x2 in results.items()}

    for metric in results.values()[0].values()[0]['metrics']:
        if metric!='solve_time_init':
            for experiment in results.values()[0]:
                val = min([results[i][experiment]['metrics'][metric] for i in results])
                inds = [i for i in results if results[i][experiment]['metrics'][metric]==val]
                if len(inds)<3:
                    for ind in inds:
                        results[ind][experiment]['metrics'][metric] = '\\textbf{%s}' % "{0:.3f}".format(results[ind][experiment]['metrics'][metric])
                                                                      
    return results

def rows_to_latex(rows):

    parse = lambda x: '& '+ ('-' if x is None else ("{0:.3f}".format(round(x,2)) if isinstance(x, float) else x.replace('_',' ')))
    return [[parse(v) for v in row[:-1]]+[parse(row[-1])+' \\\\'] for row in rows]
        
def results_table(experiments, latex=False):
    metric_cols = ['solve_time_init','solve_time_final','solve_time','pos_cost_avg','ang_cost_avg']
    col_names = ['experiment'] + metric_cols
    
    vals = lambda k,v: [k]+[v['metrics'][m] for m in metric_cols]
    row_vals = [vals(*x) for x in sorted(experiments.items(),key=lambda y: y[0])]
    
    if latex:
        row_vals = rows_to_latex(row_vals)
    
    return tabulate(row_vals, headers=col_names, floatfmt=".3f")

def print_results_tables(results, latex=False):    
    if latex:
        results = bold_mins(results)
    
    for k,v in results.items():
        print k
        print results_table(v, latex=latex)