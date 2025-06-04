from shutil import which
import pulp

COIN_CMD_PATH = which("cbc") or "/opt/homebrew/opt/cbc/bin/cbc"

def solve_optimization_problem(prob, time_limit=60, gap_rel=0.01, warm_start=True):
    """
    Solve a PuLP optimization problem using the CBC solver with optimized parameters.
    
    Args:
        prob (pulp.LpProblem): The PuLP optimization problem to solve
        time_limit (int): Time limit in seconds
        gap_rel (float): Relative optimality gap
        warm_start (bool): Whether to use warm start
        
    Returns:
        tuple: (status, objective_value) - The solution status and objective value
    """
    try:
        # Convert gap_rel to string to avoid type issues
        gap_rel_str = str(gap_rel)
        
        # Configure the solver with optimized parameters
        solver = pulp.COIN_CMD(
            path=COIN_CMD_PATH,
            timeLimit=time_limit,
            warmStart=warm_start,
            options=[
                'allowableGap', gap_rel_str,
                'maxSolutions', '1',
                'maxNodes', '10000'
                
            ]
        )
        
        # Solve the problem
        status = prob.solve(solver)
        objective_value = pulp.value(prob.objective)
        
        return status, objective_value
    
    except Exception as e:
        print(f"Error solving optimization problem: {str(e)}")
        return None, None
