from shutil import which
import os
import pulp

from src.service.helpers.constants import logger

_HOMEBREW_CBC = "/opt/homebrew/opt/cbc/bin/cbc"


def _cbc_solver(time_limit=60, gap_rel=0.01, warm_start=True):
    """Return a CBC solver, preferring a system binary then PuLP's bundled CBC.

    ``msg=False`` keeps CBC off stdout so stdio MCP servers stay valid.
    """
    options = [
        "allowableGap",
        str(gap_rel),
        "maxSolutions",
        "1",
        "maxNodes",
        "10000",
    ]
    kwargs = dict(
        timeLimit=time_limit,
        warmStart=warm_start,
        options=options,
        msg=False,
    )
    system_cbc = which("cbc")
    if system_cbc:
        return pulp.COIN_CMD(path=system_cbc, **kwargs)
    if os.path.isfile(_HOMEBREW_CBC) and os.access(_HOMEBREW_CBC, os.X_OK):
        return pulp.COIN_CMD(path=_HOMEBREW_CBC, **kwargs)
    return pulp.PULP_CBC_CMD(**kwargs)


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
        solver = _cbc_solver(time_limit=time_limit, gap_rel=gap_rel, warm_start=warm_start)
        status = prob.solve(solver)
        objective_value = pulp.value(prob.objective)
        return status, objective_value
    except Exception as e:
        logger.error(f"Error solving optimization problem: {str(e)}")
        return None, None
