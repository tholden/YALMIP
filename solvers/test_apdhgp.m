% Scratch script to test the apdhgp solver in YALMIP.
% Make sure YALMIP and DoubleDouble are in your path.
% Run this in MATLAB.

% Set up Gurobi path first
GurobiPath      = "C:\gurobi1301\win64\";
Path = string( getenv( "path" ) );
if ~contains( Path, GurobiPath + "bin;" )
    setenv( "path", GurobiPath + "bin" + pathsep + Path );
end
addpath( GurobiPath + "matlab" );

% Add necessary directories to path (assuming parent is OneDrive/YieldCurve)
addpath('c:\Users\Tom\OneDrive\YieldCurve\DoubleDouble');
addpath('c:\Users\Tom\OneDrive\YieldCurve\YALMIP');
addpath('c:\Users\Tom\OneDrive\YieldCurve\YALMIP\solvers');
addpath('c:\Users\Tom\OneDrive\YieldCurve\YALMIP\extras');
addpath('c:\Users\Tom\OneDrive\YieldCurve\YALMIP\modules\global');

% Create test LP
x = sdpvar(2, 1);
Constraints = [x >= 0, x(1) + x(2) <= 1, 2*x(1) + x(2) >= 0.5];
Objective = -x(1) - 2*x(2);

% Set options for APDHGP with gurobi as internal solver
opts = sdpsettings('solver', 'apdhgp', 'apdhgp.internalsolver', 'gurobi', 'verbose', 1);

% Solve
sol = optimize(Constraints, Objective, opts);

% Display results
if sol.problem == 0
    fprintf('Success! Optimal x:\n');
    disp(value(x));
    fprintf('Optimal objective value: %f\n', value(Objective));
else
    fprintf('Failed with problem code: %d, info: %s\n', sol.problem, sol.info);
end
