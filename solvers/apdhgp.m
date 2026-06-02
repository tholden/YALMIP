function output = apdhgp(interfacedata)

% Retrieve options
options = interfacedata.options;
internal_solver_tag = options.apdhgp.internalsolver;

if isempty(internal_solver_tag)
    error('APDHGP: No internal solver specified in options.apdhgp.internalsolver.');
end

% Check if the high precision library GEM is in the path
if exist('gem','class') == 8
    highPrecisionSupported = true;
else
    highPrecisionSupported = false;
end

precdigits = options.apdhgp.precdigits;
if precdigits > 15
    if ~highPrecisionSupported
        warning('APDHGP: The GEM library was not found in MATLAB''s path. Falling back to double precision (limited to 1e-15).');
        use_gem = false;
    else
        use_gem = true;
        % Set the GEM internal precision following the same logic as in iterative_refinement
        if gem.workingPrecision < precdigits + 20
            if options.verbose >= 1
                warning(['Precision of the GEM library is low (', num2str(gem.workingPrecision), ' digits), increasing it to ', num2str(precdigits + 20), ' digits.']);
            end
            gem.workingPrecision(precdigits + 20);
        end
    end
else
    use_gem = false;
end

% Set up the internal interfacedata structure to run the initial solver
internal_interfacedata = interfacedata;
internal_interfacedata.options.solver = internal_solver_tag;

% Look up the solver structure from available solvers
solvers = getavailablesolvers(0, options);
solverindex = find(strcmpi({solvers.tag}, internal_solver_tag));
if isempty(solverindex)
    temp_solvers = solvers;
    for i = 1:length(temp_solvers)
        if ~isempty(temp_solvers(i).version)
            temp_solvers(i).tag = [temp_solvers(i).tag '-' temp_solvers(i).version];
        end
    end
    solverindex = find(strcmpi({temp_solvers.tag}, internal_solver_tag));
end

if isempty(solverindex)
    error(['APDHGP: Internal solver ' internal_solver_tag ' not found or not available.']);
end

internal_solver = solvers(solverindex(1));
internal_interfacedata.solver = internal_solver;

% Run the internal solver
if options.verbose >= 1
    fprintf('APDHGP: Running initial solver %s...\n', internal_solver.tag);
end
internal_output = feval(internal_solver.call, internal_interfacedata);

if internal_output.problem ~= 0
    if options.verbose >= 1
        fprintf('APDHGP: Initial solver failed with problem status %d.\n', internal_output.problem);
    end
    output = internal_output;
    return;
end

% Extract primal and dual solutions from the initial solver's output
x_init = internal_output.Primal;
y_init_vector = internal_output.Dual;

% If the initial solver didn't return duals, initialize with zeros
if isempty(y_init_vector)
    y_init_vector = zeros(interfacedata.K.f + interfacedata.K.l, 1);
end

% Extract equality and inequality constraints from interfacedata
F_struc = interfacedata.F_struc;
K       = interfacedata.K;
c       = interfacedata.c;
lb      = interfacedata.lb;
ub      = interfacedata.ub;

if isempty(F_struc)
    Aeq = [];
    beq = [];
    Ain = [];
    bin = [];
else
    Aeq = -F_struc(1:1:K.f, 2:end);
    beq = F_struc(1:1:K.f, 1);        
    Ain = -F_struc(K.f+1:end, 2:end);
    bin = F_struc(K.f+1:end, 1);   
end

% Extract equality and inequality dual variables
% In Yalmip:
% - The first K.f elements of output.Dual correspond to equalities
% - The next K.l elements correspond to inequalities
y_eq = y_init_vector(1:K.f);
y_in = y_init_vector(K.f+1:end);

% In apdhgp_lp, constraints are stacked as [Ain; Aeq], so dual variables
% must be stacked as [y_in; y_eq]
y_init = [y_in; y_eq];

% Setup opts for high-precision PDHG polish solver
opts = [];
opts.maxIters = options.apdhgp.maxiter;
if isempty(options.apdhgp.tol)
    if use_gem
        opts.tol = 10^(-options.apdhgp.precdigits);
    else
        opts.tol = 1e-12; % Standard double maximum feasible tolerance
    end
else
    opts.tol = options.apdhgp.tol;
end

if use_gem
    opts.prec = @(x) gem(x);
else
    opts.prec = @(x) double(x);
end

% Run high-precision PDHG polish solver
solvertime = tic;
if options.verbose >= 1
    fprintf('APDHGP: Polishing solution using high-precision Adaptive PDHG...\n');
end

[sol_dd, out] = apdhgp_lp(c, Ain, bin, Aeq, beq, ub, lb, x_init, y_init, opts);
solvertime = toc(solvertime);

% Convert outputs back to standard doubles
x_opt = double(sol_dd);

% Retrieve scaled dual variables and convert them back to original scale
y_scaled_opt = out.y;
% Recover the left preconditioner from apdhgp_lp (stored in out.LeftPrecond)
LeftPrecond_dd = out.LeftPrecond;
y_opt = y_scaled_opt .* LeftPrecond_dd;
y_opt_double = double(y_opt);

% Separate inequality and equality duals
Nin = size(Ain, 1);
y_in_opt = y_opt_double(1:Nin);
y_eq_opt = y_opt_double(Nin+1:end);

% Map dual variables back to Yalmip's format [y_eq; y_in]
D_struc = [y_eq_opt; y_in_opt];

if out.iters >= opts.maxIters
    problem = 3; % Maximum iterations reached
else
    problem = 0; % Solved successfully
end

if options.verbose >= 1
    fprintf('APDHGP: Completed in %d iterations (problem status %d).\n', out.iters, problem);
end

% Standard interface output
output = createOutputStructure(x_opt(:), D_struc, [], problem, interfacedata.solver.tag, [], [], solvertime);
end
