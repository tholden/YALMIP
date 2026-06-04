function [ sol, out ] = apdhgp_lp( f, Ain, bin, Aeq, beq, ub, lb, x_init, y_init, opts )

    %% Record problem dimensions
    Nvar = size(Ain,2);
    if isempty(Ain)
        Nvar = size(Aeq,2);
    end
    Nin = size(Ain,1);

    %%  Create the linear operator to enforce constraints
    M = sparse([Ain ; Aeq]);
    b = [bin ; beq];

    if isempty(ub)
        ub = inf(Nvar,1);
    end
    if isempty(lb)
        lb = -inf(Nvar,1);
    end

    %% Precondition the problem in double precision (Ruiz Equilibration)
    M_double = double(M);
    [m, n] = size(M_double);

    LeftPrecond = ones(m, 1);
    RightPrecond = ones(n, 1);

    % Ruiz equilibration scales rows and columns to have infinity norms close to 1.
    M_temp = M_double;
    for k = 1:20
        % Row scaling factors
        r = full(max(abs(M_temp), [], 2));
        r(r == 0) = 1;
        r_scale = sqrt(r);

        % Column scaling factors
        c = full(max(abs(M_temp), [], 1).');
        c(c == 0) = 1;
        c_scale = sqrt(c);

        % Scale the matrix
        M_temp = sparse(1:m, 1:m, 1./r_scale, m, m) * M_temp * sparse(1:n, 1:n, 1./c_scale, n, n);

        % Accumulate row and column scaling factors
        LeftPrecond = LeftPrecond ./ r_scale;
        RightPrecond = RightPrecond ./ c_scale;
    end

    % L2 equilibration to further cluster singular values (helpful for first-order methods)
    for k = 1:10
        % Row scaling factors (L2 norm of each row)
        r = full(sqrt(sum(M_temp.*M_temp, 2)));
        r(r == 0) = 1;
        r_scale = sqrt(r);

        % Column scaling factors (L2 norm of each column)
        c = full(sqrt(sum(M_temp.*M_temp, 1)).');
        c(c == 0) = 1;
        c_scale = sqrt(c);

        % Scale the matrix
        M_temp = sparse(1:m, 1:m, 1./r_scale, m, m) * M_temp * sparse(1:n, 1:n, 1./c_scale, n, n);

        % Accumulate row and column scaling factors
        LeftPrecond = LeftPrecond ./ r_scale;
        RightPrecond = RightPrecond ./ c_scale;
    end

    LeftPrecond(isnan(LeftPrecond) | isinf(LeftPrecond) | LeftPrecond == 0) = 1;
    RightPrecond(isnan(RightPrecond) | isinf(RightPrecond) | RightPrecond == 0) = 1;

    % Apply preconditioning in double precision
    LeftPrecond_sparse = sparse(1:m, 1:m, LeftPrecond, m, m);
    RightPrecond_sparse = sparse(1:n, 1:n, RightPrecond, n, n);
    M_scaled = LeftPrecond_sparse * M * RightPrecond_sparse;
    b_scaled = LeftPrecond .* b;
    f_scaled = f .* RightPrecond;

    % Global scaling for objective and right-hand side to normalize primal/dual variables
    c_f = 1 / max(1, norm(f_scaled, 2));
    c_b = 1 / max(1, norm(b_scaled, 2));

    f_scaled = f_scaled * c_f;
    b_scaled = b_scaled * c_b;

    LeftPrecond = LeftPrecond / c_f;
    RightPrecond = RightPrecond / c_b;

    % Convert inputs to DoubleDouble precision
    f_dd = DoubleDouble(f_scaled);
    LeftPrecond_dd = DoubleDouble(LeftPrecond);
    RightPrecond_dd = DoubleDouble(RightPrecond);

    bin_dd = DoubleDouble(b_scaled(1:Nin));
    beq_dd = DoubleDouble(b_scaled(Nin+1:end));

    % Scale bounds
    ub_scaled_double = ub;
    finite_ub = isfinite(ub);
    ub_scaled_double(finite_ub) = ub(finite_ub) ./ RightPrecond(finite_ub);
    ub_scaled = DoubleDouble(ub_scaled_double);

    lb_scaled_double = lb;
    finite_lb = isfinite(lb);
    lb_scaled_double(finite_lb) = lb(finite_lb) ./ RightPrecond(finite_lb);
    lb_scaled = DoubleDouble(lb_scaled_double);

    fProx = @(x,tau) fProx_helper(x, tau, f_dd, ub_scaled, lb_scaled);
    gProx = @(y,sigma) gProx_helper(y, sigma, Nin, m, bin_dd, beq_dd);
    A = @(x) A_helper(x, M_scaled);
    At = @(y) At_helper(y, M_scaled);

    % Scale the initial primal and dual guesses
    x0_dd = DoubleDouble(x_init) ./ RightPrecond_dd;
    y0_dd = DoubleDouble(y_init) ./ LeftPrecond_dd;

    %% Call the adaptive PDHG high-precision solver
    if ~exist('opts','var') || isempty(opts)
        opts = struct('maxIters', 10000, 'tol', 1e-14, 'verbose', 0);
    end

    % Calculate L as 2 * the reciprocal of the spectral radius of A.'A in double precision
    if isfield( opts, 'L' ) && opts.L > 0
        LocalL = opts.L;
    elseif coder.target( 'MATLAB' )
        LocalL = 2 / eigs( M_scaled.' * M_scaled, 1 );
    else
        % Power iteration for code generation (MEX target)
        NVars = size( M_scaled, 2 );
        XRand = randn( NVars, 1 );
        for KIter = 1 : 10
            XRand = M_scaled.' * ( M_scaled * XRand );
            NormX = norm( XRand );
            if NormX > 0
                XRand = XRand / NormX;
            else
                break;
            end
        end
        SpecRadius = norm( M_scaled.' * ( M_scaled * XRand ) ) / norm( XRand );
        if SpecRadius <= 0
            SpecRadius = 1;
        end
        LocalL = 2 / SpecRadius;
    end

    if isfield( opts, 'tau' ) && opts.tau > 0
        TauVal = opts.tau;
    else
        TauVal = sqrt( LocalL );
    end
    if isfield( opts, 'sigma' ) && opts.sigma > 0
        SigmaVal = opts.sigma;
    else
        SigmaVal = LocalL / TauVal;
    end

    % Initialize local_opts structure with all expected fields to satisfy MATLAB Coder's static typing
    local_opts = struct();
    local_opts.maxIters = opts.maxIters;
    local_opts.tol = opts.tol;
    if isfield(opts, 'verbose')
        local_opts.verbose = double(opts.verbose);
    else
        local_opts.verbose = 0;
    end
    local_opts.L = LocalL;
    local_opts.adaptive = true;
    local_opts.backtrack = true;
    local_opts.tau = TauVal;
    local_opts.sigma = SigmaVal;
    local_opts.a = 0.5;
    local_opts.eta = 0.95;
    local_opts.Delta = 2.0;
    local_opts.gamma = 0.75;
    local_opts.b = 0.95;
    local_opts.stopRule = 'absrel';
    local_opts.f1 = @(x,y) 0;
    local_opts.f2 = @(x,y) 0;
    local_opts.stopNow = @(x,y,primal,dual,maxPrimal,maxDual) all(double(primal)<opts.tol.*max(1,double(maxPrimal))) && all(double(dual)<opts.tol.*max(1,double(maxDual)));

    [sol, out] = apdhgp_adaptive(x0_dd, y0_dd, A, At, fProx, gProx, local_opts);

    % Scale the solution back to standard variables
    sol = sol .* RightPrecond_dd;

    % Add the preconditioner to the output structure so we can unscale the duals
    out.LeftPrecond = LeftPrecond_dd;
end

function val = fProx_helper(x, tau, f_dd, ub_scaled, lb_scaled)
    val = min(ub_scaled, max(x - tau.*f_dd, lb_scaled));
end

function val = gProx_helper(y, sigma, Nin, m, bin_dd, beq_dd)
    if Nin == 0
        val = y - sigma.*beq_dd;
    elseif m == Nin
        val = max(y - sigma.*bin_dd, DoubleDouble(0));
    else
        val = [ max(y.Index((1:Nin).') - sigma.*bin_dd, DoubleDouble(0)) ; y.Index((Nin+1:m).') - sigma.*beq_dd ];
    end
end

function val = A_helper(x, M_scaled)
    if isscalar(x.v2) && all(x.v2 == 0)
        val = DoubleDouble(M_scaled * x.v1);
    else
        val = DoubleDouble(M_scaled * x.v1) + DoubleDouble(M_scaled * x.v2);
    end
end

function val = At_helper(y, M_scaled)
    if isscalar(y.v2) && all(y.v2 == 0)
        val = DoubleDouble(M_scaled.' * y.v1);
    else
        val = DoubleDouble(M_scaled.' * y.v1) + DoubleDouble(M_scaled.' * y.v2);
    end
end
