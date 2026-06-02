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
    for k = 1:10
        % Row scaling factors
        r = max(abs(M_temp), [], 2);
        r(r == 0) = 1;
        r_scale = sqrt(r);
        
        % Column scaling factors
        c = max(abs(M_temp), [], 1)';
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

    % Convert inputs to DoubleDouble precision
    f_dd = DoubleDouble(f_scaled);
    M_dd = DoubleDouble(M_scaled);
    b_dd = DoubleDouble(b_scaled);

    LeftPrecond_dd = DoubleDouble(LeftPrecond);
    RightPrecond_dd = DoubleDouble(RightPrecond);

    bin_dd = b_dd(1:Nin);
    beq_dd = b_dd(Nin+1:end);

    % Scale bounds
    ub_scaled_double = ub;
    finite_ub = isfinite(ub);
    ub_scaled_double(finite_ub) = ub(finite_ub) ./ RightPrecond(finite_ub);
    ub_scaled = DoubleDouble(ub_scaled_double);

    lb_scaled_double = lb;
    finite_lb = isfinite(lb);
    lb_scaled_double(finite_lb) = lb(finite_lb) ./ RightPrecond(finite_lb);
    lb_scaled = DoubleDouble(lb_scaled_double);

    %% Define the ingredients PDHG needs to solve this problem
    fProx = @(x,tau) min(ub_scaled, max(x - tau*f_dd, lb_scaled));
    gProx = @(y,sigma) [ max(y(1:Nin)-sigma*bin_dd, DoubleDouble(0)) ; y(Nin+1:end)-sigma*beq_dd ];
    A = @(x) M_dd*x;
    At = @(y) M_dd'*y;

    % Scale the initial primal and dual guesses
    x0_dd = DoubleDouble(x_init) ./ RightPrecond_dd;
    y0_dd = DoubleDouble(y_init) ./ LeftPrecond_dd;

    %% Call the adaptive PDHG high-precision solver
    if ~exist('opts','var')
        opts = [];
    end

    [sol, out] = apdhgp_adaptive(x0_dd, y0_dd, A, At, fProx, gProx, opts);

    % Scale the solution back to standard variables
    sol = sol .* RightPrecond_dd;

    % Add the preconditioner to the output structure so we can unscale the duals
    out.LeftPrecond = LeftPrecond_dd;
end
