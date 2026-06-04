function [x,outs] = apdhgp_adaptive(x, y, A, At, fProx, gProx, opts)

    %%  Check whether we have function handles or matrices
    if ~isnumeric(A)
        assert(~isnumeric(At),'If A is a function handle, then At must be a handle as well.')
    end
    %  If we have matrices, create functions so we only have to treat one case
    if coder.target( 'MATLAB' )
        if isnumeric(A)
            At = @(x)A.'*x;
            A = @(x) A*x;
        end
    else
        assert( ~isnumeric(A) );
        assert( ~isnumeric(At) );
    end

    %%  Check preconditions
    % Make sure 'opts' struct exists, and is filled with all the options
    if ~exist('opts','var')
        opts = [];
    end
    opts = setDefaults(opts,x,A,At);
    %  Make sure A and At are adjoints of one another
    if coder.target( 'MATLAB' )
        checkAdjoints( A, At, x, y );
    end

    %% Get some commonly used values from the 'opts' struct
    tau = opts.tau;        % primal stepsize
    sigma = opts.sigma;    % dual stepsize
    maxIters = opts.maxIters;
    a = opts.a;             % adaptivity level
    L = opts.L;             % Reciprocal spectral radius of A.'A
    Delta = opts.Delta;     % Used to compare residuals to decide when to update stepsizes

    %% Allocate space for the returned variables in the 'outs' struct
    outs = struct();
    outs.tau = zeros(maxIters,1);    % primal stepsizes
    outs.sigma = zeros(maxIters,1);  % dual stepsizes
    outs.f1 = zeros(maxIters,1);     % optional function evaluation
    outs.f2 = zeros(maxIters,1);     % optional function evaluation
    outs.p = zeros(maxIters,1);      % primal residuals
    outs.d = zeros(maxIters,1);      % dual residuals
    outs.y = y;                      % dual variable output
    outs.updates = 0;                % number of stepsize updates
    outs.iters = 0;                  % number of iterations

    %% Initialize some values
    updates = 0;
    Ax = A(x);
    Aty = At(y);

    maxPrimal = -DoubleDouble.Inf( 1, 1 );
    maxDual = -DoubleDouble.Inf( 1, 1 );

    %% Begin Iteration
    for iter = 1:maxIters

        % store old iterates
        x0 = x;
        y0 = y;
        Ax0 = Ax;
        Aty0 = Aty;

        % primal update
        x = fProx(x-tau.*Aty,tau);
        Ax = A(x);
        Axh = 2.*Ax-Ax0;

        % dual update
        y = gProx(y+sigma.*Axh,sigma);
        Aty = At(y);

        % compute and store residuals
        dx = x-x0;
        dy = y-y0;
        r1 = dx./tau + Aty0;
        r2 = Aty;
        d1 = dy./sigma + Axh;
        d2 = Ax;
        primal = VecNorm( r1 - r2 );
        dual = VecNorm( d1 - d2 );
        maxPrimal = max(maxPrimal,primal);
        maxDual = max(maxDual,dual);

        outs.p(iter) = double(primal);
        outs.d(iter) = double(dual);

        % store various values that we wish to track
        outs.f1(iter) = double(opts.f1(x,y));
        outs.f2(iter) = double(opts.f2(x,y));
        outs.tau(iter) = double(tau);
        outs.sigma(iter) = double(sigma);

        if opts.verbose >= 1 && mod(iter, 100) == 0
            primal_double = double(primal);
            dual_double = double(dual);
            fprintf('APDHGP: Iteration %d, primal res: %e, dual res: %e\n', int32(iter), primal_double(1, 1), dual_double(1, 1));
        end

        % Test stopping conditions
        if ( opts.stopNow(x,y,primal,dual,maxPrimal,maxDual) && iter>5) || iter>=maxIters
            outs.y = y;
            outs.p = outs.p(1:iter);
            outs.d = outs.d(1:iter);
            outs.f1 = outs.f1(1:iter);
            outs.f2 = outs.f2(1:iter);
            outs.updates  = updates;
            outs.tau = outs.tau(1:iter);
            outs.sigma = outs.sigma(1:iter);
            outs.iters = iter;
            return;
        end

        % Test the backtracking/stability condition
        DotProduct = ( Ax - Ax0 ) .* dy;
        Axy = 2 .* sum( DotProduct, 1 );
        Hnorm = ( VecNorm( dx ) .* VecNorm( dx ) ) ./ tau + ( VecNorm( dy ) .* VecNorm( dy ) ) ./ sigma;
        if opts.backtrack
            BacktrackCond = ( opts.gamma * double( Hnorm ) ) < double( Axy );
            if all( BacktrackCond( : ) )
                x = x0;
                y = y0;
                Ax = Ax0;
                Aty = Aty0;
                decay_val = opts.b * opts.gamma * double( Hnorm ) / double( Axy );
                decay = decay_val(1);
                tau = tau * decay;
                sigma = sigma * decay;
                L = L * decay * decay;
            end
        end

        % Perform adaptive update
        if opts.adaptive && iter > 1
            MaxRes = max( double( primal ), double( dual ) );
            PrevMaxRes = max( outs.p( iter - 1 ), outs.d( iter - 1 ) );
            if all( MaxRes( : ) < PrevMaxRes )
                PrimalCond = double( primal ) > Delta * double( dual );
                if all( PrimalCond( : ) )
                    tau = tau / ( 1 - a );
                    sigma = L / tau;
                    a = a * opts.eta;
                    updates = updates + 1;
                end
                DualCond = double( primal ) < double( dual ) / Delta;
                if all( DualCond( : ) )
                    tau = tau * ( 1 - a );
                    sigma = L / tau;
                    a = a * opts.eta;
                    updates = updates + 1;
                end
            end
        end

    end  % end for loop

end


%% Check that A and At represent adjoints
function checkAdjoints( A, At, X, Y )
    RX = DoubleDouble( randn( numel( X ), 1 ) );
    RY = DoubleDouble( randn( numel( Y ), 1 ) );
    Prod1 = A( RX ) .* RY;
    Prod2 = RX .* At( RY );
    Dot1 = sum( Prod1, 1 );
    Dot2 = sum( Prod2, 1 );
    RelativeError = abs( Dot1 - Dot2 ) ./ ( abs( Dot1 ) + abs( Dot2 ) );
    assert( double( RelativeError ) < 1e-6, 'At is not the adjoint of A' );
end


%% Fill in the struct of options with the default values
function opts = setDefaults(opts,x0,A,At)

    %  L:  The reciprocal of the spectral radius of A.'A.
    %  Approximate the spectral radius of A.'A if we don't know L
    if coder.target( 'MATLAB' )
        if ~isfield( opts, 'L' ) || opts.L <= 0

            X = randn( numel( x0 ), 1 );
            Transform = At( A( X ) );
            SpecRadius = norm( Transform ) ./ norm( X );
            opts.L = 2 ./ SpecRadius;

        end
    else
        assert( isfield( opts, 'L' ) && opts.L > 0, 'L must be provided for MEX target');
    end
    %  verbose: The verbosity level
    if ~isfield(opts,'verbose')
        opts.verbose = 0;
    end
    %  maxIters: The maximum number of iterations
    if ~isfield(opts,'maxIters')
        opts.maxIters = 10000;
    end
    % tol:  The relative decrease in the residuals before the method stops
    if ~isfield(opts,'tol') % Stopping tolerance
        opts.tol = 1e-14;
    end
    % adaptive:  If 'true' then use adaptive method.
    if ~isfield(opts,'adaptive')    %  is Adaptive?
        opts.adaptive = true;
    end

    % backtrack:  If 'true' then use backtracking method.
    if ~isfield(opts,'backtrack')    %  is backtracking?
        opts.backtrack = true;
    end

    % f1:  An optional function that is computed and stored after every
    % iteration
    if ~isfield(opts,'f1')
        opts.f1 = @(x,y) 0;
    end
    % f2:  An optional function that is computed and stored after every
    % iteration
    if ~isfield(opts,'f2')
        opts.f2 = @(x,y) 0;
    end
    % tau:  The initial stepsize for the primal variables
    if coder.target( 'MATLAB' )
        if ~isfield(opts,'tau') || opts.tau<=0         % starting value of tau
            opts.tau = sqrt(opts.L);
        end
    else
        assert( isfield(opts,'tau') && opts.tau > 0 );
    end
    % sigma: The initial stepsize for the dual variables
    if coder.target( 'MATLAB' )
        if ~isfield(opts,'sigma') || opts.sigma<=0       % starting value of sigma
            opts.sigma = opts.L./opts.tau;
        end
    else
        assert( isfield(opts,'sigma') && opts.sigma > 0 );
    end

    %% Adaptivity parameters
    if ~isfield(opts,'a')   %  Initial adaptive update strength for stepsizes
        opts.a = 0.5;
    end
    if ~isfield(opts,'eta') %  How fast does the adaptivity level decay
        opts.eta = 0.95;
    end
    if ~isfield(opts,'Delta') % update stepsizes when primal/dual ratio exceeds Delta
        opts.Delta = 2.0;
    end
    if ~isfield(opts,'gamma') % Used to determine when need to backtrack to maintain positivity conditions
        opts.gamma = 0.75;
    end
    if ~isfield(opts,'b')  % Adaptivity parameter used for backtracking update
        opts.b = 0.95;
    end

    %% Stopping conditions
    if coder.target( 'MATLAB' )
        if isfield(opts,'stopNow')
            opts.stopRule = 'custom';
        end

        if ~isfield(opts,'stopRule')
            opts.stopRule = 'absrel';
        end

        if strcmp(opts.stopRule,'abs')
            opts.stopNow = @(x,y,primal,dual,maxPrimal,maxDual) all( double(primal)<opts.tol ) && all( double(dual)<opts.tol );
        end

        if strcmp(opts.stopRule,'rel')
            opts.stopNow = @(x,y,primal,dual,maxPrimal,maxDual) (all(double(primal)<opts.tol.*max(1,double(maxPrimal))) && all(double(dual)<opts.tol.*max(1,double(maxDual)))) || (all( double(primal)<1e-28 ) && all( double(dual)<1e-28 ));
        end

        if strcmp(opts.stopRule,'absrel')
            opts.stopNow = @(x,y,primal,dual,maxPrimal,maxDual) all(double(primal)<opts.tol.*max(1,double(maxPrimal))) && all(double(dual)<opts.tol.*max(1,double(maxDual)));
        end

        if strcmp(opts.stopRule,'iter')
            opts.stopNow = @(x,y,primal,dual,maxPrimal,maxDual) iter > opts.maxIters;
        end

        assert(isfield(opts,'stopNow'),['Invalid choice for stopping rule: ' opts.stopRule ]);
    else
        assert(~isempty(opts.stopRule));
    end
end

function NormVal = VecNorm( VectorVal )
    NormVal = sqrt( sum( VectorVal .* VectorVal, 1 ) );
end
