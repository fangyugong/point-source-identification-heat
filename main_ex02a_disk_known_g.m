function result = main_ex02a_disk_known_g(noise_level, gfun)
%EX02A_DISC_KNOWN_GT
% Setup (paper example: known time-dependent amplitude on the unit disk):
%   Unit disk + known g(t) + 2 sensors,
%   recover p = (x1,x2) from sparse boundary flux data.
%
% PDE:
%   u_t - Δu = g(t) * δ(x - p),  u|_{∂Ω}=0,  u(·,0)=0,  Ω = unit disk
%
% Forward:
%   MATLAB PDE Toolbox (P1 FEM + implicit time stepping via solvepde)
%   Point source δ approximated by a normalized Gaussian with sigma=0.03
%
% Inverse:
%   Regularized Gauss-Newton step for p with fixed damping lambda,
%   Jacobian by central finite differences, Armijo backtracking line search.
%
% Reproducibility:
%   - Fine mesh/time for data generation, coarse mesh/time for inversion (no inverse crime)
%   - Multiplicative Gaussian noise: y^δ = y * (1 + δ ξ), ξ~N(0,1)
%
% Usage:
%   result = ex02a_disc_known_gt();                       % Reproduce Table 2 case(a) (loops noise levels)
%   result = ex02a_disc_known_gt(0.03);                   % 3% noise
%   result = ex02a_disc_known_gt(0.03, @(t) sin(2*pi*t)+1);
%
% Output:
%   result struct with truth, estimates, errors, history, settings.

    addpath(fullfile(fileparts(mfilename('fullpath')), 'functions'));

    if nargin < 1
        fprintf('=======================================================\n');
        fprintf('Running Example 2a: Known Amplitude on Disk\n');
        fprintf('=======================================================\n');
        noises = [0.03, 0.05, 0.10];
        for d = noises
            main_ex02a_disk_known_g(d);
        end
        if nargout > 0, result = []; end
        return;
    end

    %% -------------------- 0) Defaults --------------------
    if nargin < 1 || isempty(noise_level)
        noise_level = 0;
    end
    if nargin < 2 || isempty(gfun)
        % Default time-dependent amplitude
        gfun = @(t) sin(2*pi*t) + 1;
    end

    fprintf('=== EX02a: Unit disk, known g(t), 2 sensors ===\n');
    fprintf('Noise level δ = %.1f%%\n', 100*noise_level);

    %% -------------------- 1) Problem setup --------------------
    R     = 1.0;
    T_end = 1.0;

    % Truth (polar -> Cartesian)
    truth.r     = 0.4;
    truth.theta = 2.0;
    truth.p     = [truth.r*cos(truth.theta), truth.r*sin(truth.theta)]; % [x1,x2]

    % Sensors on boundary
    sensorAngles = [1.7, 2.0];
    sensors      = [R*cos(sensorAngles)', R*sin(sensorAngles)'];
    Ns           = size(sensors,1);

    % Dirac approximation
    sigma  = 0.03;

    % Flux evaluation point: slightly inside boundary to avoid NaNs in evaluateGradient
    shrink = 0.997;

    % source margin: ||p|| <= srcMax*R
    srcMax = 0.997;

    % No inverse crime: fine vs coarse
    meshFineH   = 0.02;
    NtFine      = 600;
    tFine       = linspace(0, T_end, NtFine+1);

    meshCoarseH = 0.04;
    NtCoarse    = 300;
    tCoarse     = linspace(0, T_end, NtCoarse+1);

    %% -------------------- 2) Generate synthetic data (fine) --------------------
    fprintf('--- Generating synthetic data (fine mesh/time)...\n');

    modelFine = createpde(1);
    create_disk_geometry(modelFine, R);
    generateMesh(modelFine, 'Hmax', meshFineH);

    resFine  = solve_heat_fem_time_dependent(modelFine, truth.p, gfun, tFine, sigma);
    fluxFine = extract_flux_disk(resFine, sensors, shrink);

    % Interpolate to coarse time grid
    fluxExact = zeros(NtCoarse+1, Ns);

    % If the coarse time grid is (numerically) a subset of the fine grid,
    % we can simply reuse the corresponding samples instead of interpolating.
    tolGrid = 1e-10;
    [lia, locb] = ismembertol(tCoarse, tFine, tolGrid);

    if all(lia)
        for s = 1:Ns
            fluxExact(:,s) = fluxFine(locb, s);
        end
    else
        for s = 1:Ns
            fluxExact(:,s) = interp1(tFine, fluxFine(:,s), tCoarse, 'pchip');
        end
    end

    % Multiplicative noise
    rng(1,'twister'); % fixed seed for reproducibility
    noise   = randn(size(fluxExact));
    fluxObs = fluxExact .* (1 + noise_level * noise);

    y_vec = fluxObs(:);

    %% -------------------- 3) Inversion (coarse) --------------------
    fprintf('--- Solving inverse problem (coarse mesh/time)...\n');

    modelCoarse = createpde(1);
    create_disk_geometry(modelCoarse, R);
    generateMesh(modelCoarse, 'Hmax', meshCoarseH);

    % Initial guess (polar -> Cartesian)
    r0     = 0.5;
    theta0 = 1.8;
    p      = [r0*cos(theta0), r0*sin(theta0)];
    p      = project_to_disk(p, srcMax*R);

    % Gauss-Newton (fixed damping)
    maxIter     = 10;
    lambdaFixed = 1e-4;

    fdStep      = [1e-3, 1e-3];  % [hx1, hx2]
    tolCostAbs  = 1e-8;
    tolStepRel  = 1e-4;
    tolGrad     = 1e-8;

    % Armijo backtracking
    alphaInit = 1.0;
    alphaMin  = 1/16;
    rho       = 0.5;
    cArmijo   = 1e-4;

    % History
    history.p    = zeros(maxIter+1, 2);
    history.cost = zeros(maxIter+1, 1);

    % Prediction map: p -> Y(p) (vectorized)
    computeY = @(pp) local_forward_flux_vec_known_g( ...
        modelCoarse, pp, gfun, tCoarse, sigma, sensors, shrink);

    % Initial residual/cost
    Y = computeY(p);
    r = Y - y_vec;
    cost = 0.5*(r'*r);

    nIter = 0;
    history.p(1,:)  = p;
    history.cost(1) = cost;

    fprintf('Iter %2d: cost=%.4e, p=(%.6f, %.6f)\n', 0, cost, p(1), p(2));

    stopReason = "maxIter reached";

    if cost < tolCostAbs
        stopReason = "cost below tolCostAbs at init";
    else
        for k = 1:maxIter

            % ----- 3a) Central FD Jacobian of residual r(p) wrt (x1,x2) -----
            x1 = p(1); x2 = p(2);
            epsStep = 1e-10;

            % ===== x1-direction (central if possible; else one-sided fallback) =====
            p_plus  = project_to_disk([x1 + fdStep(1), x2], srcMax*R);
            p_minus = project_to_disk([x1 - fdStep(1), x2], srcMax*R);
            % Use effective step after projection/clamp for numerical consistency
            hx_eff  = 0.5*(p_plus(1) - p_minus(1));

            if abs(hx_eff) >= epsStep
                Y_plus  = computeY(p_plus);
                Y_minus = computeY(p_minus);
                Jx1 = (Y_plus - Y_minus) / (2*hx_eff);
            else
                % one-sided fallback: forward first, then backward
                h_for = (p_plus(1) - x1);
                if abs(h_for) >= epsStep
                    Y_plus = computeY(p_plus);
                    Jx1 = (Y_plus - Y) / h_for;
                else
                    h_back = (x1 - p_minus(1));
                    if abs(h_back) >= epsStep
                        Y_minus = computeY(p_minus);
                        Jx1 = (Y - Y_minus) / h_back;
                    else
                        warning('FD step in x1 collapsed after projection (both sides). Setting Jx1 = 0.');
                        Jx1 = zeros(size(r));
                    end
                end
            end

            % ===== x2-direction =====
            p_plus  = project_to_disk([x1, x2 + fdStep(2)], srcMax*R);
            p_minus = project_to_disk([x1, x2 - fdStep(2)], srcMax*R);
            hy_eff  = 0.5*(p_plus(2) - p_minus(2));

            if abs(hy_eff) >= epsStep
                Y_plus  = computeY(p_plus);
                Y_minus = computeY(p_minus);
                Jx2 = (Y_plus - Y_minus) / (2*hy_eff);
            else
                h_for = (p_plus(2) - x2);
                if abs(h_for) >= epsStep
                    Y_plus = computeY(p_plus);
                    Jx2 = (Y_plus - Y) / h_for;
                else
                    h_back = (x2 - p_minus(2));
                    if abs(h_back) >= epsStep
                        Y_minus = computeY(p_minus);
                        Jx2 = (Y - Y_minus) / h_back;
                    else
                        warning('FD step in x2 collapsed after projection (both sides). Setting Jx2 = 0.');
                        Jx2 = zeros(size(r));
                    end
                end
            end

            J = [Jx1, Jx2];

            % ----- 3b) Regularized Gauss-Newton step + Armijo backtracking -----
            H    = J'*J;
            grad = J'*r;

            if norm(grad) < tolGrad
                stopReason = "gradient below tolGrad";
                break;
            end

            dp = - (H + lambdaFixed*eye(2)) \ grad;
            slope = grad' * dp;

            % Safety: if dp is not a descent direction, fall back to steepest descent
            if ~(isfinite(slope)) || slope >= 0
                warning('Gauss-Newton step is not a descent direction; falling back to steepest descent.');
                dp    = -grad;
                slope = -grad' * grad;
            end

            stepNorm = norm(dp);
            if stepNorm / max(1, norm(p)) < tolStepRel
                stopReason = "relative step below tolStepRel";
                break;
            end

            alpha = alphaInit;
            accepted = false;

            while alpha >= alphaMin
                p_try = project_to_disk(p + (alpha*dp)', srcMax*R);

                Y_try = computeY(p_try);
                r_try = Y_try - y_vec;
                cost_try = 0.5*(r_try'*r_try);

                % Armijo: F(p+αdp) <= F(p) + c α grad^T dp
                if cost_try <= cost + cArmijo*alpha*slope
                    % ACCEPT
                    p    = p_try;
                    Y    = Y_try;
                    r    = r_try;
                    cost = cost_try;
                    accepted = true;
                    break;
                else
                    alpha = rho * alpha;
                end
            end

            if ~accepted
                stopReason = "line search failed (alpha < alphaMin)";
                break;
            end

            % Record accepted update
            nIter = nIter + 1;
            history.p(nIter+1,:)  = p;
            history.cost(nIter+1) = cost;

            fprintf('Iter %2d: cost=%.4e, step=%.3e, p=(%.6f, %.6f)\n', ...
                nIter, cost, stepNorm, p(1), p(2));

            if cost < tolCostAbs
                stopReason = "cost below tolCostAbs";
                break;
            end
        end
    end

    % Trim history
    history.p    = history.p(1:nIter+1,:);
    history.cost = history.cost(1:nIter+1);

    %% -------------------- 4) Report --------------------
    x1_true = truth.p(1); x2_true = truth.p(2);
    x1_est  = p(1);       x2_est  = p(2);

    err.x1 = abs(x1_est - x1_true);
    err.x2 = abs(x2_est - x2_true);

    fprintf('--- Final (EX02a) ---\n');
    fprintf('Stop reason: %s\n', stopReason);
    fprintf('True: x1=%.4f, x2=%.4f\n', x1_true, x2_true);
    fprintf('Est : x1=%.4f, x2=%.4f\n', x1_est,  x2_est);
    fprintf('Err : |Δx1|=%.2e, |Δx2|=%.2e\n', err.x1, err.x2);

    %% -------------------- 5) Pack result --------------------
    result.name        = "EX02a_disc_known_gt";
    result.noise_level = noise_level;

    result.truth.p = truth.p;

    result.est.p = p;

    result.errors  = err;
    result.history = history;

    settings.R           = R;
    settings.T_end       = T_end;
    settings.sigma       = sigma;
    settings.shrink      = shrink;
    settings.srcMax      = srcMax;
    settings.meshFineH   = meshFineH;
    settings.meshCoarseH = meshCoarseH;
    settings.NtFine      = NtFine;
    settings.NtCoarse    = NtCoarse;
    settings.sensors     = sensors;
    settings.lambdaFixed = lambdaFixed;
    settings.fdStep      = fdStep;
    settings.alphaInit   = alphaInit;
    settings.alphaMin    = alphaMin;
    settings.rho         = rho;
    settings.cArmijo     = cArmijo;

    result.settings   = settings;
    result.stopReason = stopReason;
    result.nIter      = nIter;
end

function Y_vec = local_forward_flux_vec_known_g(model, p, gfun, tlist, sigma, sensors, shrink)
% Convenience wrapper: forward solve + flux extraction + vectorization
    res = solve_heat_fem_time_dependent(model, p, gfun, tlist, sigma);
    Y   = extract_flux_disk(res, sensors, shrink);
    Y_vec = Y(:);
end