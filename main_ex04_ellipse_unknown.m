function result = main_ex04_ellipse_unknown(noise_level, Nbasis, gtrue, a, b)
%EX04_ELLIPSE_UNKNOWN_G
% Setup (paper example: unknown g(t) on an ellipse domain):
%   Ellipse domain + unknown time-dependent g(t) + 3 sensors,
%   recover source location p=(x1,x2) and g(t) from sparse boundary flux data.
%
% Domain:
%   Ω = { (x/a)^2 + (y/b)^2 <= 1 }  (ellipse)
%
% PDE:
%   u_t - Δu = g(t) * δ(x - p),  u|_{∂Ω}=0,  u(·,0)=0
%
% Forward:
%   MATLAB PDE Toolbox (P1 FEM + implicit time stepping via solvepde)
%   Point source δ approximated by a normalized Gaussian with sigma=0.03
%   Flux evaluated via evaluateGradient at points slightly inside boundary
%
% Inverse:
%   Represent g(t) by time hat basis on coarse time grid:
%       g(t) = B(t) * c,   c ∈ R^{Nbasis}
%   Given p, estimate c by least squares:
%       minimize ||Phi(p)*c - y||_2,   Phi columns are flux responses to basis hats
%   Outer: regularized Gauss-Newton on p=(x1,x2) with fixed damping,
%          Jacobian by central finite differences, Armijo backtracking line search.
%
% Reproducibility / no inverse crime:
%   - Fine mesh/time for data generation, coarse mesh/time for inversion
%   - Multiplicative Gaussian noise: y^δ = y * (1 + δ ξ), ξ~N(0,1)
%
% Usage:
%   result = ex04_ellipse_unknown_g();  % Reproduce Table 4 (loops noise levels)
%   result = ex04_ellipse_unknown_g(0.05, 20);
%   result = ex04_ellipse_unknown_g(0.03, 30, @(t) sin(2*pi*t)+1, 1.2, 0.8);
%
% Output:
%   result struct with truth, estimates, errors, history, settings.

    addpath(fullfile(fileparts(mfilename('fullpath')), 'functions'));

    if nargin < 1
        fprintf('=======================================================\n');
        fprintf('Running Example 4: Unknown Amplitude on Ellipse\n');
        fprintf('=======================================================\n');
        noises = [0.01, 0.03, 0.05]; 
        for d = noises
            main_ex04_ellipse_unknown(d);
        end
        if nargout > 0, result = []; end
        return;
    end

    %% -------------------- 0) Defaults --------------------
    if nargin < 1 || isempty(noise_level)
        noise_level = 0;
    end
    if nargin < 2 || isempty(Nbasis)
        Nbasis = 20;
    end
    if nargin < 4 || isempty(a)
        a = 1.2;
    end
    if nargin < 5 || isempty(b)
        b = 0.8;
    end

    T_end = 1.0;

    % Default true g(t): Hann bump on [0,T1]
    if nargin < 3 || isempty(gtrue)
        T1_true = 0.5;
        A_true  = 2.0;
        gtrue = @(t) A_true .* (t>=0 & t<=T1_true) .* sin(pi*t/T1_true).^2;
    end

    fprintf('=== EX04: Ellipse, unknown g(t), 3 sensors ===\n');
    fprintf('Ellipse axes: a=%.3f, b=%.3f\n', a, b);
    fprintf('Noise level δ = %.1f%%\n', 100*noise_level);
    fprintf('Nbasis = %d\n', Nbasis);

    %% -------------------- 1) Problem setup --------------------
    % Truth (ellipse-"polar": x=a*r*cos, y=b*r*sin)
    truth.r     = 0.40;
    truth.theta = 2.00;
    truth.p     = [a*truth.r*cos(truth.theta), b*truth.r*sin(truth.theta)]; % [x1,x2]

    % Sensors on ellipse boundary
    sensorAngles = [pi/6, 1.1*pi/2, 5*pi/6];
    sensors      = [a*cos(sensorAngles)', b*sin(sensorAngles)'];
    Ns           = size(sensors,1);

    % Dirac approximation
    sigma = 0.03;

    % Flux evaluation point: slightly inside boundary to avoid NaNs in evaluateGradient
    shrink = 0.997;

    % source margin (ellipse-radius factor): s = sqrt((x/a)^2+(y/b)^2) <= srcMax
    srcMax = 0.997;

    % No inverse crime: fine vs coarse (space + time)
    meshFineH   = 0.02;
    NtFine      = 600;
    tFine       = linspace(0, T_end, NtFine+1).';

    meshCoarseH = 0.04;
    NtCoarse    = 300;
    tCoarse     = linspace(0, T_end, NtCoarse+1).';

    % Time hat basis on coarse grid
    basis = build_time_hats(tCoarse, Nbasis);

    %% -------------------- 2) Generate synthetic data (fine) --------------------
    fprintf('--- Generating synthetic data (fine mesh/time)...\n');

    modelFine = createpde(1);
    create_ellipse_geometry(modelFine, a, b);
    generateMesh(modelFine, 'Hmax', meshFineH);

    resFine  = solve_heat_fem_time_dependent(modelFine, truth.p, gtrue, tFine, sigma);
    fluxFine = extract_flux_ellipse(resFine, sensorAngles, a, b, shrink);

    % Interpolate to coarse time grid
    fluxExact = zeros(numel(tCoarse), Ns);

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
    rng(1,'twister');
    noise   = randn(size(fluxExact));
    fluxObs = fluxExact .* (1 + noise_level * noise);

    y_vec = fluxObs(:);

    % Store true g on coarse grid (for reporting/plotting)
    g_true_vec = gtrue(tCoarse);

    %% -------------------- 3) Inversion (coarse) --------------------
    fprintf('--- Solving inverse problem (coarse mesh/time)...\n');

    modelCoarse = createpde(1);
    create_ellipse_geometry(modelCoarse, a, b);
    generateMesh(modelCoarse, 'Hmax', meshCoarseH);

    % Initial guess (ellipse-"polar")
    r0     = 0.50;
    theta0 = 1.80;
    p      = [a*r0*cos(theta0), b*r0*sin(theta0)]; % [x1,x2]
    p      = project_to_ellipse(p, a, b, srcMax);

    % Outer GN (fixed damping)
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
    history.p       = zeros(maxIter+1, 2);
    history.cost    = zeros(maxIter+1, 1);
    history.norm_c  = zeros(maxIter+1, 1);

    % Forward+fit: p -> (Y_fit_vec, cHat, g_est_vec)
    forwardFit = @(pp) local_forward_fit_unknown_g( ...
        modelCoarse, pp, gtrue, tCoarse, sensorAngles, a, b, shrink, sigma, basis, y_vec);

    % Init
    [Y_vec, cHat, g_est_vec] = forwardFit(p);
    r = Y_vec - y_vec;
    cost = 0.5*(r.'*r);

    nIter = 0;
    history.p(1,:)      = p;
    history.cost(1)     = cost;
    history.norm_c(1)   = norm(cHat);

    fprintf('Iter %2d: cost=%.4e, p=(%.6f, %.6f), ||c||=%.3e\n', ...
        0, cost, p(1), p(2), norm(cHat));

    stopReason = "maxIter reached";

    if cost < tolCostAbs
        stopReason = "cost below tolCostAbs at init";
    else
        for k = 1:maxIter

            % ----- 3a) Central FD Jacobian of residual r(p) wrt (x1,x2) -----
            x1 = p(1); x2 = p(2);
            epsStep = 1e-10;

            % === x1-direction ===
            p_plus  = project_to_ellipse([x1 + fdStep(1), x2], a, b, srcMax);
            p_minus = project_to_ellipse([x1 - fdStep(1), x2], a, b, srcMax);
            hx_eff  = 0.5*(p_plus(1) - p_minus(1));

            if abs(hx_eff) >= epsStep
                Yp = forwardFit(p_plus);  rp = Yp - y_vec;
                Ym = forwardFit(p_minus); rm = Ym - y_vec;
                Jx1 = (rp - rm) / (2*hx_eff);
            else
                h_for = (p_plus(1) - x1);
                if abs(h_for) >= epsStep
                    Yp = forwardFit(p_plus); rp = Yp - y_vec;
                    Jx1 = (rp - r) / h_for;
                else
                    h_back = (x1 - p_minus(1));
                    if abs(h_back) >= epsStep
                        Ym = forwardFit(p_minus); rm = Ym - y_vec;
                        Jx1 = (r - rm) / h_back;
                    else
                        warning('FD step in x1 collapsed after projection (both sides). Setting Jx1=0.');
                        Jx1 = zeros(size(r));
                    end
                end
            end

            % === x2-direction ===
            p_plus  = project_to_ellipse([x1, x2 + fdStep(2)], a, b, srcMax);
            p_minus = project_to_ellipse([x1, x2 - fdStep(2)], a, b, srcMax);
            hy_eff  = 0.5*(p_plus(2) - p_minus(2));

            if abs(hy_eff) >= epsStep
                Yp = forwardFit(p_plus);  rp = Yp - y_vec;
                Ym = forwardFit(p_minus); rm = Ym - y_vec;
                Jx2 = (rp - rm) / (2*hy_eff);
            else
                h_for = (p_plus(2) - x2);
                if abs(h_for) >= epsStep
                    Yp = forwardFit(p_plus); rp = Yp - y_vec;
                    Jx2 = (rp - r) / h_for;
                else
                    h_back = (x2 - p_minus(2));
                    if abs(h_back) >= epsStep
                        Ym = forwardFit(p_minus); rm = Ym - y_vec;
                        Jx2 = (r - rm) / h_back;
                    else
                        warning('FD step in x2 collapsed after projection (both sides). Setting Jx2=0.');
                        Jx2 = zeros(size(r));
                    end
                end
            end

            J = [Jx1, Jx2];

            % ----- 3b) Regularized GN step + Armijo backtracking -----
            H    = J.'*J;
            grad = J.'*r;

            if norm(grad) < tolGrad
                stopReason = "gradient below tolGrad";
                break;
            end

            dp = - (H + lambdaFixed*eye(2)) \ grad;
            slope = grad.' * dp;

            % Safety: if not descent, fallback to steepest descent
            if ~(isfinite(slope)) || slope >= 0
                warning('Gauss-Newton step not a descent direction; falling back to steepest descent.');
                dp    = -grad;
                slope = -grad.'*grad;
            end

            stepNorm = norm(dp);
            if stepNorm / max(1, norm(p)) < tolStepRel
                stopReason = "relative step below tolStepRel";
                break;
            end

            alpha = alphaInit;
            accepted = false;

            while alpha >= alphaMin
                p_try = project_to_ellipse(p + (alpha*dp).', a, b, srcMax);

                [Y_try, c_try, g_try] = forwardFit(p_try);
                r_try = Y_try - y_vec;
                cost_try = 0.5*(r_try.'*r_try);

                if cost_try <= cost + cArmijo*alpha*slope
                    % ACCEPT
                    p         = p_try;
                    cHat      = c_try;
                    g_est_vec = g_try;
                    r         = r_try;
                    cost      = cost_try;
                    accepted  = true;
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
            history.p(nIter+1,:)     = p;
            history.cost(nIter+1)    = cost;
            history.norm_c(nIter+1)  = norm(cHat);

            fprintf('Iter %2d: cost=%.4e, step=%.3e, p=(%.6f, %.6f), ||c||=%.3e\n', ...
                nIter, cost, stepNorm, p(1), p(2), norm(cHat));

            if cost < tolCostAbs
                stopReason = "cost below tolCostAbs";
                break;
            end
        end
    end

    % Trim history
    history.p      = history.p(1:nIter+1,:);
    history.cost   = history.cost(1:nIter+1);
    history.norm_c = history.norm_c(1:nIter+1);

    %% -------------------- 4) Report --------------------
    x1_true = truth.p(1); x2_true = truth.p(2);
    x1_est  = p(1);       x2_est  = p(2);

    err.x1 = abs(x1_est - x1_true);
    err.x2 = abs(x2_est - x2_true);

    % g(t) relative L2 error on coarse grid
    dt = tCoarse(2) - tCoarse(1);
    g_est_vec = g_est_vec(:);
    g_true_vec = g_true_vec(:);

    err_g_L2  = sqrt(sum((g_est_vec - g_true_vec).^2) * dt);
    norm_g_L2 = sqrt(sum(g_true_vec.^2) * dt);
    err_g_rel = err_g_L2 / max(norm_g_L2, 1e-14);

    fprintf('--- Final (EX04) ---\n');
    fprintf('Stop reason: %s\n', stopReason);
    fprintf('True: x1=%.4f, x2=%.4f\n', x1_true, x2_true);
    fprintf('Est : x1=%.4f, x2=%.4f\n', x1_est,  x2_est);
    fprintf('Err : |Δx1|=%.2e, |Δx2|=%.2e\n', ...
        err.x1, err.x2);
    fprintf('g(t) errors: ||g_est-g_true||_2 = %.2e, rel = %.2e\n', err_g_L2, err_g_rel);

    %% -------------------- 5) Plots --------------------
    fs    = 16;
    lw    = 1.3;
    ms    = 8;
    legfs = fs - 2;

    % Recompute final fitted Y for plotting
    [Y_fit_vec, ~, ~] = forwardFit(p);
    Y_fit = reshape(Y_fit_vec, numel(tCoarse), Ns);

    figure;
    for s = 1:Ns
        subplot(Ns,1,s);
        plot(tCoarse, fluxObs(:,s), 'k.', 'MarkerSize', ms-2); hold on;
        plot(tCoarse, Y_fit(:,s), '-',  'LineWidth', lw);

        xlabel('t', 'FontSize', fs);
        legend({'obs','fit'}, 'Location','best', 'FontSize', legfs);
        grid on; box on;
        set(gca, 'FontSize', fs);
    end

    figure;
    th = linspace(0, 2*pi, 400);
    plot(a*cos(th), b*sin(th), 'k-', 'LineWidth', lw); hold on;
    plot(x1_true, x2_true, 'ko', 'MarkerFaceColor','g', 'MarkerSize', ms);
    plot(x1_est,  x2_est,  'ks', 'MarkerFaceColor','r', 'MarkerSize', ms-4);
    plot(sensors(:,1), sensors(:,2), 'k^', 'MarkerFaceColor','b', 'MarkerSize', ms-1);

    axis equal; grid on; box on;
    xlabel('x_1', 'FontSize', fs);
    ylabel('x_2', 'FontSize', fs);
    set(gca, 'FontSize', fs);
    legend({'boundary','true p','est p','sensors'}, 'Location','best', 'FontSize', legfs);

    figure;
    plot(tCoarse, g_est_vec, 'r-', 'LineWidth', lw); hold on;
    plot(tCoarse, g_true_vec, 'k--', 'LineWidth', lw);
    grid on; box on;
    xlabel('t', 'FontSize', fs);
    ylabel('g(t)', 'FontSize', fs);
    set(gca, 'FontSize', fs);
    legend({'g_{est}','g_{true}'}, 'Location','best', 'FontSize', legfs);

    %% -------------------- 6) Pack result --------------------
    result.name        = "EX04_ellipse_unknown_g";
    result.noise_level = noise_level;

    result.ellipse.a = a;
    result.ellipse.b = b;

    result.truth.p = truth.p;
    result.est.p   = p;

    result.errors          = err;
    result.errors_g.L2     = err_g_L2;
    result.errors_g.relL2  = err_g_rel;

    result.g_true = g_true_vec;
    result.g_est  = g_est_vec;

    result.history = history;

    settings.a           = a;
    settings.b           = b;
    settings.T_end       = T_end;
    settings.sigma       = sigma;
    settings.shrink      = shrink;
    settings.srcMax      = srcMax;
    settings.meshFineH   = meshFineH;
    settings.meshCoarseH = meshCoarseH;
    settings.NtFine      = NtFine;
    settings.NtCoarse    = NtCoarse;
    settings.sensors     = sensors;
    settings.sensorAngles= sensorAngles;
    settings.Nbasis      = Nbasis;
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

function [Y_fit_vec, cHat, g_est_vec] = local_forward_fit_unknown_g( ...
    model, p, ~, tlist, sensorAngles, a, b, shrink, sigma, basis, y_vec)
% Given p:
%   1) Build Phi(p) columns by solving forward for each hat basis
%   2) Solve plain LS for cHat: min ||Phi c - y||_2
%   3) Return fitted Y = Phi cHat and g_est = B cHat

    Nt = numel(tlist);
    Ns = numel(sensorAngles);
    K  = size(basis.B,2);

    YB = zeros(Nt, Ns, K);

    for k = 1:K
        Bk = basis.B(:,k);
        gk = @(t) interp1(tlist, Bk, t, 'linear', 0.0);

        resk  = solve_heat_fem_time_dependent(model, p, gk, tlist, sigma);

        YB(:,:,k) = extract_flux_ellipse(resk, sensorAngles, a, b, shrink);
    end

    Phi = reshape(YB, [], K);

    % Plain LS (prefer min-norm if available)
    if exist('lsqminnorm','file') == 2
        cHat = lsqminnorm(Phi, y_vec);
    else
        cHat = Phi \ y_vec;
    end

    Y_fit_vec = Phi * cHat;
    g_est_vec = basis.B * cHat;
end