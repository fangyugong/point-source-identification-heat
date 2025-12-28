function result = main_ex03_disk_pwc(noise_level)
%EX03_DISC_PWC_G
% Setup (paper example: piecewise-constant g(t) with unknown switch time on unit disk):
%   Unit disk + unknown (c1,c2,T1) in
%       g(t) = c1,  t <= T1
%              c2,  t >  T1
%   recover q = (x1,x2,T1,c1,c2) from sparse boundary flux data (2 sensors).
%
% PDE:
%   u_t - Δu = g(t) * δ(x - p),  u|_{∂Ω}=0,  u(·,0)=0,  Ω = unit disk
%
% Forward:
%   MATLAB PDE Toolbox (P1 FEM + implicit time stepping via solvepde)
%   Point source δ approximated by a normalized Gaussian with sigma=0.03
%   Flux evaluated via evaluateGradient at points slightly inside boundary
%
% Inverse:
%   Outer: regularized Gauss-Newton on q = [x1,x2,T1] with fixed damping,
%          Jacobian by central finite differences, Armijo backtracking line search.
%   Inner (given q): LS for (c1,c2):
%       minimize || Phi(q)*c - y ||_2,  Phi=[Y1(:),Y2(:)]
%
% Reproducibility / no inverse crime:
%   - Fine mesh/time for data generation, coarse mesh/time for inversion
%   - Multiplicative Gaussian noise: y^δ = y * (1 + δ ξ), ξ~N(0,1)
%
% Usage:
%   result = ex03_disc_pwc_g();        % Reproduce Table 3 (loops noise levels)
%   result = ex03_disc_pwc_g(0.03);    % 3% noise
%
% Output:
%   result struct with truth, estimates, errors, history, settings.

    addpath(fullfile(fileparts(mfilename('fullpath')), 'functions'));

    if nargin < 1
        fprintf('=======================================================\n');
        fprintf('Running Example 3: Piecewise Constant (Unknown T1)\n');
        fprintf('=======================================================\n');
        noises = [0.01, 0.03, 0.05]; 
        for d = noises
            main_ex03_disk_pwc(d);
        end
        if nargout > 0, result = []; end
        return;
    end

    %% -------------------- 0) Defaults --------------------
    if nargin < 1 || isempty(noise_level)
        noise_level = 0;
    end

    fprintf('=== EX03: Unit disk, piecewise-constant g(t), unknown T1, 2 sensors ===\n');
    fprintf('Noise level δ = %.1f%%\n', 100*noise_level);

    %% -------------------- 1) Problem setup --------------------
    R     = 1.0;
    T_end = 1.0;

    % Truth (polar -> Cartesian)
    truth.r     = 0.40;
    truth.theta = 2.00;
    truth.p     = [truth.r*cos(truth.theta), truth.r*sin(truth.theta)]; % [x1,x2]
    truth.T1    = 0.40;
    truth.c1    = 2.00;
    truth.c2    = 1.00;

    % Sensors on boundary
    sensorAngles = [1.7, 2.0];
    sensors      = [R*cos(sensorAngles)', R*sin(sensorAngles)'];
    Ns           = size(sensors,1);

    % Dirac approximation
    sigma = 0.03;

    % Flux evaluation point: slightly inside boundary to avoid NaNs in evaluateGradient
    shrink = 0.997;

    % source margin: ||p|| <= srcMax*R
    srcMax = 0.997;

    % No inverse crime: fine vs coarse (space + time)
    meshFineH   = 0.02;
    NtFine      = 600;
    tFine       = linspace(0, T_end, NtFine+1).';

    meshCoarseH = 0.04;
    NtCoarse    = 300;
    tCoarse     = linspace(0, T_end, NtCoarse+1).';

    % Initial guess (polar -> Cartesian)
    r0     = 0.50;
    theta0 = 1.80;
    p0     = [r0*cos(theta0), r0*sin(theta0)];  % [x1,x2]
    T10    = 0.35;
    q      = [p0, T10];                         % [x1,x2,T1]

    % Constraints for T1
    T_bounds = [0, T_end];

    % Outer GN parameters (fixed damping)
    maxIter     = 10;
    lambdaFixed = 1e-4;

    fdStepXY    = [1e-3, 1e-3];    % [hx1, hx2]
    hT_min      = 5e-3;            % min step for T1 FD
    tolCostAbs  = 1e-8;
    tolStepRel  = 1e-4;
    tolGrad     = 1e-8;

    % Armijo backtracking
    alphaInit = 1.0;
    alphaMin  = 1/16;
    rho       = 0.5;
    cArmijo   = 1e-4;

    %% -------------------- 2) Generate synthetic data (fine) --------------------
    fprintf('--- Generating synthetic data (fine mesh/time)...\n');

    modelFine = createpde(1);
    create_disk_geometry(modelFine, R);
    generateMesh(modelFine, 'Hmax', meshFineH);

    % Fine bases (g1 = 1_{t<=T1}, g2 = 1_{t>T1})
    [Y1_f, Y2_f] = local_forward_flux_bases_pwc( ...
        modelFine, truth.p, truth.T1, tFine, sensors, sigma, shrink);

    Y_true_fine = truth.c1 * Y1_f + truth.c2 * Y2_f;

    % Interpolate to coarse time grid
    Y_exact = zeros(numel(tCoarse), Ns);
    for s = 1:Ns
        Y_exact(:,s) = interp1(tFine, Y_true_fine(:,s), tCoarse, 'pchip');
    end
    
    % Multiplicative noise
    rng(1,'twister');
    noise = randn(size(Y_exact));
    Y_obs = Y_exact .* (1 + noise_level * noise);
    
    y_vec = Y_obs(:);

    %% -------------------- 3) Inversion (coarse) --------------------
    fprintf('--- Solving inverse problem (coarse mesh/time)...\n');

    modelCoarse = createpde(1);
    create_disk_geometry(modelCoarse, R);
    generateMesh(modelCoarse, 'Hmax', meshCoarseH);

    % Project initial q into feasible set
    q = project_q_disk_T1(q, srcMax*R, T_bounds);

    % History
    history.q    = zeros(maxIter+1, 3);   % [x1,x2,T1]
    history.c    = zeros(maxIter+1, 2);   % [c1,c2]
    history.cost = zeros(maxIter+1, 1);

    % Forward+fit handle: given q -> (Y_fit vec, cHat)
    forwardFit = @(qq) local_forward_fit_pwc( ...
        modelCoarse, qq, tCoarse, sensors, sigma, shrink, y_vec);

    % Init
    [Y_fit_vec, cHat] = forwardFit(q);
    r    = Y_fit_vec - y_vec;
    cost = 0.5*(r'*r);

    nIter = 0;
    history.q(1,:)  = q;
    history.c(1,:)  = cHat(:).';
    history.cost(1) = cost;

    fprintf('Iter %2d: cost=%.4e, q=(%.6f, %.6f, %.6f), c=(%.6f, %.6f)\n', ...
        0, cost, q(1), q(2), q(3), cHat(1), cHat(2));

    stopReason = "maxIter reached";

    if cost < tolCostAbs
        stopReason = "cost below tolCostAbs at init";
    else
        for k = 1:maxIter

            % ----- 3a) Central FD Jacobian of residual r(q) wrt (x1,x2,T1) -----
            epsStep = 1e-10;
            J = zeros(numel(r), 3);

            % === x1 ===
            q_plus  = project_q_disk_T1([q(1)+fdStepXY(1), q(2), q(3)], srcMax*R, T_bounds);
            q_minus = project_q_disk_T1([q(1)-fdStepXY(1), q(2), q(3)], srcMax*R, T_bounds);
            hx1_eff  = 0.5*(q_plus(1) - q_minus(1));

            if abs(hx1_eff) >= epsStep
                [Yp, ~] = forwardFit(q_plus);   rp = Yp - y_vec;
                [Ym, ~] = forwardFit(q_minus);  rm = Ym - y_vec;
                J(:,1) = (rp - rm) / (2*hx1_eff);
            else
                h_for = q_plus(1) - q(1);
                if abs(h_for) >= epsStep
                    [Yp, ~] = forwardFit(q_plus); rp = Yp - y_vec;
                    J(:,1) = (rp - r) / h_for;
                else
                    h_back = q(1) - q_minus(1);
                    if abs(h_back) >= epsStep
                        [Ym, ~] = forwardFit(q_minus); rm = Ym - y_vec;
                        J(:,1) = (r - rm) / h_back;
                    else
                        warning('FD step in x1 collapsed after projection. Setting J(:,1)=0.');
                        J(:,1) = zeros(size(r));
                    end
                end
            end

            % === x2 ===
            q_plus  = project_q_disk_T1([q(1), q(2)+fdStepXY(2), q(3)], srcMax*R, T_bounds);
            q_minus = project_q_disk_T1([q(1), q(2)-fdStepXY(2), q(3)], srcMax*R, T_bounds);
            hx2_eff  = 0.5*(q_plus(2) - q_minus(2));

            if abs(hx2_eff) >= epsStep
                [Yp, ~] = forwardFit(q_plus);   rp = Yp - y_vec;
                [Ym, ~] = forwardFit(q_minus);  rm = Ym - y_vec;
                J(:,2) = (rp - rm) / (2*hx2_eff);
            else
                h_for = q_plus(2) - q(2);
                if abs(h_for) >= epsStep
                    [Yp, ~] = forwardFit(q_plus); rp = Yp - y_vec;
                    J(:,2) = (rp - r) / h_for;
                else
                    h_back = q(2) - q_minus(2);
                    if abs(h_back) >= epsStep
                        [Ym, ~] = forwardFit(q_minus); rm = Ym - y_vec;
                        J(:,2) = (r - rm) / h_back;
                    else
                        warning('FD step in x2 collapsed after projection. Setting J(:,2)=0.');
                        J(:,2) = zeros(size(r));
                    end
                end
            end

            % === T1 (grid-crossing FD step) ===
            dt  = tCoarse(2) - tCoarse(1);
            eta = min(abs(tCoarse - q(3)));
            hT1 = max([1.25*eta, 2*dt, hT_min]);

            q_plus  = project_q_disk_T1([q(1), q(2), q(3)+hT1], srcMax*R, T_bounds);
            q_minus = project_q_disk_T1([q(1), q(2), q(3)-hT1], srcMax*R, T_bounds);
            hT_eff  = 0.5*(q_plus(3) - q_minus(3));

            if abs(hT_eff) >= epsStep
                [Yp, ~] = forwardFit(q_plus);   rp = Yp - y_vec;
                [Ym, ~] = forwardFit(q_minus);  rm = Ym - y_vec;
                J(:,3) = (rp - rm) / (2*hT_eff);
            else
                h_for = q_plus(3) - q(3);
                if abs(h_for) >= epsStep
                    [Yp, ~] = forwardFit(q_plus); rp = Yp - y_vec;
                    J(:,3) = (rp - r) / h_for;
                else
                    h_back = q(3) - q_minus(3);
                    if abs(h_back) >= epsStep
                        [Ym, ~] = forwardFit(q_minus); rm = Ym - y_vec;
                        J(:,3) = (r - rm) / h_back;
                    else
                        warning('FD step in T1 collapsed after clamping. Setting J(:,3)=0.');
                        J(:,3) = zeros(size(r));
                    end
                end
            end

            % ----- 3b) Regularized GN step + Armijo backtracking -----
            H    = J.'*J;
            grad = J.'*r;

            if norm(grad) < tolGrad
                stopReason = "gradient below tolGrad";
                break;
            end

            dq    = - (H + lambdaFixed*eye(3)) \ grad;
            slope = grad.' * dq;

            % Safety: if not descent, fallback to steepest descent
            if ~(isfinite(slope)) || slope >= 0
                warning('Gauss-Newton step not a descent direction; falling back to steepest descent.');
                dq    = -grad;
                slope = -grad.'*grad;
            end

            stepNorm = norm(dq);
            if stepNorm / max(1, norm(q)) < tolStepRel
                stopReason = "relative step below tolStepRel";
                break;
            end

            alpha = alphaInit;
            accepted = false;

            while alpha >= alphaMin
                q_try = project_q_disk_T1(q + (alpha*dq).', srcMax*R, T_bounds);

                [Y_try_vec, c_try] = forwardFit(q_try);
                r_try    = Y_try_vec - y_vec;
                cost_try = 0.5*(r_try.'*r_try);

                if cost_try <= cost + cArmijo*alpha*slope
                    % ACCEPT
                    q     = q_try;
                    cHat  = c_try;
                    r     = r_try;
                    cost  = cost_try;
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
            history.q(nIter+1,:)  = q;
            history.c(nIter+1,:)  = cHat(:).';
            history.cost(nIter+1) = cost;

            fprintf('Iter %2d: cost=%.4e, step=%.3e, q=(%.6f, %.6f, %.6f), c=(%.6f, %.6f)\n', ...
                nIter, cost, stepNorm, q(1), q(2), q(3), cHat(1), cHat(2));

            if cost < tolCostAbs
                stopReason = "cost below tolCostAbs";
                break;
            end
        end
    end

    % Trim history
    history.q    = history.q(1:nIter+1,:);
    history.c    = history.c(1:nIter+1,:);
    history.cost = history.cost(1:nIter+1);

    %% -------------------- 4) Report --------------------
    err.x1 = abs(q(1)    - truth.p(1));
    err.x2 = abs(q(2)    - truth.p(2));
    err.T1 = abs(q(3)    - truth.T1);
    err.c1 = abs(cHat(1) - truth.c1);
    err.c2 = abs(cHat(2) - truth.c2);

    fprintf('--- Final (EX03) ---\n');
    fprintf('Stop reason: %s\n', stopReason);
    fprintf('True: p=(%.4f, %.4f), T1=%.4f, c=(%.4f, %.4f)\n', ...
        truth.p(1), truth.p(2), truth.T1, truth.c1, truth.c2);
    fprintf('Est : p=(%.4f, %.4f), T1=%.4f, c=(%.4f, %.4f)\n', ...
        q(1), q(2), q(3), cHat(1), cHat(2));
    fprintf('Err : |Δx1|=%.2e, |Δx2|=%.2e, |ΔT1|=%.2e, |Δc1|=%.2e, |Δc2|=%.2e\n', ...
        err.x1, err.x2, err.T1, err.c1, err.c2);

    %% -------------------- 5) Plots --------------------
    % Recompute final fitted Y for plotting (Nt x Ns)
    [Y_fit_vec, ~] = forwardFit(q);
    Y_fit = reshape(Y_fit_vec, numel(tCoarse), Ns);

    % --------- flux traces ---------
    fs    = 16;     % axes font size
    lw    = 1.3;    % line width
    ms    = 8;      % marker size
    legfs = fs - 2;

    figure;
    for s = 1:2
        subplot(2,1,s);

        plot(tCoarse, Y_obs(:,s), 'k.', 'MarkerSize', ms-2); hold on;
        plot(tCoarse, Y_fit(:,s), '-',  'LineWidth', lw);

        xlabel('t', 'FontSize', fs);
        legend({'obs','fit'}, 'Location','best', 'FontSize', legfs);

        grid on; box on;
        set(gca, 'FontSize', fs);
    end

    % --------- geometry ---------
    figure;
    theta = linspace(0, 2*pi, 400);

    plot(R*cos(theta), R*sin(theta), 'k-', 'LineWidth', lw); hold on;
    plot(truth.p(1), truth.p(2), 'ko', 'MarkerFaceColor','g', 'MarkerSize', ms);
    plot(q(1), q(2), 'ks', 'MarkerFaceColor','r', 'MarkerSize', ms-4);
    plot(sensors(:,1), sensors(:,2), 'k^', 'MarkerFaceColor','b', 'MarkerSize', ms-1);

    axis equal;
    xlim([-R, R]); ylim([-R, R]);
    grid on; box on;

    xlabel('x_1', 'FontSize', fs);
    ylabel('x_2', 'FontSize', fs);
    set(gca, 'FontSize', fs);

    legend({'boundary','true p','est p','sensors'}, 'Location', 'best', 'FontSize', legfs);

    %% -------------------- 6) Pack result --------------------
    result.name        = "EX03_disc_pwc_g";
    result.noise_level = noise_level;

    result.truth   = truth;
    result.est.q   = q;
    result.est.c   = cHat(:).';
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
    settings.fdStepXY    = fdStepXY;
    settings.hT_min      = hT_min;
    settings.alphaInit   = alphaInit;
    settings.alphaMin    = alphaMin;
    settings.rho         = rho;
    settings.cArmijo     = cArmijo;

    result.settings   = settings;
    result.stopReason = stopReason;
    result.nIter      = nIter;
end

function [Y1, Y2] = local_forward_flux_bases_pwc(model, p_source, T1, tlist, sensors, sigma, shrink)
% Compute bases for g1=1_{t<=T1}, g2=1_{t>T1}
    r1 = solve_heat_fem_pwc_basis(model, p_source, T1, tlist, sigma, 1);
    Y1 = extract_flux_disk(r1, sensors, shrink);

    r2 = solve_heat_fem_pwc_basis(model, p_source, T1, tlist, sigma, 2);
    Y2 = extract_flux_disk(r2, sensors, shrink);
end

function [Y_fit_vec, cHat] = local_forward_fit_pwc(model, q, tlist, sensors, sigma, shrink, y_vec)
% Given q=[x1,x2,T1], compute (Y_fit, cHat) with plain LS (NO ridge)
    p  = q(1:2);
    T1 = q(3);

    [Y1, Y2] = local_forward_flux_bases_pwc(model, p, T1, tlist, sensors, sigma, shrink);

    Phi = [Y1(:), Y2(:)];

    % Plain least squares (prefer min-norm if available)
    if exist('lsqminnorm','file') == 2
        cHat = lsqminnorm(Phi, y_vec);
    else
        cHat = Phi \ y_vec;
    end

    Y_fit_vec = Phi * cHat;
end