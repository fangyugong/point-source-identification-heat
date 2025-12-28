function result = solve_heat_fem_pwc_basis(model, p_source, T1, tlist, sigma, which_g)
% Solve u_t - Δu = g(t)*Gaussian(x-p), with g(t)=1_{t<=T1} or 1_{t>T1}
    c = 1; a = 0; d = 1;
    f = @(location, state) local_source_pwc(location, state, p_source, T1, sigma, which_g);

    specifyCoefficients(model, 'm', 0, 'd', d, 'c', c, 'a', a, 'f', f);
    applyBoundaryCondition(model, 'dirichlet', 'Edge', 1:model.Geometry.NumEdges, 'u', 0);
    setInitialConditions(model, 0);

    result = solvepde(model, tlist);
end

function fval = local_source_pwc(location, state, p_source, T1, sigma, which_g)
% g(t) * Gaussian(x-p), handle NaN time queries from toolbox
    x  = location.x;
    y  = location.y;
    nr = numel(x);

    t = state.time;
    if isnan(t) || any(isnan(t))
        fval = nan(1, nr);
        return;
    end

    dx  = x - p_source(1);
    dy  = y - p_source(2);
    phi = (1/(2*pi*sigma^2)) .* exp(-(dx.^2 + dy.^2) / (2*sigma^2));

    if which_g == 1
        g_t = double(t <= T1);
    else
        g_t = double(t >  T1);
    end

    fval = g_t .* phi;
end