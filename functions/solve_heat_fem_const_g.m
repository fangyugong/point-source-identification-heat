function result = solve_heat_fem_const_g(model, p_source, g_val, tlist, sigma)
% Solve u_t - Δu = g_val * Gaussian(x - p_source), u=0 on boundary, u0=0
    c = 1; a = 0; d = 1;

    f = @(location, state) g_val .* ...
        (1/(2*pi*sigma^2)) .* exp(-((location.x - p_source(1)).^2 + ...
                                   (location.y - p_source(2)).^2) ...
                                   /(2*sigma^2));

    specifyCoefficients(model, 'm', 0, 'd', d, 'c', c, 'a', a, 'f', f);
    applyBoundaryCondition(model, 'dirichlet', 'Edge', 1:model.Geometry.NumEdges, 'u', 0);
    setInitialConditions(model, 0);
    result = solvepde(model, tlist);
end