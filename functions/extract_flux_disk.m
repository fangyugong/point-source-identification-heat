function flux = extract_flux_disk(result, sensors, shrink)
% Evaluate normal flux near boundary: ∂u/∂n ≈ ∇u(xq)·n, xq slightly inside
    Nt = size(result.NodalSolution, 2);
    Ns = size(sensors, 1);
    flux = zeros(Nt, Ns);
    tIdx = 1:Nt;

    for s = 1:Ns
        xb = sensors(s,1);
        yb = sensors(s,2);

        xq = shrink * xb;
        yq = shrink * yb;

        % On unit circle: outward unit normal is (xb, yb) since hypot(xb,yb)=1
        nrm = max(hypot(xb,yb), eps);
        nx  = xb / nrm;
        ny  = yb / nrm;

        [Gx, Gy] = evaluateGradient(result, xq, yq, tIdx);
        Gx = Gx(:); Gy = Gy(:);

        vals = Gx*nx + Gy*ny;

        % NaN handling
        good = ~isnan(vals);
        nBad = sum(~good);
        frac = nBad / Nt;

        if ~any(good)
            warning('Sensor %d: all NaN flux at (%.3f, %.3f). Set to zero.', s, xq, yq);
            vals(:) = 0;
        elseif sum(good) == 1
            warning('Sensor %d: only one non-NaN flux sample. Constant extrapolation.', s);
            vals(:) = vals(find(good,1));
        else
            if frac > 0.05
                firstBad = find(~good, 1, 'first');
                warning('Sensor %d: NaN ratio %.1f%% at (%.3f, %.3f). First bad idx=%d.', ...
                    s, 100*frac, xq, yq, firstBad);
            end
            idx  = find(good);
            vals = interp1(idx, vals(good), 1:Nt, 'pchip', 'extrap').';
        end

        flux(:,s) = vals;
    end
end