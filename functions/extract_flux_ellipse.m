function flux = extract_flux_ellipse(result, sensorAngles, aAxis, bAxis, shrink)
% Evaluate near-boundary normal flux on ellipse:
%   n ∝ ∇[(x/a)^2+(y/b)^2] = (2x/a^2, 2y/b^2), normalize.
% Query point is shrink*(xb,yb) to avoid NaNs.
    Nt = size(result.NodalSolution, 2);
    Ns = numel(sensorAngles);
    flux = zeros(Nt, Ns);
    tIdx = 1:Nt;

    for s = 1:Ns
        th = sensorAngles(s);

        xb = aAxis * cos(th);
        yb = bAxis * sin(th);

        xq = shrink * xb;
        yq = shrink * yb;

        nx_raw = xb / (aAxis^2);
        ny_raw = yb / (bAxis^2);
        nrm    = max(hypot(nx_raw, ny_raw), eps);
        nx     = nx_raw / nrm;
        ny     = ny_raw / nrm;

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