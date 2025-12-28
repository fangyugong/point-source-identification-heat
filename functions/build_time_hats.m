function basis = build_time_hats(tlist, K)
% Build hat basis on [0,T] sampled on tlist, with K knots
    T = tlist(end);
    knots = linspace(0, T, K).';
    Nt = numel(tlist);

    B = zeros(Nt, K);
    for k = 1:K
        if k == 1
            idx = tlist <= knots(2);
            B(idx,k) = max(0, 1 - (tlist(idx)-knots(1))/(knots(2)-knots(1)));
        elseif k == K
            idx = tlist >= knots(K-1);
            B(idx,k) = max(0, 1 - (knots(K)-tlist(idx))/(knots(K)-knots(K-1)));
        else
            L = knots(k-1); C = knots(k); R = knots(k+1);
            left  = (tlist>=L) & (tlist<=C);
            right = (tlist>=C) & (tlist<=R);
            B(left ,k) = (tlist(left) - L)/(C-L);
            B(right,k) = (R - tlist(right))/(R-C);
        end
    end

    basis.B     = B;
    basis.knots = knots;
end