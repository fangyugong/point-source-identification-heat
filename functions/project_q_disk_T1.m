function q_proj = project_q_disk_T1(q, Rmax, T_bounds)
% Project (x1,x2) into disk of radius Rmax, clamp T1 into [Tmin,Tmax]
    x1 = q(1); x2 = q(2); T1 = q(3);

    r = hypot(x1,x2);
    if r > Rmax
        s = Rmax / max(r, eps);
        x1 = x1 * s; x2 = x2 * s;
    end

    T1 = min(max(T1, T_bounds(1)), T_bounds(2));

    q_proj = [x1, x2, T1];
end