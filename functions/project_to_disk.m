function p_proj = project_to_disk(p, Rmax)
% Project point to disk of radius Rmax
    r = norm(p);
    if r > Rmax
        p_proj = p * (Rmax / r);
    else
        p_proj = p;
    end
end