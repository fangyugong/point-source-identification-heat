function p_proj = project_to_ellipse(p, aAxis, bAxis, srcMax)
% Project point to ellipse interior in "ellipse radius" metric:
%   s = sqrt((x1/a)^2+(x2/b)^2) <= srcMax
    x1 = p(1); x2 = p(2);
    s = sqrt((x1/aAxis)^2 + (x2/bAxis)^2);
    if s > srcMax
        scale = srcMax / max(s, eps);
        x1 = x1 * scale;
        x2 = x2 * scale;
    end
    p_proj = [x1, x2];
end