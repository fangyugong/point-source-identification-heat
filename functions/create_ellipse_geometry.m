function create_ellipse_geometry(model, aAxis, bAxis)
% Create ellipse geometry via decsg: ellipse centered at (0,0), axes a,b, no rotation.
    E1 = [4; 0; 0; aAxis; bAxis; 0; 0; 0; 0; 0];
    gd = E1;
    sf = 'E1';
    ns = 'E1';
    [dl, ~] = decsg(gd, sf, ns');
    geometryFromEdges(model, dl);
end