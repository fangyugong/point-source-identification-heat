function create_disk_geometry(model, R)
% Create circle geometry (disk of radius R)
    gd = [1; 0; 0; R];   % Circle: [1; center_x; center_y; radius]
    ns = char('C');
    sf = 'C';
    [dl, ~] = decsg(gd, sf, ns);
    geometryFromEdges(model, dl);
end