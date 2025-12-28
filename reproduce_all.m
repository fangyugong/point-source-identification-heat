function reproduce_all()
%REPRODUCE_ALL Run all numerical experiments and print tables matching the paper.

    clc; 
    close all;
    baseDir = fileparts(mfilename('fullpath'));
    addpath(fullfile(baseDir, 'functions'));

    fprintf('============================================================\n');
    fprintf('REPRODUCING ALL NUMERICAL EXPERIMENTS\n');
    fprintf('============================================================\n');
    fprintf('This script will run Examples 1-4 sequentially.\n');
    fprintf('Iteration details will appear above (from the solvers).\n');
    fprintf('Summary tables (matching the paper) will appear below each example.\n\n');
    
    %% ============================================================
    %  Table 1: Example 1 (Unit Disk, Constant g)
    % ============================================================
    print_separator('Example 1 (Table 1 Replication)');
    
    noises_ex1 = [0.03, 0.05, 0.10];
    results_ex1 = [];
    
    % Run experiments
    for d = noises_ex1
        fprintf('\n[Running Ex01 with noise %.0f%%]...\n', d*100);
        % This will print iteration details
        res = main_ex01_disk_const(d); 
        results_ex1 = [results_ex1; res]; %#ok<AGROW>
    end
    
    % Print Table 1
    fprintf('\nTable 1: Recovery of location (x1,x2) and amplitude g for Example 1\n');
    fprintf('------------------------------------------------------------------\n');
    fprintf('%-10s %-12s %-12s %-12s\n', '', 'x1', 'x2', 'g');
    fprintf('------------------------------------------------------------------\n');
    
    % Exact Row
    t = results_ex1(1).truth;
    fprintf('%-10s %-12.4f %-12.4f %-12.4f\n', 'Exact', t.p(1), t.p(2), t.g);
    fprintf('------------------------------------------------------------------\n');
    
    % Data Rows
    for i = 1:length(results_ex1)
        r = results_ex1(i);
        lbl = sprintf('%.0f%%', r.noise_level*100);
        
        % Estimate row
        fprintf('%-10s %-12.4f %-12.4f %-12.4f (estimate)\n', ...
            lbl, r.est.p(1), r.est.p(2), r.est.g);
        
        % Error row
        fprintf('%-10s %-12.2e %-12.2e %-12.2e (error)\n', ...
            '', r.errors.x1, r.errors.x2, r.errors.g);
        
        if i < length(results_ex1), fprintf('\n'); end
    end
    fprintf('------------------------------------------------------------------\n\n');

    %% ============================================================
    %  Table 2: Example 2 (Known Time-Dependent g)
    % ============================================================
    print_separator('Example 2 (Table 2 Replication)');
    
    noises_ex2 = [0.03, 0.05, 0.10];
    results_ex2a = [];
    results_ex2b = [];
    
    fprintf('\n[Running Ex02a (Disk)]...\n');
    for d = noises_ex2
        fprintf('  -> Noise %.0f%%\n', d*100);
        results_ex2a = [results_ex2a; main_ex02a_disk_known_g(d)]; %#ok<AGROW>
    end
    
    fprintf('\n[Running Ex02b (Ellipse)]...\n');
    for d = noises_ex2
        fprintf('  -> Noise %.0f%%\n', d*100);
        results_ex2b = [results_ex2b; main_ex02b_ellipse_known_g(d)]; %#ok<AGROW>
    end
    
    % Print Table 2
    fprintf('\nTable 2: Recovery of location (x1,x2) for Example 2\n');
    fprintf('--------------------------------------------------------------------------\n');
    fprintf('%-10s | %-25s | %-25s\n', '', 'Case (a) Disk', 'Case (b) Ellipse');
    fprintf('%-10s | %-12s %-12s | %-12s %-12s\n', '', 'x1', 'x2', 'x1', 'x2');
    fprintf('--------------------------------------------------------------------------\n');
    
    % Exact Row
    ta = results_ex2a(1).truth.p;
    tb = results_ex2b(1).truth.p;
    fprintf('%-10s | %-12.4f %-12.4f | %-12.4f %-12.4f\n', 'Exact', ta(1), ta(2), tb(1), tb(2));
    fprintf('--------------------------------------------------------------------------\n');
    
    % Data Rows
    for i = 1:length(noises_ex2)
        ra = results_ex2a(i);
        rb = results_ex2b(i);
        lbl = sprintf('%.0f%%', ra.noise_level*100);
        
        fprintf('%-10s | %-12.4f %-12.4f | %-12.4f %-12.4f (est)\n', ...
            lbl, ra.est.p(1), ra.est.p(2), rb.est.p(1), rb.est.p(2));
        
        fprintf('%-10s | %-12.2e %-12.2e | %-12.2e %-12.2e (err)\n', ...
            '', ra.errors.x1, ra.errors.x2, rb.errors.x1, rb.errors.x2);
            
        if i < length(noises_ex2), fprintf('\n'); end
    end
    fprintf('--------------------------------------------------------------------------\n\n');

    %% ============================================================
    %  Table 3: Example 3 (Piecewise Constant g)
    % ============================================================
    print_separator('Example 3 (Table 3 Replication)');
    
    noises_ex3 = [0.01, 0.03, 0.05];
    results_ex3 = [];
    
    for d = noises_ex3
        fprintf('\n[Running Ex03 with noise %.0f%%]...\n', d*100);
        results_ex3 = [results_ex3; main_ex03_disk_pwc(d)]; %#ok<AGROW>
    end
    
    % Print Table 3
    fprintf('\nTable 3: Results for Example 3 (PWC g)\n');
    fprintf('------------------------------------------------------------------------------\n');
    fprintf('%-10s %-12s %-12s %-12s %-12s %-12s\n', '', 'x1', 'x2', 'T1', 'c1', 'c2');
    fprintf('------------------------------------------------------------------------------\n');
    
    t = results_ex3(1).truth;
    fprintf('%-10s %-12.4f %-12.4f %-12.4f %-12.4f %-12.4f\n', ...
        'Exact', t.p(1), t.p(2), t.T1, t.c1, t.c2);
    fprintf('------------------------------------------------------------------------------\n');
    
    for i = 1:length(results_ex3)
        r = results_ex3(i);
        lbl = sprintf('%.0f%%', r.noise_level*100);
        
        fprintf('%-10s %-12.4f %-12.4f %-12.4f %-12.4f %-12.4f (est)\n', ...
            lbl, r.est.q(1), r.est.q(2), r.est.q(3), r.est.c(1), r.est.c(2));
            
        fprintf('%-10s %-12.2e %-12.2e %-12.2e %-12.2e %-12.2e (err)\n', ...
            '', r.errors.x1, r.errors.x2, r.errors.T1, r.errors.c1, r.errors.c2);
            
        if i < length(results_ex3), fprintf('\n'); end
    end
    fprintf('------------------------------------------------------------------------------\n\n');

    %% ============================================================
    %  Table 4: Example 4 (Unknown g, Ellipse)
    % ============================================================
    print_separator('Example 4 (Table 4 Replication)');
    
    noises_ex4 = [0.01, 0.03, 0.05];
    results_ex4 = [];
    
    for d = noises_ex4
        fprintf('\n[Running Ex04 with noise %.0f%%]...\n', d*100);
        results_ex4 = [results_ex4; main_ex04_ellipse_unknown(d)]; %#ok<AGROW>
    end
    
    % Print Table 4
    fprintf('\nTable 4: Results for Example 4 (Unknown g, Ellipse)\n');
    fprintf('----------------------------------------------------------\n');
    fprintf('%-10s %-12s %-12s %-12s\n', '', 'x', 'y', 'e(g)');
    fprintf('----------------------------------------------------------\n');
    
    t = results_ex4(1).truth;
    fprintf('%-10s %-12.4f %-12.4f %-12s\n', 'Exact', t.p(1), t.p(2), '---');
    fprintf('----------------------------------------------------------\n');
    
    for i = 1:length(results_ex4)
        r = results_ex4(i);
        lbl = sprintf('%.0f%%', r.noise_level*100);
        
        fprintf('%-10s %-12.4f %-12.4f %-12.2e (est)\n', ...
            lbl, r.est.p(1), r.est.p(2), r.errors_g.relL2);
            
        fprintf('%-10s %-12.2e %-12.2e %-12s (err)\n', ...
            '', r.errors.x1, r.errors.x2, '---');
            
        if i < length(results_ex4), fprintf('\n'); end
    end
    fprintf('----------------------------------------------------------\n\n');
    
    fprintf('All tables generated.\n');
end

function print_separator(titleStr)
    fprintf('\n');
    fprintf('############################################################\n');
    fprintf(' %s \n', titleStr);
    fprintf('############################################################\n');
end