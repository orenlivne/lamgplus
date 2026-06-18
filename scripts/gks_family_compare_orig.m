% gks_family_compare_orig.m  —  Part 2 of 3 (Octave / original-LAMG side).
%
% Runs the ORIGINAL MATLAB LAMG 2.2.1 on the SAME (A,b) operators exported by Part 1, using
% gks2023's EXACT invocation: setup('laplacian',L) for graph Laplacians (their matlab/timeLamg.m)
% and setup('sdd',M) for the SDDM system (their matlab/timeLamgSddm.m), both solved with
% 'errorReductionTol',1e-8 and LAMG's default options (notably the default numCycles cap). The only
% thing we do NOT replicate is their AC-relative subprocess timeout, which records a killed MATLAB
% process as non-convergence; we let the solver run to its own stopping criterion.
%
% LAMG 2.2.1 is run under GNU Octave (Livne's 2012 release made Octave-runnable; algorithm files
% byte-identical to the pristine release).
%
% Run:  LAMG_ORIG_DIR=/path/to/lamg-octave SCRATCH=/path/to/results/gks_cmp \
%         octave-cli --no-gui scripts/gks_family_compare_orig.m
more off; warning('off','all');
orig    = getenv('LAMG_ORIG_DIR'); if isempty(orig);    orig    = '/Users/oren/code/mg/maxflow/lamg-octave'; end
scratch = getenv('SCRATCH');       if isempty(scratch); scratch = fullfile(fileparts(mfilename('fullpath')),'..','results','gks_cmp'); end
cd(orig);
addpath(genpath('lamg/main')); addpath(genpath('core/main')); addpath(genpath('graph/main'));
addpath('core/util'); addpath('lin-api/util'); addpath('lin-solve'); addpath('lin-api');

% --- read manifest.csv (family,name,n,m,mode) ---
fid = fopen(fullfile(scratch,'manifest.csv')); fgetl(fid);   % skip header
rows = {};
while true
  ln = fgetl(fid); if ~ischar(ln); break; end
  parts = strsplit(ln, ','); rows(end+1,:) = parts(1:5); %#ok
end
fclose(fid);

out = fopen(fullfile(scratch,'orig.csv'), 'w');
fprintf(out, 'name,orig_cycles,orig_relres,orig_success,orig_acf\n');
fprintf('\n%-16s | original LAMG 2.2.1 (gks2023 call)\n', 'name');
fprintf('%-16s |  mode  cycles  relres     success  acf\n', '');
fprintf('%s\n', repmat('-',1,64));

for r = 1:size(rows,1)
  name = rows{r,2}; mode = rows{r,5};
  dl = dlmread(fullfile(scratch, [name '_A.txt']));
  A  = sparse(dl(:,1), dl(:,2), dl(:,3));
  b  = dlmread(fullfile(scratch, [name '_b.txt']));
  cyc = Inf; rel = Inf; success = 0; acf = NaN;     % gks2023's failure defaults
  try
    solver = Solvers.newSolver('lamg','randomSeed',1);
    if strcmp(mode,'sdd'); s = solver.setup('sdd', A); else; s = solver.setup('laplacian', A); end
    [x, success, ~, details] = solver.solve(s, b, 'errorReductionTol', 1e-8);
    cyc = length(details.stats.errorNormHistory);
    rel = norm(A*x - b)/norm(b);
    if isfield(details,'acf'); acf = details.acf; end
  catch err
    fprintf('  (%s errored: %s)\n', name, err.message);
  end
  fprintf('%-16s |  %-4s  %5d   %.2e    %d     %.3f\n', name, mode, cyc, rel, success, acf);
  fprintf(out, '%s,%d,%.6e,%d,%.6f\n', name, cyc, rel, success, acf);
end
fclose(out);
fprintf('\nWrote orig.csv to %s\n', scratch);
