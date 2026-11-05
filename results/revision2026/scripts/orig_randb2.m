% Original LAMG 2.2.1 (Octave) on web-Stanford / delaunay_n18 / n17 with the SAME random b as LAMG+.
more off; warning('off','all');
orig = '/Users/oren/code/old/maxflow/lamg-octave'; cd(orig);
addpath(genpath('lamg/main')); addpath(genpath('core/main')); addpath(genpath('graph/main'));
addpath('core/util'); addpath('lin-api/util'); addpath('lin-solve'); addpath('lin-api');
D = getenv('OUTDIR');
names = {'webStanford','delaunay17','delaunay18'};
for r = 1:numel(names)
  name = names{r};
  dl = dlmread(fullfile(D, [name '_A.txt'])); A = sparse(dl(:,1), dl(:,2), dl(:,3)); clear dl;
  b = dlmread(fullfile(D, [name '_b.txt']));
  t0 = tic; solver = Solvers.newSolver('lamg','randomSeed',1); s = solver.setup('laplacian', A); ts = toc(t0);
  t0 = tic; [x, success, ~, details] = solver.solve(s, b, 'errorReductionTol', 1e-10); tsol = toc(t0);
  h = details.stats.errorNormHistory; cyc = numel(h);
  rel = norm(A*x - b)/norm(b);
  f = h(2:end)./h(1:end-1); tail = f(max(1,end-4):end);
  fprintf('%-12s n=%d setup=%.1fs solve=%.1fs cycles=%d relres=%.2e success=%d last5=%s geo=%.3f\n', ...
          name, size(A,1), ts, tsol, cyc, rel, success, mat2str(tail',3), exp(mean(log(tail))));
  fprintf('   errorNormHistory log10: %s\n', mat2str(log10(h'/h(1)),2));
  fflush(stdout);
end
