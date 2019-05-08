function optimParams(model, N, objective, noise, optimiter)
%% finds model parameters that optimize objective
%
% model is one supported by simulateDiffusion(). N is the number of
% options.
%
% If optimiter (optional) is given, then its number is appended to the
% output file, and a single 're'start is used. In this case, the MATLAB
% rng is seeded with optimiter. Otherwise, it is seeded for each restart
% with the current restart number.
%
% This function does not fit non-linearities of the UCRM & CRM variants,
% but instead fixed is to 1.5 (which seems to work well).
%
% objective is
% 'RR' - maximize reward rate
% 'CR' - maximize correct rate

model = upper(model);
objective = upper(objective);
if ~any(strcmp(objective, {'RR', 'CR'}))
    error('Unknown objective %s', objective);
end
nooptimiter = nargin < 5;
if nargin < 4, noise = 0; end

% adjust base parameters
addpath('../shared/','../shared/bads-1.0.4/');
p                   = baseParameters();
p.sim.maxt          = 1;             % to save time, since max(RT) <= 0.6
p.task.N            = N;
p.model.a           = 1.5;
p.sim.nTrial        = 2e5;           % set to 1e6 if curves aren't smooth
p.task.meanZ        = ones(1,p.task.N);
p.task.covX         = eye(p.task.N);
p.task.covZ         = eye(p.task.N);
p.task.icovZ        = eye(p.task.N);
p.task.icovX        = eye(p.task.N);
p.optim.nRestarts   = 10;
p                   = baseParameters(p);
%p.task.covX         = p.task.covX / (5e3);
%[Z, z1BinBounds, z2, z3BinBounds] = drawZ([25 35], 30, [0 30], 12, 8, p.sim.nTrial);

% model-specific settings
switch model
    case {'RM', 'NRM'}              % theta = [u0]
        paramNames = {'u0', 'sigH'};
        lb  = [-1 -10];         ub  = [1 10];
        plb = [ 0  -3];         pub = [1  5];
    case {'URM', 'UNRM'}            % theta = [u0 b]
        paramNames = {'u0', 'b', 'sigH'};
        lb  = [-1   0 -10];     ub  = [1 10 10];
        plb = [-0.5 0  -3];     pub = [1  3  5];
    case {'UCRM','UCRM_PD','UCRM_DP'}          % theta = [u0 b]
        paramNames = {'u0', 'b', 'sigH'};
        lb  = [-1   -0.1 -10];      ub  = [1 10 10];
        plb = [-0.5 -0.1  -3];      pub = [1  3  5];
    case {'CRM'}
        paramNames = {'u0', 'sigH'};
        lb  = [-1   -10];      ub  = [1 10];
        plb = [-0.5  -3];      pub = [1  5];
    otherwise
        error('Unknown model %s', model);
end

% perform optimization with random restarts
timer = tic;
nRestarts = p.optim.nRestarts;
if nooptimiter
    fprintf('Optimizing model %s with objective %s for %d options\n', model, objective, N);
else
    fprintf('Optimizing model %s with objective %s for %d options, optimizer iteration %d\n', ...
        model, objective, N, optimiter);
    nRestarts = 1;
    rng(optimiter);
end
fprintf('Using %d random restart(s)\n\n', nRestarts);
thetai = NaN(nRestarts, length(lb));
obji = NaN(nRestarts, 1);
objfun = @(theta) evaluateObjective(theta, paramNames, model, objective, p, noise);
for iRestart = 1:nRestarts
    fprintf('Restart %d, ', iRestart);  toc(timer);
    if nooptimiter
        rng(iRestart);
    end
    theta0 = plb + rand(1, length(plb)) .* (pub - plb);
    [thetai(iRestart, :), obji(iRestart)] = ...
        bads(objfun,theta0,lb,ub,plb,pub,[],p.optim.opt);
    obji(iRestart) = -obji(iRestart);
    fprintf('Found optimum at %s=%f with parameters', objective, obji(iRestart));
    disp(thetai(iRestart, :));
    fprintf('\n');
end

% output & save results
p.optim.lb = lb;
p.optim.ub = ub;
p.optim.plb = plb;
p.optim.pub = pub;
p.optim.paramNames = paramNames;
p.optim.obji = obji;
p.optim.thetai = thetai;
[optobj, iopt] = max(obji);
opttheta = thetai(iopt,:);
fprintf('\nFound overall optimum %s=%f with parameters', objective, optobj);
disp(opttheta);
if nooptimiter
    outname = ['optimParams' filesep model '_' objective '_N' num2str(N) 'noise' num2str(noise) '_fitSigH.mat'];
else
    outname = ['optimParams' filesep model '_' objective '_N' num2str(N) 'noise' num2str(noise) ...
        '_iter' num2str(optimiter) '.mat'];
end
save(outname, 'p', 'model', 'objective', 'N', 'paramNames', 'optobj', 'opttheta');
fprintf('Results written to %s\n', outname);
toc(timer);


function o = evaluateObjective(theta, paramNames, model, objective, p, noise)
% objective function for optimization
for itheta = 1:length(theta)
    p.model.(paramNames{itheta}) = theta(itheta);
end
perf = simulateDiffusion(p, model, noise);
switch objective
    case 'RR'
        o = -perf.RR;
    case 'CR'
        o = -perf.CR;
end 