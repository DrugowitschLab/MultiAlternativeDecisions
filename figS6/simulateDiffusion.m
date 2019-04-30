function perf = simulateDiffusion(p, model, onlySummary, Z)
% simulates diffusion model and returns performance summary
%
% p.task  - task parameters
% p.sim   - simulation parameters
% p.model - model parameters
% r       - stochastic evidence
% model   - model string
%
% The following models are supported
% RM      - race model
%           p.model.u0: initial offset
% URM     - race model & urgency
%           p.model.u0: initial offset
%           p.model.b : urgency slope
% NRM     - normalized race model + linear normalization
%           p.model.u0: post-normalization mean activity
% UNRM    - normalized race model, linear normalization & urgency
%           p.model.u0: post-normalization initial mean activity
%           p.model.b : urgency slope
% CRM     - race model & constraint accumulation
%           p.model.u0: initial offset
%           p.model.a : power of the non-linearity
%           p.model.nIterConstraint: max. number of constraint iterations
%           p.model.iterAlpha: learning rate of constraint interations
% CRM2    - race model & constraint, non-linear accumulation
%           same parameters as CRM
% UCRM    - race model, urgency & constraint accumulation
%           p.model.u0: initial offset
%           p.model.b : urgency slope
%           p.model.a : power of the non-linearity
%           p.model.nIterConstraint: max. number of constraint iterations
%           p.model.iterAlpha: learning rate of constraint interations
% UCRM2   - race model, urgency & constraint, non-linear accumulation
%           same parameters are UCRM
% RANDOM  - random, immediate choices
%
% All models (except RANDOM) support integrator noise, with parameters
% p.model.intNoise: noise magnitude
% p.model.infFano: noise Fano factor
%
% If onlySummary (optional) is given and true, only summary statistics are
% returned, and per-trial statistics are omitted.
%
% If Z (optimal) is given, then it is expected to be an trials x N matrix
% that contains the latent state values for each trial. If it isn't given,
% it is drawn from a multivariate Gaussian per trial with means
% p.task.meanZ, and covariance p.task.covZ.

if nargin < 3, onlySummary = false; end
if nargin < 4
    Z = mvnrnd(p.task.meanZ, p.task.covZ, p.sim.nTrial);
end

cholcovXdt = chol(p.sim.dt * p.task.covX);

switch upper(model)
    case 'RM'      % standard race model
        if p.model.intNoise > 0
            [c, dt] = simulateDiffusionRTC(0, ...
                [p.model.u0 0.0 p.task.threshold p.sim.dt p.sim.maxt], ...
                Z, cholcovXdt, p.model.intNoise, p.model.intFano);
        else
            [c, dt] = simulateDiffusionRTC(0, ...
                [p.model.u0 0.0 p.task.threshold p.sim.dt p.sim.maxt], ...
                Z, cholcovXdt);
        end
        perf = performance(p, Z, c, dt, onlySummary);
        
    case 'URM'     % race model + urgency
        if p.model.intNoise > 0
            [c, dt] = simulateDiffusionRTC(0, ...
                [p.model.u0 p.model.b p.task.threshold p.sim.dt p.sim.maxt], ...
                Z, cholcovXdt, p.model.intNoise, p.model.intFano);                
        else
            [c, dt] = simulateDiffusionRTC(0, ...
                [p.model.u0 p.model.b p.task.threshold p.sim.dt p.sim.maxt], ...
                Z, cholcovXdt);                
        end
        perf = performance(p, Z, c, dt, onlySummary);
        
    case 'NRM'     % normalized race model
        if p.model.intNoise > 0
            [c, dt] = simulateDiffusionRTC(1, ...
                [p.model.u0 0.0 p.task.threshold p.sim.dt p.sim.maxt], ...
                Z, cholcovXdt, p.model.intNoise, p.model.intFano);                
        else
            [c, dt] = simulateDiffusionRTC(1, ...
                [p.model.u0 0.0 p.task.threshold p.sim.dt p.sim.maxt], ...
                Z, cholcovXdt);                
        end
        perf = performance(p, Z, c, dt, onlySummary);
        
    case 'UNRM'    % normalized race model & urgency
        if p.model.intNoise > 0
            [c, dt] = simulateDiffusionRTC(1, ...
                [p.model.u0 p.model.b p.task.threshold p.sim.dt p.sim.maxt], ...
                Z, cholcovXdt, p.model.intNoise, p.model.intFano);                
        else
            [c, dt] = simulateDiffusionRTC(1, ...
                [p.model.u0 p.model.b p.task.threshold p.sim.dt p.sim.maxt], ...
                Z, cholcovXdt);                
        end
        perf = performance(p, Z, c, dt, onlySummary);
        
    case {'CRM','CRM2'}    % constrained race model, linear/non-linear accum
        if strcmpi(model, 'CRM') == 1
            modelid = 2;  % linear accumulation
        else
            modelid = 3;  % nonlinear accumulation
        end
        if p.model.intNoise > 0
            [c, dt] = simulateDiffusionRTC(modelid, ...
                [p.model.u0 0.0 p.model.a p.task.threshold ...
                 p.model.iterAlpha p.model.nIterConstraint p.sim.dt p.sim.maxt], ...
                Z, cholcovXdt, p.model.intNoise, p.model.intFano);
        else
            [c, dt] = simulateDiffusionRTC(modelid, ...
                [p.model.u0 0.0 p.model.a p.task.threshold ...
                 p.model.iterAlpha p.model.nIterConstraint p.sim.dt p.sim.maxt], ...
                Z, cholcovXdt);
        end
        perf = performance(p, Z, c, dt, onlySummary);

    case {'UCRM','UCRM2'}    % constrained race model & urgency, lin/nonlin accum
        if strcmpi(model, 'UCRM') == 1
            modelid = 2;  % linear accumulation
        else
            modelid = 3;  % nonlinear accumulation
        end
        if p.model.intNoise > 0
            [c, dt] = simulateDiffusionRTC(modelid, ...
                [p.model.u0 p.model.b p.model.a p.task.threshold ...
                 p.model.iterAlpha p.model.nIterConstraint p.sim.dt p.sim.maxt], ...
                Z, cholcovXdt, p.model.intNoise, p.model.intFano);
        else
            [c, dt] = simulateDiffusionRTC(modelid, ...
                [p.model.u0 p.model.b p.model.a p.task.threshold ...
                 p.model.iterAlpha p.model.nIterConstraint p.sim.dt p.sim.maxt], ...
                Z, cholcovXdt);
        end
        perf = performance(p, Z, c, dt, onlySummary);
        
    case 'RANDOM'  % random, immediate choices
        perf.RT      = zeros(p.sim.nTrial, 1);
        perf.Reward  = mean(Z, 2);
        perf.RR      = mean(p.task.meanZ) / p.task.tNull;
        perf.Correct = ones(p.sim.nTrial, 1) / p.task.N;
        perf.CR      = 1 / p.task.N / p.task.tNull;

    otherwise
        error('Unknown model %s', model);
end
% add task/simulation information
perf.task = p.task;
perf.sim = p.sim;
if ~onlySummary
    perf.Z = Z;
end


function perf = performance(p, Z, c, dt, onlySummary)
nTrial = size(Z, 1);
[~, iZmax] = max(Z, [], 2);
Reward = Z((1:nTrial)+(c-1).*nTrial)';
Correct = (iZmax == c');
RT = dt';
cost = p.task.cpers * RT;
perf.mRT = mean(RT);
perf.mReward = mean(Reward);
perf.mCorrect = mean(Correct);
perf.mcost = mean(cost);
perf.RR = (perf.mReward - perf.mcost) / (perf.mRT + p.task.tNull);
perf.CR = (perf.mCorrect - perf.mcost) / (perf.mRT + p.task.tNull);
if ~onlySummary
    perf.Correct = Correct;
    perf.Reward = Reward;
    perf.RT = RT;
    perf.cost = cost;
    perf.Choice = c;
end
