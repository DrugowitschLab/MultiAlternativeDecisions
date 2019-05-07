function perf = simulateDiffusion(p, model, noise, Z)
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
% UCRM    - race model, urgency & constraint accumulation
%           p.model.u0: initial offset
%           p.model.b : urgency slope
%           p.model.a : power of the non-linearity
%           p.model.nIterConstraint: max. number of constraint iterations
%           p.model.iterAlpha: learning rate of constraint interations
% RANDOM  - random, immediate choices
%
% All models (except RANDOM) support integrator noise, with parameters
% p.model.intNoise: noise magnitude
% p.model.infFano: noise Fano factor
%
% If Z (optimal) is given, then it is expected to be an trials x N matrix
% that contains the latent state values for each trial. If it isn't given,
% it is drawn from a multivariate Gaussian per trial with means
% p.task.meanZ, and covariance p.task.covZ.

%% Common dynamics and noise setup
n_ = noise;

if nargin >= 5,  r = commonDynamics(p, Z);
else,            r = commonDynamics(p);       end

%% Models
switch upper(model)
    case 'RM'      % standard race model
        X    = cumsum(r.dX,3) + p.model.u0;
        [Y,p]= runDynamics(X, r, p, n_);                 % Also adds noise
        perf = performance(p, r, Y);
                
    case 'URM'     % race model + urgency
        u    = p.model.u0 + p.model.b * p.sim.t;
        X    = cumsum(r.dX,3) + shiftdim(repmat(u',[1 p.sim.nTrial p.task.N]), 1);
        [Y,p]= runDynamics(X, r, p, n_);                 % Also adds noise
        perf = performance(p, r, Y);
        
    case 'NRM'     % normalized race model
        % normalization ensures that mean(x) = u0, always
        X    = cumsum(r.dX,3);
        X    = X - mean(X, 2) + p.model.u0;
        [Y,p]= runDynamics(X, r, p, n_);                 % Also adds noise
        perf = performance(p, r, Y);

    case 'UNRM'     % normalized race model & urgency
        u    = p.model.u0 + p.model.b * p.sim.t;
        X    = cumsum(r.dX,3);
        X    = X - mean(X, 2) + reshape(u, [1 1 length(u)]);
        [Y,p]= runDynamics(X, r, p, n_);                 % Also adds noise
        perf = performance(p, r, Y);
        
    case 'UCRM'    % constrained race model & urgency
        pm = p.model;
        f  = @(X) max(0, X) .^ pm.a;
        u  = pm.u0 + pm.b * p.sim.t;
        % apply constraint to first step
        X = NaN(size(r.dX));
        X(:,:,1) = r.dX(:,:,1);
        for iIt = 1:pm.nIterConstraint
            err_ = u(1) - repmat(mean(f(X(:,:,1)),2), [1 p.task.N]);
            X(:,:,1) = X(:,:,1) + pm.iterAlpha * err_;
        end
        % continue integrating after that
        for iT = 2:length(p.sim.t)
            X(:,:,iT) = X(:,:,iT-1) + r.dX(:,:,iT);
            % apply constraint
            for iIt = 1:pm.nIterConstraint
                err_ = u(iT) - repmat(mean(f(X(:,:,iT)),2), [1 p.task.N]);
                X(:,:,iT) = X(:,:,iT) + pm.iterAlpha * err_;
            end
        end
        [Y,p]= runDynamics(X, r, p, n_);                 % Also adds noise
        perf = performance(p, r, Y);

    case 'CRM'    % constrained race model & urgency
        pm = p.model;
        f  = @(X) max(0, X) .^ pm.a;
        % apply constraint to first step
        X = NaN(size(r.dX));
        X(:,:,1) = r.dX(:,:,1);
        for iIt = 1:pm.nIterConstraint
            err_ = pm.u0 - repmat(mean(f(X(:,:,1)),2), [1 p.task.N]);
            X(:,:,1) = X(:,:,1) + pm.iterAlpha * err_;
        end
        % continue integrating after that
        for iT = 2:length(p.sim.t)
            X(:,:,iT) = X(:,:,iT-1) + r.dX(:,:,iT);
            % apply constraint
            for iIt = 1:pm.nIterConstraint
                err_ = pm.u0 - repmat(mean(f(X(:,:,iT)),2), [1 p.task.N]);
                X(:,:,iT) = X(:,:,iT) + pm.iterAlpha * err_;
            end
        end
        [Y,p]= runDynamics(X, r, p, n_);                 % Also adds noise
        perf = performance(p, r, Y);
        
    case 'RANDOM'  % random, immediate choices
        perf.RT      = zeros(p.sim.nTrial, 1);
        perf.Reward  = mean(r.Z, 2);
        perf.RR      = mean(p.task.meanZ) / p.task.tNull;
        perf.Correct = ones(p.sim.nTrial, 1) / p.task.N;
        perf.CR      = 1 / p.task.N / p.task.tNull;

    otherwise
        error('Unknown model %s', model);
end
% add task/simulation information
perf.task = p.task;
perf.sim = p.sim;
perf.Z = r.Z;


%% Helper functions
function [Y,p] = runDynamics(X,r,p,n_)
    % Factors for divisive normalization
    K    = mean(r.Z(:));
    sigH = p.model.sigH;                   % to be optimized 
    % Div norm on Y
    Y    = K*X./(sigH + sum(X,2));         % Assuming steady-state for Y
    % Scaling threhold (equivalent to scaling u, keeping fixed threshold)
    p.task.threshold = K*p.task.threshold./(sigH + sum(X,2));
    p.task.threshold = permute(p.task.threshold, [1 3 2]);
    % Noise addition to Y
    for iT = 1:length(p.sim.t)
        Y(:,:,iT) = mvnrnd(Y(:,:,iT), n_ * p.task.covX * p.sim.dt * iT);
    end

function perf = performance(p, r, X)
% fixed duration statistics
[Xmax_, perf.FDD] = max(permute(X, [1 3 2]),[],3);   % trial x t
perf.FDReward = varSelected(r.Z, perf.FDD);
perf.FDCorrect = (perf.FDD == repmat(r.iZmax, [1 length(p.sim.t)]));
perf.FDRR = mean(perf.FDReward - repmat(p.task.accumc, [p.sim.nTrial 1])) ./ ...
            mean(repmat(p.sim.t, [p.sim.nTrial 1]) + p.task.tNull);
perf.FDCR = mean(perf.FDCorrect - repmat(p.task.accumc, [p.sim.nTrial 1])) ./ ...
            mean(repmat(p.sim.t, [p.sim.nTrial 1]) + p.task.tNull);
% figure out which trials reached the threshold and when
aboveThresh = (Xmax_ >= p.task.threshold);  % above threshold, trial x t
aboveThresh(:,end) = 1;                     % guaranteed in the end
yetToEnd = cumsum(aboveThresh,2) == 0;      % not yet reached, trial x t
flagEnd = diff([ones(p.sim.nTrial,1) yetToEnd], 1, 2)<0; % 1 at threshold, trial x t
% reaction time statistics
perf.iTEnd = max(1, sum(yetToEnd, 2));      % index of threshold crossing
perf.RT = p.sim.t(perf.iTEnd+1)';
perf.Choice = sum(perf.FDD .* flagEnd, 2);
perf.cost = p.task.accumc(perf.iTEnd)';
perf.Reward = varSelected(r.Z, perf.Choice);
perf.Correct = (perf.Choice == r.iZmax);
perf.RR = mean(perf.Reward - perf.cost) ./ mean(perf.RT + p.task.tNull);
perf.CR = mean(perf.Correct - perf.cost) ./ mean(perf.RT + p.task.tNull);

function y = varSelected(Z, D)
nT = size(D,2);
y = (D==1) .* repmat(squeeze(Z(:,1,:)),[1 nT]);     % y(iTrial, iTime)
for iD = 2:size(Z, 2)
    y = y + (D==iD) .* repmat(squeeze(Z(:,iD,:)),[1 nT]);
end