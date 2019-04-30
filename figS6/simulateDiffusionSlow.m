function perf = simulateDiffusionSlow(p, model, onlySummary, Z)
% simulates diffusion model and returns performance summary
%
% p.task  - task parameters
% p.sim   - simulation parameters
% p.model - model parameters
% r       - stochastic evidence
% model   - model string
%
% See simulateDiffusion(.) for supported models and parameters.
%
% If onlySummary (optional) is given and true, only summary statistics are
% returned, and per-trial statistics are omitted. (this is currently not
% implemented).
%
% If Z (optimal) is given, then it is expected to be an trials x N matrix
% that contains the latent state values for each trial. If it isn't given,
% it is drawn from a multivariate Gaussian per trial with means
% p.task.meanZ, and covariance p.task.covZ.

if nargin < 3, onlySummary = false; end
if nargin >= 4
    r = commonDynamics(p, Z);
else
    r = commonDynamics(p);
end

% integration noise with variance intNoise^2 * intFano * x
intNoise = @(X) p.model.intNoise * ...
    sqrt(p.sim.dt * p.model.intFano * abs(X)) .* randn(size(X));

switch upper(model)
    case 'RM'      % standard race model
        if p.model.intNoise > 0
            X = NaN(size(r.dX));
            X(:,:,1) = r.dX(:,:,1) + p.model.u0;
            for iT = 2:length(p.sim.t)
                X(:,:,iT) = X(:,:,iT-1) + r.dX(:,:,iT) + intNoise(X(:,:,iT-1));
            end
        else
            X = cumsum(r.dX,3) + p.model.u0;
        end
        perf = performance(p, r, X);
        
    case 'URM'     % race model + urgency
        if p.model.intNoise > 0
            X = NaN(size(r.dX));
            X(:,:,1) = r.dX(:,:,1) + p.model.u0;
            db = p.model.b * p.sim.dt;
            for iT = 2:length(p.sim.t)
                X(:,:,iT) = X(:,:,iT-1) + db + r.dX(:,:,iT) + intNoise(X(:,:,iT-1));
            end
        else
            u = p.model.u0 + p.model.b * p.sim.t;
            X = cumsum(r.dX,3) + ...
                shiftdim(repmat(u', [1 p.sim.nTrial p.task.N]), 1);
        end
        perf = performance(p, r, X);
        
    case 'NRM'     % normalized race model
        % normalization ensures that mean(x) = u0, always
        if p.model.intNoise > 0
            X = NaN(size(r.dX));
            X(:,:,1) = r.dX(:,:,1) + p.model.u0;
            for iT = 2:length(p.sim.t)
                X(:,:,iT) = X(:,:,iT-1) + r.dX(:,:,iT) + intNoise(X(:,:,iT-1));
                X(:,:,iT) = X(:,:,iT) - ...
                    repmat(mean(X(:,:,iT),2), [1 p.task.N]) + p.model.u0;
            end
        else
            X = cumsum(r.dX,3);
            X = X - repmat(mean(X, 2), [1 p.task.N 1]) + p.model.u0;
        end
        perf = performance(p, r, X);

    case 'UNRM'     % normalized race model & urgency
        u = p.model.u0 + p.model.b * p.sim.t;
        if p.model.intNoise > 0
            X = NaN(size(r.dX));
            X(:,:,1) = r.dX(:,:,1) + p.model.u0;
            for iT = 2:length(p.sim.t)
                X(:,:,iT) = X(:,:,iT-1) + r.dX(:,:,iT) + intNoise(X(:,:,iT-1));
                X(:,:,iT) = X(:,:,iT) - ...
                    repmat(mean(X(:,:,iT),2), [1 p.task.N]) + u(iT);
            end
        else
            X = cumsum(r.dX,3);
            X = bsxfun(@plus, bsxfun(@minus, X, mean(X, 2)), ...
                              reshape(u, [1 1 length(u)]));
        end
        perf = performance(p, r, X);
        
    case 'CRM'     % constrained race model
        pm = p.model;
        f        = @(X) max(0, X) .^ pm.a;
        % can't use intNoise, as here scaled by nIterConstraint
        addNoise = @(X) X + pm.intNoise * ...
                        sqrt(p.sim.dt / pm.nIterConstraint * abs(X) .* pm.intFano) .* randn(size(X));
        % apply constraint to first step
        X = NaN(size(r.dX));
        X(:,:,1) = r.dX(:,:,1) + pm.u0;
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
                % add integration noise if needed
                if p.model.intNoise > 0
                    X(:,:,iT) = addNoise(X(:,:,iT));
                end
            end
        end
        perf = performance(p, r, X);

        
    case 'CRM2'     % constrained race model, nonlinear accum.
        pm = p.model;
        f        = @(X) max(0, X) .^ pm.a;
        % can't use intNoise, as here scaled by nIterConstraint
        addNoise = @(X) X + pm.intNoise * ...
                        sqrt(p.sim.dt / pm.nIterConstraint * abs(X) .* pm.intFano) .* randn(size(X));
        % apply constraint to first step
        X = NaN(size(r.dX));
        X(:,:,1) = f(r.dX(:,:,1) + pm.u0);
        for iIt = 1:pm.nIterConstraint
            err_ = pm.u0 - repmat(mean(X(:,:,1),2), [1 p.task.N]);
            X(:,:,1) = f(X(:,:,1) + pm.iterAlpha * err_);
        end
        % continue integrating after that
        for iT = 2:length(p.sim.t)
            X(:,:,iT) = f(X(:,:,iT-1) + r.dX(:,:,iT));
            % apply constraint
            for iIt = 1:pm.nIterConstraint
                err_ = pm.u0 - repmat(mean(X(:,:,iT),2), [1 p.task.N]);
                X(:,:,iT) = f(X(:,:,iT) + pm.iterAlpha * err_);
                % add integration noise if needed
                if p.model.intNoise > 0
                    X(:,:,iT) = addNoise(X(:,:,iT));
                end
            end
        end
        perf = performance(p, r, X);


    case 'UCRM'    % constrained race model & urgency
        pm = p.model;
        f        = @(X) max(0, X) .^ pm.a;
        % can't use intNoise, as here scaled by nIterConstraint
        addNoise = @(X) X + pm.intNoise * ...
                        sqrt(p.sim.dt / pm.nIterConstraint * abs(X) .* pm.intFano) .* randn(size(X));
        u = pm.u0 + pm.b * p.sim.t;
        % apply constraint to first step
        X = NaN(size(r.dX));
        X(:,:,1) = r.dX(:,:,1) + u(1);
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
                % add integration noise if needed
                if p.model.intNoise > 0
                    X(:,:,iT) = addNoise(X(:,:,iT));
                end
            end
        end
        perf = performance(p, r, X);

        
    case 'UCRM2'    % constrained race model & urgency, nonlinear accum.
        pm = p.model;
        f        = @(X) max(0, X) .^ pm.a;
        % can't use intNoise, as here scaled by nIterConstraint
        addNoise = @(X) X + pm.intNoise * ...
                        sqrt(p.sim.dt / pm.nIterConstraint * abs(X) .* pm.intFano) .* randn(size(X));
        u = pm.u0 + pm.b * p.sim.t;
        % apply constraint to first step
        X = NaN(size(r.dX));
        X(:,:,1) = f(r.dX(:,:,1) + u(1));
        for iIt = 1:pm.nIterConstraint
            err_ = u(1) - repmat(mean(X(:,:,1),2), [1 p.task.N]);
            X(:,:,1) = f(X(:,:,1) + pm.iterAlpha * err_);
        end
        % continue integrating after that
        for iT = 2:length(p.sim.t)
            X(:,:,iT) = f(X(:,:,iT-1) + r.dX(:,:,iT));
            % apply constraint
            for iIt = 1:pm.nIterConstraint
                err_ = u(iT) - repmat(mean(X(:,:,iT),2), [1 p.task.N]);
                X(:,:,iT) = f(X(:,:,iT) + pm.iterAlpha * err_);
                % add integration noise if needed
                if p.model.intNoise > 0
                    X(:,:,iT) = addNoise(X(:,:,iT));
                end
            end
        end
        perf = performance(p, r, X);

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
