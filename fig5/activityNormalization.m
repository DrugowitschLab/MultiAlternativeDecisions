function activityNormalization(noise)
%% Change of firing rate with increasing total value in UCRM (full model): 
% UCRM: Urgency + Constraint (projection) added to the Race Model
%
% Glimcher et. al. (2013) showed that the firing rate of neurons in the
% receptive field goes down as a function of the total value outside the
% receptive field. We replicate this effect here with our full model.
% 
%% Default inputs
switch nargin
    case 0
        noise           = 1;
        scale_meanZ     = 1;
    case 1
        scale_meanZ     = 1;
end


%% Setting paths and common parameters
fprintf('Setting path variables and common parameters...\n')
addpath('../shared/');
p = baseParameters;
p.sim.nTrial = 1e5;   
p.model.u0   = 0.7771;    % src: (UCRM_RR_N3noise1_fitSigH.mat>opttheta)
p.model.b    = 0.0013;    % src: (UCRM_RR_N3noise1_fitSigH.mat>opttheta)
p.sim.maxt   = 0.2;       % 200 ms
scale_dt     = 5;
scale_covX   = 5;
p.sim.dt     = p.sim.dt / scale_dt;
p.sim.t      = 0:p.sim.dt:p.sim.maxt;
p.task.covX  = p.task.covX / (scale_covX);
p.model.a    = 1.5;


%% Rewards and common dynamics
fprintf('Drawing rewards and generating reward-dependent dynamics... \n')
[Z, valOutside, nBlocks] = drawZ(p.sim.nTrial);
r = commonDynamics(p, Z);


%% Model
% Note that the code below does not follow Equation 40 of the Supplementary
% information of Tajima, S. et al. (2019) Nature Neuroscience. This is the
% efficient, vectorised version of the implementation. Note that the order
% of accumulation --> projection --> divisive normalization is maintained.
% Using Equation 40 does not alter the results; they are mathematically 
% equivalent. For an example of the other implementation, check out
% iiaViolation_inefficient.m

fprintf('Simulating the model for %i trials...\n', p.sim.nTrial)
pm = p.model;
f  = @(X) max(0, X) .^ pm.a;
u  = pm.u0 + pm.b * p.sim.t;

% Evidence accumulation and projection for t==1
X = NaN(size(r.dX));
X(:,:,1) = r.dX(:,:,1) + u(1);
for iIt = 1:pm.nIterConstraint          % constraint/projection @ t=1
    err_ = u(1) - repmat(mean(f(X(:,:,1)),2), [1 p.task.N]);
    X(:,:,1) = X(:,:,1) + pm.iterAlpha * err_;
end

% Evidence accumulation and projection for t>1
for iT = 2:length(p.sim.t)
    X(:,:,iT) = X(:,:,iT-1) + r.dX(:,:,iT);
    for iIt = 1:pm.nIterConstraint      % constraint/projection @ t>1
        err_ = u(iT) - repmat(mean(f(X(:,:,iT)),2), [1 p.task.N]);
        X(:,:,iT) = X(:,:,iT) + pm.iterAlpha * err_;
    end
end

% Divisive normalization
K    = mean(r.Z(:)); 
sigH = -min(sum(X(:,:,end),2));   % proxy for optimal sigH: avoids div by 0
s    = sum(X,2) + sigH;

% steady state implementation
Y = K*(X)./s;           % divisive normalization
Y = mvnrnd(Y(:,:,end), noise *scale_covX * scale_meanZ^2 * p.task.covX);


%% Mapping integrated values to firing rates via sigmoid
firingRate = @(X) 1./(1 + exp(-0.1*(X-mean(X,2))));
targetFR = firingRate(Y);
targetFR = targetFR(:,1);

% Averaging for each block
targetFR   = reshape(targetFR, p.sim.nTrial / nBlocks, nBlocks);
muTargetFR = nanmean(targetFR, 1);
sdTargetFR = nanstd(targetFR, 1);
seTargetFR = nanstd(targetFR, 1) / sqrt(p.sim.nTrial / nBlocks);

%% Plotting
%
fprintf('Plotting firing rate for target in RF...\n')

figure();
    scatter(valOutside, muTargetFR, 2000, 'filled'); hold on
    errorbar(valOutside, muTargetFR, sdTargetFR, 'LineStyle','none');
    xlim([0,70]); ylim([0.4,0.8]);
    xlabel('Total value of targets outside RF (a.u.)')
    ylabel('Unit activity (a.u.)')
    title('Model')
    set(gca,'FontSize',20)
    
fprintf('Done\n\n')



%% Helper functions
%
function [Z, blockSum, nBlocks] = drawZ(nTrial)
    % Draws Z in blocks with z_2 + z_3 = constant in each block
    z1       = 30;
    nBlocks  = 5;
    blockSum = linspace(10, 60, nBlocks);
    Z        = [];
    
    for ii = 1:nBlocks
        blockMu = NaN(nTrial/nBlocks, 3);   % pre-alloc
        for jj = 1:(nTrial/nBlocks)
            blockMu(jj,:) = drawConstSum(z1, blockSum(ii));
        end
        Z = vertcat(Z, blockMu);
    end
    
    function Z_ = drawConstSum(z1,sum_)
        % Draws Zs in a block with z2 + z3 = const
        z2 = sum_ * rand();
        z3 = sum_ - z2;
        Z_ = [z1 z2 z3];    