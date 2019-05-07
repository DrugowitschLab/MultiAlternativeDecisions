%% Get mean firing rates for 2, 3 ,4 choices
N = [2 3 4];
for iN = N
    meanFR{iN} = urgency_buildup_Nchoices(iN);
end

%% Plots
% Plotting mean firing rate for 4 choice options
fprintf('Plotting mean firing rate...\n')

figure();
    plot(0.001*(1:length(meanFR{4})), meanFR{4},'Linewidth',2);
    xlabel('Time (a.u.)'); xlim([-2 10]); ylim([-7 4])
    ylabel('Unit Activity (a.u.)');
    xticklabels({''}); yticklabels({''});
    box off
    set(gca,'FontSize',24);
 

% Plotting offset
fprintf('Plotting firing rate offset...\n')
figure();
hold on
for iN = N
    meanFR_offset(iN-1) = meanFR{iN}(1); %#ok<SAGROW>
end
plot(N, meanFR_offset,'-o','Linewidth',2,'MarkerSize',16)
xlabel('Number of options'); xlim([1.5 4.5]); %ylim([2.5 13.5])
ylabel('Offset (a.u.)')
xticklabels({'','2','','3','','4'}); yticklabels({''})
box off
set(gca,'FontSize',24)
    
    
fprintf('Done\n\n')   



%% Urgency Buildup

function meanFR = urgency_buildup_Nchoices(N)
%% Buildup of urgency signal in our neural model: 
%
% Plot average firing rate with time, where the average is over options and
% over trials; options have similar values.
%
%% Default inputs
if nargin < 1,  N = 4; end

%% Setting paths and common parameters
fprintf('Setting path variables and common parameters...\n')
addpath('../shared/');
p = baseParameters;
p.sim.nTrial = 1e3;   
p.model.u0   = 0.7771;    % src: (UCRM_RR_N3noise1_fitSigH.mat>opttheta)
p.model.b    = 0.0013;    % src: (UCRM_RR_N3noise1_fitSigH.mat>opttheta)
p.sim.maxt   = 10;       % 200 ms
scale_dt     = 5;
scale_covX   = 1;
p.task.N     = N;
p.task.meanZ = ones(1,p.task.N);
p.task.covX  = eye(p.task.N);
p.task.covZ  = eye(p.task.N);
p.task.icovZ = eye(p.task.N);
p.task.icovX = eye(p.task.N);
p.sim.dt     = p.sim.dt / scale_dt;
p.sim.t      = 0:p.sim.dt:p.sim.maxt;
p.task.covX  = p.task.covX / (scale_covX);
p.model.a    = 1.5;


%% Rewards and common dynamics
fprintf('Drawing rewards and generating reward-dependent dynamics... \n')
Z = repmat(30*ones(1,p.task.N),p.sim.nTrial,1);
r = commonDynamics(p, Z);


%% Model
fprintf('Simulating the model for %i trials...\n', p.sim.nTrial)
pm = p.model;
f  = @(X) max(0, X) .^ pm.a;
u  = pm.u0 + pm.b * p.sim.t;

X = NaN(size(r.dX));
X(:,:,1) = r.dX(:,:,1) + u(1);
for iIt = 1:pm.nIterConstraint          % constraint/projection @ t=1
    err_ = u(1) - repmat(mean(f(X(:,:,1)),2), [1 p.task.N]);
    X(:,:,1) = X(:,:,1) + pm.iterAlpha * err_;
end

for iT = 2:length(p.sim.t)
    X(:,:,iT) = X(:,:,iT-1) + r.dX(:,:,iT);
    for iIt = 1:pm.nIterConstraint      % constraint/projection @ t>1
        err_ = u(iT) - repmat(mean(f(X(:,:,iT)),2), [1 p.task.N]);
        X(:,:,iT) = X(:,:,iT) + pm.iterAlpha * err_;
    end
end


% Divisive normalization
K    = mean(r.Z(:)); 
sigH = -8;              % proxy for optimal sigH: avoids div by 0
s    = sum(X,2) + sigH;

% steady state implementation
Y = K*(X)./s;           % divisive normalization
%Y = mvnrnd(Y, noise * scale_covX * p.task.covX); 


%% Mapping integrated values to firing rates 
% firingRate = @(X) 100./(1 + exp(-X)); 
% meanFR     = firingRate(Y);       % squashing via sigmoid
meanFR     = mean(nanmean(Y));      % average across options & trials
meanFR     = reshape(meanFR, length(meanFR), 1);


end