function similarityEffectwTime(sim_coeff)
%% Similarity effect in UCRM (our full neural circuit model): 
% UCRM: Urgency + Constraint (projection) added to the Race Model
% 
% As has been observed experimentally (Trueblood et al), we found that the 
% similarity effect grows over time during the course of a single trial.


%% Dependencies:
% ../shared/baseParameters.m
% ../shared/commonDynamics.m


%% Default inputs
if nargin < 1,  sim_coeff = 0.5; end
noise = 1; 


%% Setting paths and common parameters
fprintf('Setting path variables and common parameters...\n')
addpath('../shared/');
p = baseParameters;
p.sim.nTrial = 1e5;   
p.model.u0   = 0.7771;      % src: (UCRM_RR_N3noise1_fitSigH.mat>opttheta)
p.model.b    = 0.0013;      % src: (UCRM_RR_N3noise1_fitSigH.mat>opttheta)
p.sim.maxt   = 0.2;         % 200 ms
scale_dt     = 5;
scale_covX   = 5;
p.sim.dt     = p.sim.dt / scale_dt;
p.sim.t      = 0:p.sim.dt:p.sim.maxt;
p.task.covX  = [1  0  0;   0  1  sim_coeff;   0  sim_coeff  1];
p.task.covX  = p.task.covX / (scale_covX);
p.model.a    = 1.5;


%% Rewards and common dynamics
fprintf('Drawing rewards and generating reward-dependent dynamics... \n')
Z = drawZ(p.sim.nTrial);
r = commonDynamics(p, Z);


%% Model (UCRM)
% Note that the code below does not follow Equation 40 of the Supplementary
% information of Tajima, S. et al. (2019) Nature Neuroscience. This is the
% efficient, vectorised version of the implementation. Note that the order
% of accumulation --> projection --> divisive normalization is maintained.
% Using Equation 40 does not alter the results; they are mathematically 
% equivalent.

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
Y = K*(X)./s;                     % divisive normalization
for iT = 1:length(p.sim.t)
    Y(:,:,iT) = mvnrnd(Y(:,:,iT), noise *scale_covX * p.task.covX);
end


%% Computing choice probability at each time-step

fprintf('Computing choice probability...\n')
listChoiceT = randi(3, p.sim.nTrial, 1);      % random @ t = 0
for iT = 20:20:length(p.sim.t)                % spaced out for cleaner plot
    [~,choice]   = max(Y(:,:,iT),[],2);
    listChoiceT  = horzcat(listChoiceT, choice);            %#ok<AGROW>
end
psimt = p.sim.maxt*(0:20:length(p.sim.t))/length(p.sim.t);  % time
pChoiceT = histc(listChoiceT,[1 2 3])/length(listChoiceT);  % choice prob


%% Plotting
fprintf('Plotting strength of similarity effect with time...\n')

figure();
    plot(psimt,pChoiceT,'LineWidth',2)
    set(gca,'FontSize',20)
    xlabel('Time (s)')
    ylabel('P(choice)')
    title('Similarity effect with time')
    legend({'Option 1','Option 2','Option 3'});

fprintf('Done\n\n')



%% Local helper functions
function Z = drawZ(nTrial)
    %% Draws Z from a normal
    % All options are competitive, but option 1 is slightly more valueable 
    % on average. Absolute rewards for options 2 & 3 are uncorrelated, but
    % their evidence will be correlated due to common underlying features.
    %
    Z = mvnrnd(repmat([30 29.5 29], nTrial, 1), 1*eye(3));   