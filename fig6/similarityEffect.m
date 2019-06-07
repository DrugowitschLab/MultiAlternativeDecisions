function similarityEffect(sim_coeff)
%% Similarity effect in UCRM (our full neural circuit model): 
% UCRM: Urgency + Constraint (projection) added to the Race Model
% 
% The similarity effect refers to the fact that when subjects are given a 
% third option which is similar to, say, option 1, the probability of
% choosing option 1 decreases. 
%
% To model this effect, we postulated that each object is defined by a set 
% of features and that its overall value is a linear combination of the 
% values of its features. We also assume that the values of the features 
% are not known exactly. Instead, the brain generates noisy samples of 
% these values over time. In this scenario, the similarity between two 
% objects is proportional to the overlap between their features. This 
% overlap implies that the stream of value samples for the two similar 
% options are correlated (with sim_coeff), while being independent for the 
% third, dissimilar option.


%% Dependencies:
% ../shared/baseParameters.m
% ../shared/commonDynamics.m


%% Default inputs
if nargin < 1,  sim_coeff = 0.1; end
scale_meanZ = 1;   % Rescaling factor
noise = 1;


%% Setting paths and common parameters
fprintf('Setting path variables and common parameters...\n')
addpath('../shared/')
p = baseParameters;
p.sim.nTrial = 1e6;   
p.model.u0   = 0.7771;      % src: (UCRM_RR_N3noise1_fitSigH.mat>opttheta)
p.model.b    = 0.0013;      % src: (UCRM_RR_N3noise1_fitSigH.mat>opttheta)
p.sim.maxt   = 0.2;         % 200 ms
scale_dt     = 5;
scale_covX   = 5;
p.sim.dt     = p.sim.dt / scale_dt;
p.sim.t      = 0:p.sim.dt:p.sim.maxt;
p.task.covX  = [ 1  0  sim_coeff;   0  1  0;   sim_coeff  0  1];
p.task.covX  = p.task.covX / (scale_covX * scale_meanZ^2);
p.model.a    = 1.5;


%% Rewards and common dynamics
fprintf('Drawing rewards and generating reward-dependent dynamics... \n')
[Z, z1BinBounds, z2, z3BinBounds] = drawZ_scale(scale_meanZ,p.sim.nTrial);
r = commonDynamics(p, Z);


%% Model

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

% Choose the maximum valued option at the end of the race
[~,choice] = max(Y,[],2);



%% Plotting
fprintf('Plotting similarity effect...\n')
figure();
pChoice = binPchoice(Z, choice, z1BinBounds, z3BinBounds);
plotPChoice1(pChoice, z2, z1BinBounds, z3BinBounds);


fprintf('Done\n\n')



%% Local helper functions
function [Z, z1BinBounds, z2, z3BinBounds] = drawZ_scale(sc_factor, nTrial)
    %% draws Z uniformly from given ranges
    z1Bins = 10;
    z3Bins = 5;
    z1Rng  = [25 35];
    z2     = 30;
    z3Rng  = [25 30];
    z1BinBounds = linspace(min(z1Rng), max(z1Rng), z1Bins+1) / sc_factor;
    z3BinBounds = linspace(min(z3Rng), max(z3Rng), z3Bins+1) / sc_factor;
    Z  = bsxfun(@plus, ...
         bsxfun(@times, rand(nTrial, 3), [diff(z1Rng) 0 diff(z3Rng)]), ...
         [z1Rng(1) z2 z3Rng(1)]);
    Z  = Z / sc_factor;
    z2 = z2 / sc_factor;

function pChoice = binPchoice(Z, Choice, z1BinBounds, z3BinBounds)
    %% returns binned choice probabilities
    %
    % Z is trials x z-values matrix. Choice is row vector of choices (1, 2,
    % 3). zjBinBounds is vector of bin boundaries for zj.
    %
    % The function returns a z1Bins x z3Bins x 3 matrix of choice 
    % distributions for each bin across all 3 possible choices.
    % 
    % Adapted from Jan Drugowitsch's scripts

    z1Bins  = length(z1BinBounds) - 1;
    z3Bins  = length(z3BinBounds) - 1;
    N       = 3; % size(Choice, 2) = 1;
    pChoice = NaN(z1Bins, z3Bins, N);
    for iZ1 = 1:z1Bins
        for iZ3 = 1:z3Bins
            binTrials = Z(:,1) >= z1BinBounds(iZ1) & Z(:,1) < z1BinBounds(iZ1+1) & ...
                Z(:,3) >= z3BinBounds(iZ3) & Z(:,3) < z3BinBounds(iZ3+1);
            nTrial = sum(binTrials);
            for iD = 1:3
                pChoice(iZ1, iZ3, iD) = sum(Choice(binTrials) == iD) / nTrial;
            end
        end
    end
    
function plotPChoice1(pChoice, z2, z1BinBounds, z3BinBounds)
    % plots psychometric curves for different z3 bins

    % determine bin centers & counts
    z1BinCenters = 0.5*(z1BinBounds(1:end-1) + z1BinBounds(2:end));
    z3BinCenters = 0.5*(z3BinBounds(1:end-1) + z3BinBounds(2:end));
    z3Bins = length(z3BinCenters);

    % plot psychometric curves
    z3Col = gradation({[0 0 1],[0 1 0]}, z3Bins);
    for iZ3 = 1:z3Bins
        plot(z2 - z1BinCenters, pChoice(:, iZ3, 2),'-', ...
            'Color', z3Col(iZ3, :), 'Linewidth', 2);  hold on;
    end
    
    % formatting
    xlabel('z_1 - z_2');
    ylabel('P(choose 1)');
    legend(arrayfun(@(z3) sprintf('%4.1f', z3), z3BinCenters, ...
                            'UniformOutput', false));
    lgd = legend({'25','','','','30'},'Location','northwest');
    title(lgd,'Value of 3^{rd} option')
    set(gca,'FontSize',20)