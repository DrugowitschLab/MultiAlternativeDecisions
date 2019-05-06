function violationOfRegularity_3to4()
%% Violation of the regularity principle for N=4 options in our UCRM model
% We show that the regularity principle is not only violated when adding a
% third option given two options in the choice set (violationOfRegularity),
% but also for adding a fourth option given three options in the choice
% set. 


%% Dependencies
% ../shared/baseParameters.m
% ../shared/commonDynamics.m


%% Default inputs
noise = 1;
scale_meanZ = 1;   % Rescaling factor

%% Setting paths and common parameters
fprintf('Setting path variables and common parameters...\n')
addpath('../shared/');
p = baseParameters;
p.sim.nTrial     = 1e6;  
p.model.u0       = 0.7771;  % src: (UCRM_RR_N3noise1_fitSigH.mat>opttheta)
p.model.b        = 0.0013;  % src: (UCRM_RR_N3noise1_fitSigH.mat>opttheta)
p.sim.maxt       = 0.2;     % 200ms
p.model.a        = 1.5;
scale_dt     = 5;
scale_covX   = 5;
p.sim.dt     = p.sim.dt / scale_dt;
p.sim.t      = 0:p.sim.dt:p.sim.maxt;
% setting up for 4 options
p.task.N     = 4;
p.task.meanZ = ones(1,p.task.N);
p.task.covX  = eye(p.task.N);
p.task.covZ  = eye(p.task.N);
p.task.icovZ = eye(p.task.N);
p.task.icovX = eye(p.task.N);
p.task.covX  = p.task.covX / (scale_covX * scale_meanZ^2);

%% Rewards and common dynamics
[Z, z1BinBounds, z2BinBounds, z3BinBounds, z4BinBounds] = ...
    drawZ4([25 35], [25 35], [0 30], [25 35], 10, 10, 5, 10, p.sim.nTrial);  % Varying z1/3, z2=c
r = commonDynamics(p, Z);


%% Model
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
Y = K*(X)./s;           % divisive normalization
Y = mvnrnd(Y(:,:,end), noise *scale_covX * scale_meanZ^2 * p.task.covX);

% Choose the maximum valued option at the end of the race
[~,choice] = max(Y,[],2);


%% Plotting
pChoose1 = binPchoice(Z, choice, z1BinBounds, z2BinBounds, z3BinBounds, z4BinBounds);


%% Local helper functions
function [Z, z1BinBounds, z2BinBounds, z3BinBounds, z4BinBounds] = ...
                drawZ4(z1Rng, z2Rng, z3Rng, z4Rng, z1Bins, z2Bins, z3Bins, z4Bins, nTrial)
%% draws Z uniformly from given ranges

z1BinBounds = linspace(min(z1Rng), max(z1Rng), z1Bins+1);
z2BinBounds = linspace(min(z2Rng), max(z2Rng), z2Bins+1);
z3BinBounds = linspace(min(z3Rng), max(z3Rng), z3Bins+1);
z4BinBounds = linspace(min(z4Rng), max(z4Rng), z4Bins+1);
Z = bsxfun(@plus, ...
    bsxfun(@times, rand(nTrial, 4), [diff(z1Rng) diff(z2Rng) diff(z3Rng) diff(z4Rng)]), ...
    [z1Rng(1) z2Rng(1) z3Rng(1) z4Rng(1)]);

function pChoose1 = binPchoice(Z, Choice, z1BinBounds, z2BinBounds, z3BinBounds, z4BinBounds)
%% returns binned choice probabilities and the plot
%
% Z is trials x z-values matrix. Choice is row vector of choices (1, 2 or
% 3). zjBinBounds is vector of bin boundaries for zj.
%
% The function returns a (z1-z2) x (z1-z3) x z4Bins matrix of choice 
% distributions for each bin for choosing option 1.

z1Bins  = length(z1BinBounds) - 1;
z2Bins  = length(z2BinBounds) - 1;
z3Bins  = length(z3BinBounds) - 1;
z4Bins  = length(z4BinBounds) - 1;

% Added for symmetric plot
x_ = min(z1BinBounds)-max(z2BinBounds):max(z1BinBounds)-min(z2BinBounds);
y_ = min(z1BinBounds)-max(z4BinBounds):max(z1BinBounds)-min(z4BinBounds);
pChoose1 = zeros(length(x_), length(y_), 5);
for ix = 1:length(x_)
for iy = 1:length(y_)
for iZ3 = 1:z3Bins
    binTrials = ...
        Z(:,1)-Z(:,2) >= x_(ix)-0.5 & Z(:,1)-Z(:,2) < x_(ix)+0.5 & ...
        Z(:,1)-Z(:,4) >= y_(iy)-0.5 & Z(:,1)-Z(:,4) < y_(iy)+0.5 & ...
        Z(:,3) >= z3BinBounds(iZ3) & Z(:,3) < z3BinBounds(iZ3+1);
    nTrial = sum(binTrials);
    pChoose1(ix, iy, iZ3) = sum(Choice(binTrials) == 1) / nTrial;
end
end
end

% Plotting the surfaces
figure();
op4abs = surf(x_,y_,pChoose1(:,:,1));
set(op4abs,'FaceColor','blue'); hold on;
op4prs = surf(x_,y_,pChoose1(:,:,5));
set(op4prs,'FaceColor','green');
xlim([-5,5]); xlabel('z_1-z_2');
ylim([-5,5]); ylabel('z_1-z_3');
zlabel('P(choose 1)');
legend({'Option 4 absent','Option 4 present'});
set(gca,'FontSize',30)