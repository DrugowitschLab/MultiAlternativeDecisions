function reactionTime()
%% Hick's law in choice reaction times
% Hick’s law is one of the most robust properties of choice RTs in 
% perceptual decision tasks. In its classic form, it suggests the linear 
% relationship between mean RT and the logarithm of the number of options.
% Our model replicates this near-logarithmic relationship.


%% Dependencies
% ../shared/baseParameters.m
% ./optimParams_percep/
% ./simulateDiffusion_percep.m


%% Fetching the reaction times
addpath('../shared/');
N = [2 3 4 6 8];

rTime_percep = computeRT(N,'CR');
rTime_value  = computeRT(N,'RR');

plotHick(N,rTime_percep); title('Perceptual Task')
plotHick(N,rTime_value); title('Value-based task')


%% Helper functions
function rTime = computeRT(N,obj)
    rTime = nan(1,length(N));
    
    for iN = 1:length(N)
        % Setting parameters
        p                   = baseParameters();
        p.sim.maxt          = 0.5;
        p.task.N            = N(iN);
        p.model.a           = 1.5;
        p.sim.nTrial        = 1e4;
        p.task.meanZ        = ones(1,p.task.N);
        p.task.covX         = eye(p.task.N);
        p.task.covZ         = eye(p.task.N);
        p.task.icovZ        = eye(p.task.N);
        p.task.icovX        = eye(p.task.N);
        p                   = baseParameters(p);
        p.task.covX         = eye(p.task.N);
        
        % Fetching optimal parameters
        fname           = ['UCRM_' obj '_N' num2str(N(iN)) '_fitSigH.mat'];
        load(['../shared/optimParams_percep/' fname],'opttheta')
        p.model.u0      = opttheta(1);
        p.model.b       = opttheta(2);
        
        % Running model
        perf  = simulateDiffusion_percep(p,'UCRM',0);
        rTime(iN) = mean(perf.RT);
    end

function plotHick(N,rTime)
    n=log(N+1);
    mdl = fitlm(n,rTime);
    r_sq = mdl.Rsquared.Ordinary;

    figure()
    semilogx(N+1,rTime,'o-','MarkerSize',10,'Linewidth',2)
    xticklabels({'2','3','4','','6','','8'})
    xlabel('Number of options, scaled with log(N+1)')
    ylabel('Reaction time (a.u.)')
    ylim([0 0.3])
    text(7, 0.29, ['R^2 = ' num2str(r_sq)],'FontSize',20)
    set(gca,'FontSize',24)
    box off