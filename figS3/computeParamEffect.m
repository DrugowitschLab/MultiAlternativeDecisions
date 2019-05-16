function computeParamEffect()
tic;

% load network parameters
load('paramOptimResult_N3-UCRM.mat');

% simulation settings
p.dt = 0.005;
p.t = 0 : p.dt : 1;
p.nTrial = 50000;
%p.nTrial = 500000;
modelId = {'uc' , 'ur' , 'lu'};

% adjust network parameters
p.c = 0 * p.t;
p.cTotal = p.c * triu(ones(length(p.t))) * p.dt;
p.noiseGain    = 0;
p.noiseGainDiv = 0;
p.uc.u0 = 0.7;
p.uc.nIterConstrain = 10;

% parameter ranges
offsetVary = 0.7:.005:0.85;
slopeVary = 1:1:30;
costVary = 0:1:50;
nlinVary = 1:0.02:2;
%offsetVary = 0.7:.05:0.85;
%slopeVary = 1:5:30;
%costVary = [0 10 50];
%nlinVary = [1 1.5 2];


%% 1) Urgency offset vs. accumulation cost:
param1Label = 'Offset';
param1  = offsetVary; 
param2Label = 'AccumulationCost';
param2  = costVary;

for iParam2 = length(param2):-1:1
    fprintf('- %s = %d\n', param2Label, param2(iParam2));
    %parfor iParam1 = 1:length(param1)
    for iParam1 = 1:length(param1)
        % adjust parameters and evaluate performance
        p_ = p;
        p_.c = param2(iParam2) * p.t;
        p_.cTotal = p_.c * triu(ones(length(p.t))) * p.dt;
        p_.ur.u0 = param1(iParam1);              % RM
        p_.uc.u0 = param1(iParam1);              % Full
        fprintf('\t- %s = %+2.2f\t', param1Label, param1(iParam1));
        r2(iParam2,iParam1) = modelPerformance(p_, commonDynamics(p_));
        % clean up to save memory, and take averages
        for iModel = 1:length(modelId)
            r2(iParam2,iParam1).(modelId{iModel}) = rmfield(r2(iParam2,iParam1).(modelId{iModel}),...
                                                           {'X','D','iTEnd','cost','DFixedTime','RewardFixedTime','RRFixedTime','correctFixedTime'});
            if isfield(r2(iParam2,iParam1).(modelId{iModel}),'err')
                r2(iParam2,iParam1).(modelId{iModel}) = rmfield(r2(iParam2,iParam1).(modelId{iModel}),'err');
            end
            r2(iParam2,iParam1).(modelId{iModel}).RT = finAvg(r2(iParam2,iParam1).(modelId{iModel}).RT);
            r2(iParam2,iParam1).(modelId{iModel}).Reward = finAvg(r2(iParam2,iParam1).(modelId{iModel}).Reward);
            r2(iParam2,iParam1).(modelId{iModel}).correct = finAvg(r2(iParam2,iParam1).(modelId{iModel}).correct);
        end
        r2(iParam2,iParam1).X0 = 0;  r2(iParam2,iParam1).Z = 0;  r2(iParam2,iParam1).Zmax = 0;
        r2(iParam2,iParam1).iZmax = 0;  r2(iParam2,iParam1).dX = 0;  r2(iParam2,iParam1).base = 0;
        r2(iParam2,iParam1).normalized = 0;
        r(iParam2,iParam1) = rmfield(r2(iParam2,iParam1), ...
                                      {'X0','Z','Zmax','iZmax','dX','base','normalized'});
    end
end
clear r2;
toc;
rfilename = [mfilename '_' param1Label '_vs_' param2Label];
save(rfilename, 'r', 'param1Label', 'param2Label', 'param1', 'param2');
clear r;
fprintf('Results written to %s\n', rfilename);
toc;

return


%% 2) Urgency slope vs. accumulation cost:
param1Label = 'Slope';
param1  = slopeVary;
param2Label = 'AccumulationCost';
param2  = costVary;

for iParam2 = length(param2):-1:1
    fprintf('- %s = %d\n', param2Label, param2(iParam2));    
        
    parfor iParam1 = 1:length(param1)
        % adjust parameters and evaluate performance
        p_ = p;
        p_.c = param2(iParam2) * p.t;
        p_.cTotal = p_.c * triu(ones(length(p.t))) * p.dt;
        p_.ur.b = param1(iParam1);              % RM
        p_.uc.b = param1(iParam1);              % Full
        fprintf('\t- %s = %+2.2f\t', param1Label, param1(iParam1));
        r2(iParam2,iParam1) = modelPerformance(p_, commonDynamics(p_));
        % clean up to save memory
        for iModel = 1:length(modelId)
            r2(iParam2,iParam1).(modelId{iModel}) = rmfield(r2(iParam2,iParam1).(modelId{iModel}),...
                                                           {'X','D','iTEnd','cost','DFixedTime','RewardFixedTime','RRFixedTime','correctFixedTime'});
            if isfield(r2(iParam2,iParam1).(modelId{iModel}),'err')
                r2(iParam2,iParam1).(modelId{iModel}) = rmfield(r2(iParam2,iParam1).(modelId{iModel}),'err');
            end
            r2(iParam2,iParam1).(modelId{iModel}).RT = finAvg(r2(iParam2,iParam1).(modelId{iModel}).RT);
            r2(iParam2,iParam1).(modelId{iModel}).Reward = finAvg(r2(iParam2,iParam1).(modelId{iModel}).Reward);
            r2(iParam2,iParam1).(modelId{iModel}).correct = finAvg(r2(iParam2,iParam1).(modelId{iModel}).correct);
        end
        r2(iParam2,iParam1).X0 = 0;  r2(iParam2,iParam1).Z = 0;  r2(iParam2,iParam1).Zmax = 0;
        r2(iParam2,iParam1).iZmax = 0;  r2(iParam2,iParam1).dX = 0;  r2(iParam2,iParam1).base = 0;
        r2(iParam2,iParam1).normalized = 0;
        r(iParam2,iParam1) = rmfield(r2(iParam2,iParam1), ...
                                      {'X0','Z','Zmax','iZmax','dX','base','normalized'});        
    end
end
clear r2;
toc;
rfilename = [mfilename '_' param1Label '_vs_' param2Label];
save(rfilename, 'r', 'param1Label', 'param2Label', 'param1', 'param2');
clear r;
fprintf('Results written to %s\n', rfilename);
toc;


%% 3) Urgency offset vs. nonlinearity:
param1Label = 'Offset';
param1  = offsetVary;
param2Label = 'Nonlinearity';
param2  = nlinVary;

for iParam2 = length(param2):-1:1
    fprintf('- %s = %d\n', param2Label, param2(iParam2));

    parfor iParam1 = 1:length(param1)
        % adjust parameters and evaluate performance
        p_ = p;
        p_.uc.a = param2(iParam2);
        p_.ur.u0 = param1(iParam1);              % RM 
        p_.uc.u0 = param1(iParam1);              % Full
        fprintf('\t- %s = %+2.2f\t', param1Label, param1(iParam1));
        r2(iParam2,iParam1) = modelPerformance(p_, commonDynamics(p_));
        % clean up to save memory
        for iModel = 1:length(modelId)
            r2(iParam2,iParam1).(modelId{iModel}) = rmfield(r2(iParam2,iParam1).(modelId{iModel}),...
                                                           {'X','D','iTEnd','cost','DFixedTime','RewardFixedTime','RRFixedTime','correctFixedTime'});
            if isfield(r2(iParam2,iParam1).(modelId{iModel}),'err')
                r2(iParam2,iParam1).(modelId{iModel}) = rmfield(r2(iParam2,iParam1).(modelId{iModel}),'err');
            end
            r2(iParam2,iParam1).(modelId{iModel}).RT = finAvg(r2(iParam2,iParam1).(modelId{iModel}).RT);
            r2(iParam2,iParam1).(modelId{iModel}).Reward = finAvg(r2(iParam2,iParam1).(modelId{iModel}).Reward);
            r2(iParam2,iParam1).(modelId{iModel}).correct = finAvg(r2(iParam2,iParam1).(modelId{iModel}).correct);
        end
        r2(iParam2,iParam1).X0 = 0;  r2(iParam2,iParam1).Z = 0;  r2(iParam2,iParam1).Zmax = 0;
        r2(iParam2,iParam1).iZmax = 0;  r2(iParam2,iParam1).dX = 0;  r2(iParam2,iParam1).base = 0;
        r2(iParam2,iParam1).normalized = 0;
        r(iParam2,iParam1) = rmfield(r2(iParam2,iParam1), ...
                                      {'X0','Z','Zmax','iZmax','dX','base','normalized'});        
    end
end
clear r2;
toc;
rfilename = [mfilename '_' param1Label '_vs_' param2Label];
save(rfilename, 'r', 'param1Label', 'param2Label', 'param1', 'param2');
clear r;
fprintf('Results written to %s\n', rfilename);
toc;


%% 4) Urgency slope vs. nonlinearity:
param1Label = 'Slope';
param1  = slopeVary;
param2Label = 'Nonlinearity';
param2  = nlinVary;

for iParam2 = length(param2):-1:1
    fprintf('- %s = %d\n', param2Label, param2(iParam2));
    
    parfor iParam1 = 1:length(param1)
        % adjust parameters and evaluate performance
        p_ = p;
        p_.uc.a = param2(iParam2);
        p_.ur.b = param1(iParam1);              % RM 
        p_.uc.b = param1(iParam1);              % Full
        fprintf('\t- %s = %+2.2f\t', param1Label, param1(iParam1));
        r2(iParam2,iParam1) = modelPerformance(p_, commonDynamics(p_));
        % clean up to save memory
        for iModel = 1:length(modelId)
            r2(iParam2,iParam1).(modelId{iModel}) = rmfield(r2(iParam2,iParam1).(modelId{iModel}),...
                                                           {'X','D','iTEnd','cost','DFixedTime','RewardFixedTime','RRFixedTime','correctFixedTime'});
            if isfield(r2(iParam2,iParam1).(modelId{iModel}),'err')
                r2(iParam2,iParam1).(modelId{iModel}) = rmfield(r2(iParam2,iParam1).(modelId{iModel}),'err');
            end
            r2(iParam2,iParam1).(modelId{iModel}).RT = finAvg(r2(iParam2,iParam1).(modelId{iModel}).RT);
            r2(iParam2,iParam1).(modelId{iModel}).Reward = finAvg(r2(iParam2,iParam1).(modelId{iModel}).Reward);
            r2(iParam2,iParam1).(modelId{iModel}).correct = finAvg(r2(iParam2,iParam1).(modelId{iModel}).correct);
        end
        r2(iParam2,iParam1).X0 = 0;  r2(iParam2,iParam1).Z = 0;  r2(iParam2,iParam1).Zmax = 0;
        r2(iParam2,iParam1).iZmax = 0;  r2(iParam2,iParam1).dX = 0;  r2(iParam2,iParam1).base = 0;
        r2(iParam2,iParam1).normalized = 0;
        r(iParam2,iParam1) = rmfield(r2(iParam2,iParam1), ...
                                      {'X0','Z','Zmax','iZmax','dX','base','normalized'});        
    end
end
clear r2;
toc;
rfilename = [mfilename '_' param1Label '_vs_' param2Label];
save(rfilename, 'r', 'param1Label', 'param2Label', 'param1', 'param2');
clear r;
fprintf('Results written to %s\n', rfilename);
toc;


function r = commonDynamics(p)
r.X0 = 0.00001*randn(p.nTrial, p.N, 1);        % A small noise to avoid waiting just on the boundary)
r.Z = mvnrnd(p.meanZ, p.covZ, p.nTrial);
[r.Zmax, r.iZmax] = max(r.Z,[],2);
for iT = length(p.t):-1:1
    r.dX(:,:,iT) = mvnrnd(p.dt*r.Z, p.dt*p.covX, p.nTrial);
end
r.base.X(:,:,1) = r.X0;
for iT = 2:length(p.t)
    r.base.X(:,:,iT) = r.base.X(:,:,iT-1) + r.dX(:,:,iT);
    icovZh = p.icovZ + p.t(iT)*p.icovX;
    r.base.Zh(:,:,iT) = (repmat(p.meanZ,[p.nTrial 1])*p.icovZ + r.base.X(:,:,iT)*p.icovX)/icovZh;     % Hidden state estimates
end
constrain = @(X) X - repmat(mean(X,2), [1 p.N 1]);
r.normalized.X = constrain(r.base.X);


function r = modelPerformance(p, rCommon)
tStart = tic;
r = rCommon;
r.ur  = simulateDiffusion3(r, p, 'URM');       % RM with urgency alone
r.uc  = simulateDiffusion3(r, p, p.UCRMType);  % Urgency-constrained race
r.lu = simulateDiffusion3(r, p, 'LUCRM');      % Urgency-constrained linear-normalization race
toc(tStart);
