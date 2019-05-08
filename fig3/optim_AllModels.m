%% Models to optimize for each N, task type
model   = {'RM', 'URM', 'UCRM','CRM'};
N       = [2 3 4 6 8];
noise   = 0;
obj     = 'RR';

%% optimizing all models
for iN = 1:length(N)
    for iModel = 1:length(model)
        optimParams(model{iModel},N(iN),obj, noise);
    end
end