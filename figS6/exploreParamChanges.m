function exploreParamChanges()
%% explores how performance in UCRM model changes with parameters

%% settings
% model settings
model = 'UCRM';
N = 3;
optimParamsFolder = 'optimParams';
objFns = {'RR', 'CR'};
exploreParamChangesFolder = 'exploreParamChanges';
% parameter ranges to explore
pVary = struct(...
    'u0', 0.25:0.05:0.95, ...
    'b' , 0:0.25:5, ...
    'a' , 1:0.5:6, ...
    'c' , 0:0.5:5);
% parameter combinations
compNames = {'u0_vs_c', 'b_vs_c', 'u0_vs_a', 'b_vs_a'};
compParams = {'u0' 'c'; 'b' 'c'; 'u0' 'a'; 'b' 'a'};


%% load optimal parameters
for iobj = 1:length(objFns)
    optimParamFile = [optimParamsFolder filesep model '_' objFns{iobj} '_N' num2str(N) '.mat'];
    doptim.(objFns{iobj}) = load(optimParamFile);
end


%% iterate over objective functions and parameter comparisons
fprintf('Parameter exploration, model %s, N=%d\n\n', model, N);
for iobj = 1:length(objFns)
    pPerf = struct();
    objFn = objFns{iobj};
    for icomp = 1:length(compNames)
        fprintf('Objective %s, %s\n', objFn, compNames{icomp});
        i1Vary = pVary.(compParams{icomp, 1});
        i2Vary = pVary.(compParams{icomp, 2});
        i1Num = length(i1Vary);
        i2Num = length(i2Vary);
        compPerf = cell(i1Num, i2Num);
        parfor i1 = 1:i1Num
            fprintf('%s = %f, %s = %f..%f\n', ...
                compParams{icomp, 1}, i1Vary(i1), ...
                compParams{icomp, 2}, min(i2Vary), max(i2Vary));
            for i2 = 1:i2Num
                % set parameters
                p = setBaseParams(doptim.(objFn));
                if compParams{icomp, 1} == 'c'
                    p.task.cpers = i1Vary(i1);
                else
                    p.model.(compParams{icomp, 1}) = i1Vary(i1);
                end
                if compParams{icomp, 2} == 'c'
                    p.task.cpers = i2Vary(i2);
                else
                    p.model.(compParams{icomp, 2}) = i2Vary(i2);
                end
                p = baseParameters(p); % update in case cost changed
                % run simulation and store summary
                compPerf{i1, i2} = simulateDiffusion(p, model, true);
            end
        end
        pPerf.(compNames{icomp}) = compPerf;  % to get parfor to work
        fprintf('\n');
    end
    outFile = [exploreParamChangesFolder filesep model '_' objFn '_N' num2str(N) '.mat'];
    save(outFile, 'model', 'N', 'objFn', 'pVary', 'compNames', 'compParams', 'pPerf');
    fprintf('Model performance written to %s\n\n', outFile);
end


function p = setBaseParams(doptim)
%% sets base parameters according to the ones specified in doptim

p = baseParameters();
% set optimal parameters according to doptim
p.task.N = doptim.N;
for iParam = 1:length(doptim.paramNames)
    p.model.(doptim.paramNames{iParam}) = doptim.opttheta(iParam);
end
% adjust other structures that depend on N
p = baseParameters(p);
