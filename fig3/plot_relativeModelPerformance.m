function plot_relativeModelPerformance(taskType)
%% Plots relative model performance after optimal params for all are found
% Takes in one argument, taskType, which can assume one of two values:
% 'p': perceptual
% 'v': value-based (default)
% This function uses the pre-optimised model parameters to plot the
% relative reward rate.
%
% Currently, this function is setup to plot the parameters used for the
% figure in the paper. Thus, on line 49, it accesses the files from
% ../shared/optimParams_paper/. 


%%
% Check number of inputs.
if nargin > 1
    error('TooManyInputs', 'requires at most 1 optional input');
end

% By default, decision taskType == perceptual
if nargin == 0
    taskType = 'v';
elseif (taskType ~= 'p' && taskType ~= 'v')
    fprintf(2,'Invalid task type. Plotting for value-based task.\n')
    taskType = 'v';
end

% Objective function to optimize:
switch taskType
    case 'p'            % perceptual
        obj = 'CR';     % correct rate
    case 'v'            % value-based
        obj = 'RR';     % reward rate
end



model   = {'RM','URM','UCRM','CRM'};
N       = [2 3 4 6 8];
% for iModel = 1:length(model)
%     for iN = 1:length(N)
%         optimParams_percep(model{iModel},N(iN),obj);
%     end
% end



%% Plot
% 
cd('../shared/optimParams_paper/');     % ('./optimParams/')
clear perf

% Fetching the optimal performance measures for each model
N = [2 3 4 6 8];
for iN = 1:length(N)
    str_file  = ['*' obj '_N' num2str(N(iN)) '*.mat'];
    filenames = dir(str_file);
    for iModel = 1:length(filenames)
        load(filenames(iModel).name,'optobj')
        perf(iN,iModel) = optobj; %#ok<AGROW>
    end
end
cd ..
clearvars -except perf N filenames mom_noise taskType


% Organizing performance matrix for plotting
switch taskType
    case 'p'
        perf_rand = (2./N)';        % performance for random choice
        perf = [perf, perf_rand];
        clear perf_rand;
        perf = perf-perf(:,end);    % perf relative to random choice (sub.)
        perf = perf(:,1:end-1);
        perf = perf./perf(:,3);     % perf relative to full model (divided)
    case 'v'
        perf = perf - 2;            % perf relative to random choice (sub.)
        perf = perf./perf(:,3);     % perf relative to full model (divided)
end


% Plot
figure(); hold on;
for iModel=1:length(filenames)
    plot(N,perf(:,iModel),'linewidth',2);
    hold on
end
% Formatting
xticks(N);
ylim([-inf 1]);
xlabel('Number of options')
ylabel('Relative reward rate')
switch taskType
    case 'v'
        title('Value-based task')
    case 'p'
        title('Perceptual task')
end
set(gca,'FontSize',20)

% Changing legend order
    labels = get(legend(), 'String');
    plots = flipud(get(gca, 'children'));
    % Now re-create the legend
    neworder = [3 1 4 2];
    legend(plots(neworder), labels(neworder))
    legend('Full model','RM + constraint only','RM + urgency only',...
            'Race Model (RM)','Location','southeast')