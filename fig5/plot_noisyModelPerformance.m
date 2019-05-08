function plot_noisyModelPerformance()
%% Plots relative model performance in presence of noise
% Just something that works for now. Needs better organization, etc.


%% Plotting
    startDir = pwd;
    cd('../shared/optimParamsNoisy/')
    
    N = [2 3 4 6 8];
    for iN = 1:length(N)
        str_file  = ['*N', num2str(N(iN)),'noise0.25_fitSigH.mat'];
        filenames = dir(str_file);
        for iModel = 1:length(filenames)
            load(filenames(iModel).name,'optobj')
            perf(iN,iModel) = optobj;
            if iN==1 & strfind(filenames(iModel).name,'UCRM')
                iUCRM = iModel;
            end
        end
    end
    perf = perf-2;
    perf = perf./perf(:,iUCRM);

    figure(); hold on;
    for iModel=1:4 %length(filenames)
        plot(N,perf(:,iModel),'linewidth',4);
        hold on
    end
    % Formatting
    xticks(N);    
    box on
    lgd = legend({'CRM','RM','UCRM','URM'},'Location','southeast');
    clear filenames i* str_file perf optobj lgd N
    %}

    % Changing legend order
    labels = get(legend(), 'String');
    plots = flipud(get(gca, 'children'));
    % Now re-create the legend
    neworder = [3 1 4 2];
    legend(plots(neworder), labels(neworder))
    
    cd(startDir)