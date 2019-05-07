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

    %% plot stuff
    %{
    % cd ./optimParams/
    % N = [2 3 4 6 8];
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
    for iModel=1:5 %length(filenames)
        plot(N,perf(:,iModel),'linewidth',4);
        hold on
    end
    % Formatting
    xticks(N);    
    box on
    lgd = legend({'NRM','RM','UCRM','UNRM','URM'},'Location','southeast');
    clear filenames i* str_file perf optobj lgd N
    %}
    
    %{
    model   = {'NRM','RM','UCRM','CRM','URM'}; %
    N = [2 3 4 6 8];
    for iN = 1:length(N)
        for iModel = 1:length(model)
            fileName  = [model{iModel}, '_RR_N', num2str(N(iN)),'.mat'];
            load(fileName,'optobj')
            perf(iN,iModel) = optobj;
            if iN==1 & strfind(model{iModel},'UCRM')
                iUCRM = iModel;
            end
        end
    end
    
    % Changing legend order
    labels = get(legend(), 'String');
    plots = flipud(get(gca, 'children'));
    % Now re-create the legend
    neworder = [3 1 5 2 4];
    legend(plots(neworder), labels(neworder))
    %}