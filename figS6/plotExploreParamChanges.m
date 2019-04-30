function plotExploreParamChanges()
%% plots results of exploreParamChanges


%% settings
% model
model = 'UCRM';
N = 3;
objFns = {'RR', 'CR'};
exploreParamChangesFolder = 'exploreParamChanges';
% plotting
plotmkr = '-o';
mkrsize = 6;
plotbasecol = [0 0 0];
lwidth = 2;
paramcol = struct('u0', [0.8 0 0], 'b', [0.8 0 0.8], ...
                  'a', [0 0.8 0], 'c', [0 0 0.8]);


%% plot results separately per objective function
for iobj = 1:length(objFns)
    objFn = objFns{iobj};
    d = load([exploreParamChangesFolder filesep ...
        model '_' objFn '_N' num2str(N) '.mat']);
    compNames = d.compNames;
    compParams = d.compParams;
    pVary = d.pVary;
    
    %% plot of objective over parameter pairs / parameters
    for icomp = 1:length(compNames)
        % collect data
        compPerf = d.pPerf.(compNames{icomp});
        i1Vary = pVary.(compParams{icomp, 1});
        i2Vary = pVary.(compParams{icomp, 2});
        perf = NaN(length(i1Vary), length(i2Vary));
        for i1 = 1:length(i1Vary)
            for i2 = 1:length(i2Vary)
                perf(i1, i2) = compPerf{i1,i2}.(objFn);
            end
        end
        % plot heat map
        figure('Color','white');
        imagesc(i2Vary, i1Vary, perf);  hold on;
        [~,iperfmax] = max(perf, [], 1);
        plot(i2Vary, i1Vary(iperfmax), plotmkr, ...
            'MarkerSize', mkrsize,'Color', plotbasecol, 'LineWidth',lwidth, ...
            'MarkerFaceColor', plotbasecol);
        colormap hot;
        c = colorbar; c.Label.String = objFn;
        xlim([min(i2Vary) max(i2Vary)]);  ylim([min(i1Vary) max(i1Vary)]);
        xlabel(compParams{icomp, 2});  ylabel(compParams{icomp, 1});
        set(gca,'Ydir','Normal');
        title(sprintf('%s N=%d obj=%s', model, N, objFn));
        % plot separate color plots
        figure('Color','white');  hold on;
        for i2 = 1:length(i2Vary)
            plot(i1Vary, perf(:, i2)', 'LineWidth', lwidth, ...
                'Color', sat(i2 / length(i2Vary), paramcol.(compParams{icomp, 2})));
        end
        xlim([min(i1Vary) max(i1Vary)]);
        xlabel(compParams{icomp, 1});  ylabel(objFn);
        title(sprintf('%s N=%d obj=%s, %s vs. %s', model, N, objFn, ...
            compParams{icomp, 1}, compParams{icomp, 2}));        
    end
end
