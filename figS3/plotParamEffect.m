function plotParamEffect()

dataPrefix = 'computeParamEffect_';
dataSuffix = '';

%% 1) Urgency offset vs. accumulation cost:
load([dataPrefix 'Offset_vs_AccumulationCost' dataSuffix '.mat']);
figureFull(1,4,1);
showResults(r, param1Label, param2Label, param1, param2);
saveInPDF(gcf, ['computeParamEffect_' param1Label '_vs_' param2Label]);
showOptimParam(r, param1Label, param2Label, param1, param2);
clear r;

%% 2) Urgency slope vs. accumulation cost:
load([dataPrefix 'Slope_vs_AccumulationCost' dataSuffix '.mat']);
figureFull(1,4,2);
showResults(r, param1Label, param2Label, param1, param2);
saveInPDF(gcf, ['computeParamEffect_' param1Label '_vs_' param2Label]);
showOptimParam(r, param1Label, param2Label, param1, param2);
clear r;

%% 3) Urgency offset vs. nonlinearity:
load([dataPrefix 'Offset_vs_Nonlinearity' dataSuffix '.mat']);
figureFull(1,4,3);
showResults(r, param1Label, param2Label, param1, param2);
saveInPDF(gcf, ['computeParamEffect_' param1Label '_vs_' param2Label]);
showOptimParam(r, param1Label, param2Label, param1, param2);
clear r;

%% 4) Urgency slope vs. nonlinearity:
load([dataPrefix 'Slope_vs_Nonlinearity' dataSuffix '.mat']);
figureFull(1,4,4);
showResults(r, param1Label, param2Label, param1, param2);
saveInPDF(gcf, ['computeParamEffect_' param1Label '_vs_' param2Label]);
showOptimParam(r, param1Label, param2Label, param1, param2);
clear r;


function showResults(r,param1Label, param2Label, param1, param2)
modelIdLong = {'Full model','RM (+urgency)','Random'};
modelId = {'uc' , 'ur', 'lu', 'rnd'};
mkr     = {'o-', 's-', '--'};
col     = {[1 0 0],[0 0.5 0],[0 0 0]};
mkrsize = 3;

putFigureLabel(param2Label);

for iModel = 1:2
    subplotXY(5,2,1,iModel); hold on; xlim(axisMinMax(param1)); xlabel(param1Label);
    for iParam2 = 1:length(param2)
        RR_(:,iParam2) = mkvector(r(iParam2,:),@(r_,iParam1) r_(iParam1).(modelId{iModel}).RR);
        plot(param1, RR_(:,iParam2), mkr{iModel}, 'MarkerSize', mkrsize, 'Color',sat(iParam2/length(param2),col{iModel}));
    end
%     plot(param1, mkvector(r(iParam2,:),@(r_,iParam1) r_(iParam1).(modelId{3}).RR), mkr{iModel}, 'MarkerSize', mkrsize, 'Color',sat(iParam2/length(param2),col{3}));
%     for iParam2 = 1:length(param2)
%         plot(param1, mkvector(r(iParam2,:),@(r_,iParam1) r_(iParam1).rnd.RR), mkr{3}, 'MarkerSize', mkrsize, 'Color',col{3});
%     end
    if iModel==1;   ylabel('(Reward - cost) / time');  moveObject(legend(modelIdLong,'Location','Northwest','FontSize',7),[-1.6 0]); legend boxoff;  end
    ylim([max(0,min(RR_(:))) max(RR_(:))]);
    title(modelIdLong{iModel});
    moveObject(legend(num2str(param2')),[0 1.5]); legend boxoff;
    
    subplotXY(5,2,2,iModel); hold on; xlim(axisMinMax(param1)); xlabel(param1Label);
    for iParam2 = 1:length(param2)
        plot(param1, mkvector(r(iParam2,:),@(r_,iParam1) r_(iParam1).(modelId{iModel}).CR), mkr{iModel}, 'MarkerSize', mkrsize, 'Color',sat(iParam2/length(param2),col{iModel}));
%         plot(param1, mkvector(r(iParam2,:),@(r_,iParam1) r_(iParam1).rnd.CR), mkr{3}, 'MarkerSize', mkrsize, 'Color',col{3});
    end
    if iModel==1;  ylabel('(Correct - cost) / time');  end
    
    subplotXY(5,2,3,iModel); hold on; xlim(axisMinMax(param1)); xlabel(param1Label);
    for iParam2 = 1:length(param2)
        plot(param1, mkvector(r(iParam2,:),@(r_,iParam1) r_(iParam1).(modelId{iModel}).RT), mkr{iModel}, 'MarkerSize', mkrsize, 'Color',sat(iParam2/length(param2),col{iModel}));
%         plot(param1, mkvector(r(iParam2,:),@(r_,iParam1) r_(iParam1).rnd.RT), mkr{3}, 'MarkerSize', mkrsize, 'Color',col{3});
    end
    if iModel==1;  ylabel('RT');  end
    
    subplotXY(5,2,4,iModel); hold on; xlim(axisMinMax(param1)); xlabel(param1Label);
    for iParam2 = 1:length(param2)
        plot(param1, mkvector(r(iParam2,:),@(r_,iParam1) r_(iParam1).(modelId{iModel}).Reward), mkr{iModel}, 'MarkerSize', mkrsize, 'Color',sat(iParam2/length(param2),col{iModel}));
%         plot(param1, mkvector(r(iParam2,:),@(r_,iParam1) r_(iParam1).rnd.Reward), mkr{3}, 'MarkerSize', mkrsize, 'Color',col{3});
    end
    if iModel==1;  ylabel('Reward / trial');  end
    
    subplotXY(5,2,5,iModel); hold on; xlim(axisMinMax(param1)); xlabel(param1Label);
    for iParam2 = 1:length(param2)
        plot(param1, mkvector(r(iParam2,:),@(r_,iParam1) r_(iParam1).(modelId{iModel}).correct), mkr{iModel}, 'MarkerSize', mkrsize, 'Color',sat(iParam2/length(param2),col{iModel}));
%         plot(param1, mkvector(r(iParam2,:),@(r_,iParam1) r_(iParam1).rnd.correct), mkr{3}, 'MarkerSize', mkrsize, 'Color',col{3});
    end
    if iModel==1;  ylabel('Correct / trial');  end
end


function showOptimParam(r, param1Label, param2Label, param1, param2)
modelIdLong = {'Full model','RM (+urgency)','Random'};
modelId = {'uc' , 'ur', 'lu', 'rnd'};
mkr     = {'o-', 's-', '--'};
col     = {[0 0 0],[0 0 0],[0 0 0]};
mkrsize = 6;
lwidth  = 2;


for iModel = 1:2
    % collect reward/correct rates
    for iParam2 = 1:length(param2)
        RR_(:,iParam2) = mkvector(r(iParam2,:),@(r_,iParam1) r_(iParam1).(modelId{iModel}).RR);
        CR_(:,iParam2) = mkvector(r(iParam2,:),@(r_,iParam1) r_(iParam1).(modelId{iModel}).CR);
    end
    % plot reward rate image and maximum of these points over image
    figure('Color','white');
    imagesc(param2,param1,RR_);  hold on;
    [~,iRRmax] = max(RR_,[],1);
    plot(param2,param1(iRRmax),mkr{iModel},'MarkerSize',mkrsize,'Color',col{iModel},'LineWidth',lwidth);
    colormap hot;
    c = colorbar; c.Label.String = '(Reward - cost) / time';
    xlim(axisMinMax(param2));  ylim(axisMinMax(param1));
    xlabel(param2Label);  ylabel(param1Label);
    title(modelIdLong{iModel})
    % plot correct rate image and maximum of these points over image
    figure('Color','white');
    imagesc(param2,param1,CR_);  hold on;
    [~,iRRmax] = max(CR_,[],1);
    plot(param2,param1(iRRmax),mkr{iModel},'MarkerSize',mkrsize,'Color',[0 0 0],'LineWidth',lwidth);
    colormap hot;
    c = colorbar; c.Label.String = '(Correct - cost) / time';
    xlim(axisMinMax(param2));  ylim(axisMinMax(param1));
    xlabel(param2Label);  ylabel(param1Label);
    title(modelIdLong{iModel})
end


function y = mkvector(r, func)
try
    for iParam1 = length(r):-1:1
        y(iParam1) = func(r,iParam1);  
    end
catch ME
    display(ME);
end

