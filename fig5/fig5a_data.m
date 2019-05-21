%% Plotting figure 5a (data)
% Data grabbed from Louie et al 2011 PNAS, Figure 1e, using manual data-
% grabbing tool, WebPlotDigitizer (https://apps.automeris.io/wpd/).

%% Data:
v_out       = [129.5455, 161.3636, 195.4545, 227.2727, 259.0909]; % ul
firingRate  = [0.4836,   0.4570,   0.4279,   0.3745,   0.3745];   % norm
errorbars   = [0.0485,   0.0461,   0.0412,   0.0364,   0.0436];

%% Plot
%scatter(v_out, firingRate)
errorbar(v_out, firingRate, errorbars, 'o','MarkerEdgeColor',[.2 .6 1],...
    'MarkerFaceColor',[.2 .6 1], 'MarkerSize',25,'LineWidth',2)
xlim([0,400])
ylim([0.2,0.6])
xticks([0,400])
yticks([0.2,0.6])
xlabel('V_{out} magnitude (\mu\it{l})')
ylabel('Firing rate (norm)')