function valueDecisionBoundaryRR2_3D()
tic;
Smax = 4;      % Grid range of states space (now we assume: S = [(Rhat1+Rhat2)/2, (Rhat1-Rhat2)/2]); Rhat(t) = (varR*X(t)+varX)/(t*varR+varX) )
resSL  = 15;      % Grid resolution of state space
resS = 41;      % Grid resolution of state space
tmax = 3;       % Time limit
dt   = .05;       % Time step
c    = 0;       % Cost of evidene accumulation
tNull = .25;     % Non-decision time + inter trial interval
g{1}.meanR = 0; % Prior mean of state (dimension 1)
g{1}.varR  = 5; % Prior variance of stte
g{1}.varX  = 2; % Observation noise variance
g{2}.meanR = 0; % Prior mean of state (dimension 2)
g{2}.varR  = 5; % Prior variance of state
g{2}.varX  = 2; % Observation noise variance
g{3}.meanR = 0; % Prior mean of state (dimension 3)
g{3}.varR  = 5; % Prior variance of state
g{3}.varX  = 2; % Observation noise variance
t = 0:dt:tmax;
Slabel = {'r_1^{hat}', 'r_2^{hat}', 'r_3^{hat}'};

%% Utililty function:
utilityFunc = @(X) X;
% utilityFunc = @(X) tanh(X);
% utilityFunc = @(X) sign(X).*abs(X).^0.5;

%% Reward rate, Average-adjusted value, Decision (finding solution):
SscaleL  = linspace(-Smax, Smax, resSL);
[S{1},S{2},S{3}] = ndgrid(SscaleL, SscaleL, SscaleL);
iS0 = [findnearest(g{1}.meanR, SscaleL) findnearest(g{2}.meanR, SscaleL) findnearest(g{3}.meanR, SscaleL)];
for iC = 3:-1:1;  Rh{iC} = utilityFunc(S{iC});  end                                                          % Expected reward for option iC
[RhMax, Dd] = max_({Rh{1}, Rh{2}, Rh{3}});                                                                   % Expected reward for decision
[V0, V, D, EVnext, rho, Ptrans, iStrans] = backwardInduction(g{1}.meanR,c,tNull,g,Rh,S,t,dt,iS0);
rho_ = fzero(@(rho) backwardInduction(rho,c,tNull,g,Rh,S,t,dt,iS0), g{1}.meanR);                            % Reward rate

%% Reward rate, Average-adjusted value, Decision (high resolution):
Sscale = linspace(-Smax, Smax, resS);
[S{1},S{2},S{3}] = ndgrid(Sscale, Sscale, Sscale);
iS0 = [findnearest(g{1}.meanR, Sscale) findnearest(g{2}.meanR, Sscale) findnearest(g{3}.meanR, Sscale)];
for iC = 3:-1:1;  Rh{iC} = utilityFunc(S{iC});  end                                                          % Expected reward for option iC
[RhMax, Dd] = max_({Rh{1}, Rh{2}, Rh{3}});                                                                   % Expected reward for decision
[V0, V, D, EVnext, rho, Ptrans, iStrans] = backwardInduction(rho_,c,tNull,g,Rh,S,t,dt,iS0);                  % Average-adjusted value, Decision, Transition prob. etc.

%% Transform to the space of accumulated evidence:
% dbX = transformDecBound(dbS2,Sscale,t,g);

%% - Show -
figure(4565); clf; colormap bone;
iS2 = findnearest(.5, Sscale, -1);
iS3 = 1;
iTmax = length(t);
rect = [-1 1 -1 1 -2.3 .5];
myCol = [1 0 0; 0 1 0; 0 0 1];
%% t=0:
iT = 1;
subplotXY(5,4,2,1); [r1Max,r2Max,vMax] = plotSurf(Sscale, V(:,:,iS3,iT)                , iS2, [0 0 0], Slabel); axis(rect); title('V(0)');
%                     plot3(g{1}.meanR, g{2}.meanR, V0, 'g.', 'MarkerSize',15);
subplotXY(5,4,3,1); [r1Acc,r2Acc,vAcc] = plotSurf(Sscale, EVnext(:,:,iS3,iT)-(rho+c)*dt, iS2, [1 0 0], Slabel); axis(rect); title('<V(\deltat)|R^{hat}(0)> - (\rho+c)\deltat');
subplotXY(5,4,4,1); [r1Dec,r2Dec,vDec] = plotSurf(Sscale, RhMax(:,:,iS3)-rho*tNull     , iS2, [0 0 1], Slabel); axis(rect); title('max(R_1^{hat},R_2^{hat}) - \rho t_{Null}');
subplotXY(5,4,5,1); hold on;
    plot((r1Max-r2Max)/2, vMax, 'k:', (r1Acc-r2Acc)/2, vAcc, 'r', (r1Dec-r2Dec)/2, vDec, 'b');
    xlabel(['(' Slabel{1} '-' Slabel{2} ')/2']); xlim(rect(1:2)); %ylim(rect(5:6));
subplotXY(5,4,1,1); imagesc(Sscale, Sscale, D(:,:,  1), [1 4]); axis square; axis xy; title(['D(0) \rho=' num2str(rho_,3)]); xlabel(Slabel{1}); ylabel(Slabel{2}); hold on; axis(rect(1:4));
                    plot(r1Max, r2Max, 'r-');
%                     plot(g{1}.meanR, g{2}.meanR, 'g.');
%% t=0 (superimposed & difference):
subplotXY(5,4,3,2); plotSurf(Sscale, EVnext(:,:,iS3,iT)-(rho+c)*dt                             , iS2, [1 0 0], Slabel); hold on;
                    plotSurf(Sscale, RhMax(:,:,iS3)-rho*tNull                                  , iS2, [0 0 1], Slabel); axis(rect);
subplotXY(5,4,4,2); plotSurf(Sscale, RhMax(:,:,iS3)-rho*tNull - (EVnext(:,:,iS3,iT)-(rho+c)*dt), iS2, [0 1 0], Slabel); xlim(rect(1:2)); ylim(rect(1:2));
subplotXY(5,4,5,2); plotDecisionVolume(S, D(:,:,:,iT), rect(1:2));
%% t=dt:
subplotXY(5,4,1,2); imagesc(Sscale, Sscale, D(:,:,iS3,iT+1), [1 4]); axis square; axis xy; title('D(\deltat)'); xlabel(Slabel{1}); ylabel(Slabel{2}); hold on; axis(rect(1:4));
subplotXY(5,4,2,2); plotSurf(Sscale, V(:,:,iS3,iT+1), iS2, [0 0 0], Slabel); axis(rect); title('V(\deltat)');
%% t=T-dt:
% subplotXY(5,4,3,2); surfl(Sscale(iStrans{1}{2}), Sscale(iStrans{1}{2}), Ptrans{1}); title('P(R^{hat}(\deltat)|R^{hat}(0))'); shading interp; hold on; axis([rect 0 Inf]); axis off;
subplotXY(5,4,1,3); imagesc(Sscale, Sscale, D(:,:,iS3,iTmax-1), [1 4]); axis square; axis xy; title('D(T-\deltat)'); hold on; rectangle('Position',[rect(1) rect(3) rect(2)-rect(1) rect(4)-rect(3)]); axis(rect);
subplotXY(5,4,2,3); [r1Max,r2Max,vMax] = plotSurf(Sscale, V(:,:,iS3,iTmax-1)                , iS2, [0 0 0], Slabel); axis(rect); title('V(T-\deltat)')
subplotXY(5,4,3,3); [r1Acc,r2Acc,vAcc] = plotSurf(Sscale, EVnext(:,:,iS3,iTmax-1)-(rho+c)*dt, iS2, [1 0 0], Slabel); axis(rect); title('<V(T)|R^{hat}(T-\deltat)> - (\rho+c) \deltat');
subplotXY(5,4,4,3); [r1Dec,r2Dec,vDec] = plotSurf(Sscale, RhMax(:,:,iS3)-rho*tNull          , iS2, [0 0 1], Slabel); axis(rect); title('max(R_1^{Hat},R_2^{Hat}) - \rho t_{Null}');
subplotXY(5,4,5,3); hold on;
    plot((r1Max-r2Max)/2, vMax, 'k:', (r1Acc-r2Acc)/2, vAcc, 'r', (r1Dec-r2Dec)/2, vDec, 'b');
    xlabel(['(' Slabel{1} '-' Slabel{2} ')/2']); xlim(rect(1:2)); %ylim(rect(5:6));
subplotXY(5,4,5,4); plotDecisionVolume(S, D(:,:,:,iTmax-1), rect(1:2));
%% t=T:
subplotXY(5,4,1,4); imagesc(Sscale, Sscale, D(:,:,iS3,iTmax), [1 4]); axis square; axis xy; title('D(T)'); hold on; axis(rect(1:4));
subplotXY(5,4,2,4); plotSurf(Sscale, V(:,:,iS3,iTmax), iS2, [0 0 0], Slabel); title('V(T) = max(R_1^{hat},R_2^{hat}) - \rho t_{Null}'); axis(rect);
% subplotXY(5,4,3,4); surfl(Sscale(iStrans{iTmax-1}{2}), Sscale(iStrans{iTmax-1}{1}), Ptrans{iTmax-1}); title('P(R^{hat}(T)|R^{hat}(T-\deltat))'); shading interp; hold on; axis([rect 0 Inf]); axis off;


%% Decision boundaries superimposed:
figure(4566); clf;
iT = [1 3 11 41];
for iiT = 1:length(iT)
    subplot(2,1,1);
        plotDecisionVolume(S, D(:,:,:,iT(iiT)), rect(1:2), [myCol, [0.3; 0.05; 0.05]]); hold on;
    subplotXY(4,length(iT),3,iiT);
        dbIS = plotDecisionVolume(S, D(:,:,:,iT(iiT)), rect(1:2), [myCol, [0.5;0.5;0.5]]); hold on;
        title(['Time=' num2str(t(iT(iiT)))]);
    subplotXY(4,length(iT),4,iiT);
        patch([-sqrt(2) sqrt(2) 0 -sqrt(2)], [-1/sqrt(3) -1/sqrt(3) sqrt(3) -1/sqrt(3)],'w', 'EdgeColor',0.5*[1 1 1]);
        for iD = 1:3
            line([dbIS{iD}.vertices2d(dbIS{iD}.edges(:,1),1), dbIS{iD}.vertices2d(dbIS{iD}.edges(:,2),1)]',...
                 [dbIS{iD}.vertices2d(dbIS{iD}.edges(:,1),2), dbIS{iD}.vertices2d(dbIS{iD}.edges(:,2),2)]',...
                 [dbIS{iD}.vertices2d(dbIS{iD}.edges(:,1),3), dbIS{iD}.vertices2d(dbIS{iD}.edges(:,2),3)]', ...
                 'Color',myCol(iD,1:3),'LineWidth',1.5); hold on;
        end
        axis([-2 2 -2 2 -1 1]); view([0 90]); axis off;
end
toc;


function [V0, V, D, EVnext, rho, Ptrans, iStrans] = backwardInduction(rho_,c,tNull,g,Rh,S,t,dt,iS0)
rho = rho_;                                                                        % Reward rate estimate
[V(:,:,:,length(t)), D(:,:,:,length(t))] = max_({Rh{1}-rho*tNull, Rh{2}-rho*tNull, Rh{3}-rho*tNull});       % Max V~ at time tmax
for iT = length(t)-1:-1:1
    [EVnext(:,:,:,iT), Ptrans{iT}, iStrans{iT}] = E(V(:,:,:,iT+1),S,t(iT),dt,g);                            % <V~(t+1)|S(t)> for waiting
    [V(:,:,:,iT), D(:,:,:,iT)] = max_({Rh{1}-rho*tNull, Rh{2}-rho*tNull, Rh{3}-rho*tNull, EVnext(:,:,:,iT)-(rho+c)*dt});       % [Average-adjusted value (V~), decision] at time t
%     fprintf('%d/%d\t',iT,length(t)-1); toc;
end
V0 = mean(vector(V(iS0(1),iS0(2),1)));
D(D==0) = 4;
fprintf('rho = %d\tV0 = %d\t', rho_, V0); toc;

function [EV, Ptrans, iStrans] = E(V,S,t,dt,g)
aSscale = abs(S{1}(:,1,1));
CR = [g{1}.varR 0 0; 0 g{2}.varR 0; 0 0 g{3}.varR];
CX = [g{1}.varX 0 0; 0 g{2}.varX 0; 0 0 g{3}.varX];
for k = 1:3
    g{k}.varRh = g{k}.varR * g{k}.varX / (t * g{k}.varR + g{k}.varX);
    v{k} = varTrans(g{k}.varRh, g{k}.varR, g{k}.varX, t, dt);
    iStrans{k} = find(aSscale<3*sqrt(v{k}));
end
Ptrans = normal3({S{1}(iStrans{1},iStrans{2},iStrans{3}), S{2}(iStrans{1},iStrans{2},iStrans{3}), S{3}(iStrans{1},iStrans{2},iStrans{3})}, [0 0 0], [v{1} 0 0; 0 v{2} 0; 0 0 v{3}]);
mgn = ceil(size(Ptrans)/2);
% V = extrap(V,mgn,[5 5 5]);
EV = convn(V,Ptrans,'same') ./ convn(ones(size(V)),Ptrans,'same');
% EV = EV(mgn(1)+1:end-mgn(1), mgn(2)+1:end-mgn(2), mgn(3)+1:end-mgn(3));

function v = varTrans(varRh, varR, varX, t, dt)
% v = (varR * (varX + varRh)) / ((1 + t/dt) * varR + varX / dt);
v = (varR ./ (varR*(t+dt) + varX)).^2 .* (varX + varRh * dt) * dt;

function prob = normal3(x, m, C)
d1 = x{1} - m(1);
d2 = x{2} - m(2);
d3 = x{3} - m(3);
H = -1/2*(C\eye(3)); prob = exp(d1.*d1*H(1,1) + d1.*d2*H(1,2) + d1.*d3*H(1,3) + ...
                                d2.*d1*H(2,1) + d2.*d2*H(2,2) + d2.*d3*H(2,3) + ...
                                d3.*d1*H(3,1) + d3.*d2*H(3,2) + d3.*d3*H(3,3));
% prob = exp(-(d1.^2/C(1,1)/2 + d2.^2/C(2,2))/2);
prob = prob ./ sum(prob(:));

function [V, D] = max_(x)
x_ = zeros(size(x{1},1), size(x{1},2), size(x{1},3), length(x));
for k = 1:length(x)
    x_(:,:,:,k) = x{k};
end
[V, D] = max(x_,[],4);
D(D==1 & x{1}==x{2} & x{2}==x{3}) = 123;
D(D==1 & x{1}==x{2}) = 12;
D(D==2 & x{2}==x{3}) = 23;
D(D==1 & x{3}==x{1}) = 13;

function [x_,y_,v_] = plotSurf(Sscale, Val, iS, col, Slabel)
[x,y] = meshgrid(1:length(Sscale), 1:length(Sscale));
x_ = Sscale(x(x+y==iS+round(length(Sscale)/2)));
y_ = Sscale(y(x+y==iS+round(length(Sscale)/2)));
v_ = Val(x+y==iS+round(length(Sscale)/2));
h = surfl(Sscale, Sscale, Val); hold on; %camproj perspective;
set(h,'FaceColor',sat(.5,col), 'EdgeColor','none'); camlight left; lighting phong; alpha(0.7);
if ischar(col);  plot3(x_, y_, v_,         col); hold on;
else             plot3(x_, y_, v_, 'Color',col); hold on;  end
xlabel(Slabel{1}); ylabel(Slabel{2}); %zlim([-50 50]);
% h = get(gca,'XLabel'); set(h,'FontSize',8, 'Position',get(h,'Position')+[0 .2 0]);
% h = get(gca,'YLabel'); set(h,'FontSize',8, 'Position',get(h,'Position')+[1 .2 0]);

function [dbIS] = plotDecisionVolume(S, D, minmax, myCol)
if nargin < 4;  myCol = [1 0 0 0.5; 0 1 0 0.5; 0 0 1 0.5];  end;
shiftMin = 0.01 * [1 0 0; 0 1 0; 0 0 1];
for iD = 3:-1:1
    switch iD
        case 1
            idx = D==1 | D==12 | D==13 | D==123;
        case 2
            idx = D==2 | D==12 | D==23 | D==123;
        case 3
            idx = D==3 | D==23 | D==13 | D==123;
    end
%     plot3(vector(S{1}(idx)), vector(S{2}(idx)), vector(S{3}(idx)), '.', 'Color', myCol(iD,:));
    db{iD}.vertices = [vector(S{1}(idx)), vector(S{2}(idx)), vector(S{3}(idx))];
    db{iD}.faces = convhull(vector(S{1}(idx)), vector(S{2}(idx)), vector(S{3}(idx)));
    trisurf(db{iD}.faces, db{iD}.vertices(:,1)+shiftMin(iD,1), db{iD}.vertices(:,2)+shiftMin(iD,2), db{iD}.vertices(:,3)+shiftMin(iD,3), 'FaceColor',myCol(iD,1:3),'FaceAlpha',myCol(iD,4),'EdgeColor','none'); hold on;
end
attractor.vertices = [[1;-1;-1], [-1;1;-1], [-1;-1;1]];
attractor.faces = [1 2 3; 1 2 3; 1 2 3];
trisurf(attractor.faces, attractor.vertices(:,1), attractor.vertices(:,2), attractor.vertices(:,3), 'FaceColor',[0 0 0],'FaceAlpha',0.1,'EdgeColor','none'); hold on;
for iD = 3:-1:1
    [~, dbIS{iD}] = SurfaceIntersection(db{iD}, attractor);
    dbIS{iD}.vertices2d = dbIS{iD}.vertices * [1/sqrt(2) -1/sqrt(2) 0; -1/sqrt(3) -1/sqrt(3) 1/sqrt(3); 0 0 0]';
    line([dbIS{iD}.vertices(dbIS{iD}.edges(:,1),1), dbIS{iD}.vertices(dbIS{iD}.edges(:,2),1)]',...
         [dbIS{iD}.vertices(dbIS{iD}.edges(:,1),2), dbIS{iD}.vertices(dbIS{iD}.edges(:,2),2)]',...
         [dbIS{iD}.vertices(dbIS{iD}.edges(:,1),3), dbIS{iD}.vertices(dbIS{iD}.edges(:,2),3)]', ...
         'Color',myCol(iD,1:3),'LineWidth',1.5); hold on;
end
a = minmax(1);  b = minmax(2);
line([a b; a a; a b; a a;   a a; a a; b b; b b;   a b; a a; a b; a a]', ...
     [a a; a b; b b; a b;   a a; b b; a a; b b;   a a; a b; b b; a b]', ...
     [b b; b b; b b; b b;   a b; a b; a b; a b;   a a; a a; a a; a a]', 'Color',.7*[1 1 1]);
axis square; camproj perspective; grid on; axis([minmax minmax minmax]); view([-25 15]); camlight left; lighting phong;
xlabel('r_1^{hat}'); ylabel('r_2^{hat}'); zlabel('r_3^{hat}'); set(gca,'XTick',-100:100,'YTick',-100:100,'ZTick',-100:100);
