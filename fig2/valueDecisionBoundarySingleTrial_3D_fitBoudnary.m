function valueDecisionBoundarySingleTrial_3D()
tic;
Smax = 6;      % Grid range of states space (now we assume: S = [(Rhat1+Rhat2)/2, (Rhat1-Rhat2)/2]); Rhat(t) = (varR*X(t)+varX)/(t*varR+varX) )
resS = 41;      % Grid resolution of state space
tmax = 2;       % Time limit
dt   = .05;       % Time step
c    = 1;       % Cost of evidene accumulation
g.meanR = [0 0 0]';
g.covR  = [5 0 0; 0 5 0; 0 0 5];
g.covX  = [2 0 0; 0 2 0; 0 0 2];
t = 0:dt:tmax;
Slabel = {'r_1^{hat}', 'r_2^{hat}', 'r_3^{hat}'};

%% Utililty function:
utilityFunc = @(X) X;
% utilityFunc = @(X) tanh(X);
% utilityFunc = @(X) sign(X).*abs(X).^0.5;

%% Covariance matrices:
Sscale = linspace(-Smax, Smax, resS);
[S{1},S{2},S{3}] = ndgrid(Sscale, Sscale, Sscale);
g.invCovR  = inv(g.covR);
g.invCovX  = inv(g.covX);
aSscale = abs(S{1}(:,1,1));
for iT = length(t):-1:1
    g.invCovRh(:,:,iT)   = g.invCovR + t(iT)*g.invCovX;
    g.covRh(:,:,iT)      = inv(g.invCovR);
    g.covRhTrans(:,:,iT) = covarianceOfTranslation(g.invCovRh(:,:,iT), g.invCovX, dt);
    for k = 3:-1:1
        diagCTrans = diag(g.covRhTrans(:,:,iT));
        g.iSTrans{k,iT} = find(aSscale < 3*sqrt(diagCTrans(k)));
    end
    g.PTrans{iT} = normal3({S{1}(g.iSTrans{1,iT},g.iSTrans{2,iT},g.iSTrans{3,iT}), ...
                            S{2}(g.iSTrans{1,iT},g.iSTrans{2,iT},g.iSTrans{3,iT}), ...
                            S{3}(g.iSTrans{1,iT},g.iSTrans{2,iT},g.iSTrans{3,iT})}, [0 0 0], g.covRhTrans(:,:,iT));
end

%% Transformation to the diffusion space:
for iT = length(t):-1:1
    X_(:,:,iT) = g.covX * (g.invCovRh(:,:,iT) * [S{1}(:) S{2}(:) S{3}(:)]' - g.invCovR * repmat(g.meanR, [1 numel(S{1})])) ;
    for iC = 3:-1:1;  X{iC,iT} = reshape(X_(iC,:,iT), size(S{iC}));  end
end
Xminmax = [min(X_(:)) max(X_(:))];

%% Reward rate, value, decision:
% iS0 = [findnearest(g{1}.meanR, Sscale) findnearest(g{2}.meanR, Sscale) findnearest(g{3}.meanR, Sscale)];
iS0 = [findnearest(g.meanR(1), Sscale) findnearest(g.meanR(2), Sscale) findnearest(g.meanR(3), Sscale)];
for iC = 3:-1:1;  Rh{iC} = utilityFunc(S{iC});  end             % Expected reward for option iC
[RhMax, Dd] = max_({Rh{1}, Rh{2}, Rh{3}});                      % Expected reward for decision
[V0, V, D, EVnext] = backwardInduction(c,g,Rh,t,dt,iS0);        % Value, decision, transition prob. etc.


%% - Show -
iS2 = findnearest(.5, Sscale, -1);
iS3 = 1;
iTmax = length(t);
rect = [-2 2 -2 2 -2.3 .5];
myCol = [1 0 0; 0 1 0; 0 0 1];

%% Backward induction demo:
% figure(4563); clf; colormap bone;
% %% t=0:
% iT = 1;
% subplotXY(5,4,2,1); [r1Max,r2Max,vMax] = plotSurf(Sscale, V(:,:,iS3,iT)          , iS2, [0 0 0], Slabel); axis(rect); title('V(0)');
% subplotXY(5,4,3,1); [r1Acc,r2Acc,vAcc] = plotSurf(Sscale, EVnext(:,:,iS3,iT)-c*dt, iS2, [1 0 0], Slabel); axis(rect); title('<V(\deltat)|R^{hat}(0)> - (\rho+c)\deltat');
% subplotXY(5,4,4,1); [r1Dec,r2Dec,vDec] = plotSurf(Sscale, RhMax(:,:,iS3)         , iS2, [0 0 1], Slabel); axis(rect); title('max(R_1^{hat},R_2^{hat}) - \rho t_{Null}');
% subplotXY(5,4,5,1); hold on;
%     plot((r1Max-r2Max)/2, vMax, 'k:', (r1Acc-r2Acc)/2, vAcc, 'r', (r1Dec-r2Dec)/2, vDec, 'b');
%     xlabel(['(' Slabel{1} '-' Slabel{2} ')/2']); xlim(rect(1:2)); %ylim(rect(5:6));
% subplotXY(5,4,1,1); imagesc(Sscale, Sscale, D(:,:,  1), [1 4]); axis square; axis xy; title('D(0)'); xlabel(Slabel{1}); ylabel(Slabel{2}); hold on; axis(rect(1:4));
%                     plot(r1Max, r2Max, 'r-');
% %% t=0 (superimposed & difference):
% subplotXY(5,4,3,2); plotSurf(Sscale, EVnext(:,:,iS3,iT)-c*dt                   , iS2, [1 0 0], Slabel); hold on;
%                     plotSurf(Sscale, RhMax(:,:,iS3)                            , iS2, [0 0 1], Slabel); axis(rect);
% subplotXY(5,4,4,2); plotSurf(Sscale, RhMax(:,:,iS3) - (EVnext(:,:,iS3,iT)-c*dt), iS2, [0 1 0], Slabel); xlim(rect(1:2)); ylim(rect(1:2));
% subplotXY(5,4,5,2); plotDecisionVolume(S, D(:,:,:,iT), rect(1:2));
% %% t=dt:
% subplotXY(5,4,1,2); imagesc(Sscale, Sscale, D(:,:,iS3,iT+1), [1 4]); axis square; axis xy; title('D(\deltat)'); xlabel(Slabel{1}); ylabel(Slabel{2}); hold on; axis(rect(1:4));
% subplotXY(5,4,2,2); plotSurf(Sscale, V(:,:,iS3,iT+1), iS2, [0 0 0], Slabel); axis(rect); title('V(\deltat)');
% %% t=T-dt:
% % subplotXY(5,4,3,2); surfl(Sscale(iStrans{1}{2}), Sscale(iStrans{1}{2}), Ptrans{1}); title('P(R^{hat}(\deltat)|R^{hat}(0))'); shading interp; hold on; axis([rect 0 Inf]); axis off;
% subplotXY(5,4,1,3); imagesc(Sscale, Sscale, D(:,:,iS3,iTmax-1), [1 4]); axis square; axis xy; title('D(T-\deltat)'); hold on; rectangle('Position',[rect(1) rect(3) rect(2)-rect(1) rect(4)-rect(3)]); axis(rect);
% subplotXY(5,4,2,3); [r1Max,r2Max,vMax] = plotSurf(Sscale, V(:,:,iS3,iTmax-1)           , iS2, [0 0 0], Slabel); axis(rect); title('V(T-\deltat)')
% subplotXY(5,4,3,3); [r1Acc,r2Acc,vAcc] = plotSurf(Sscale, EVnext(:,:,iS3,iTmax-1)-c*dt , iS2, [1 0 0], Slabel); axis(rect); title('<V(T)|R^{hat}(T-\deltat)> - (\rho+c) \deltat');
% subplotXY(5,4,4,3); [r1Dec,r2Dec,vDec] = plotSurf(Sscale, RhMax(:,:,iS3)               , iS2, [0 0 1], Slabel); axis(rect); title('max(R_1^{Hat},R_2^{Hat}) - \rho t_{Null}');
% subplotXY(5,4,5,3); hold on;
%     plot((r1Max-r2Max)/2, vMax, 'k:', (r1Acc-r2Acc)/2, vAcc, 'r', (r1Dec-r2Dec)/2, vDec, 'b');
%     xlabel(['(' Slabel{1} '-' Slabel{2} ')/2']); xlim(rect(1:2)); %ylim(rect(5:6));
% subplotXY(5,4,5,4); plotDecisionVolume(S, D(:,:,:,iTmax-1), rect(1:2));
% %% t=T:
% subplotXY(5,4,1,4); imagesc(Sscale, Sscale, D(:,:,iS3,iTmax), [1 4]); axis square; axis xy; title('D(T)'); hold on; axis(rect(1:4));
% subplotXY(5,4,2,4); plotSurf(Sscale, V(:,:,iS3,iTmax), iS2, [0 0 0], Slabel); title('V(T) = max(R_1^{hat},R_2^{hat}) - \rho t_{Null}'); axis(rect);
% % subplotXY(5,4,3,4); surfl(Sscale(iStrans{iTmax-1}{2}), Sscale(iStrans{iTmax-1}{1}), Ptrans{iTmax-1}); title('P(R^{hat}(T)|R^{hat}(T-\deltat))'); shading interp; hold on; axis([rect 0 Inf]); axis off;


%% Decision boundaries superimposed:
figure(4564); clf; set(gcf,'Color',[1 1 1]);
iT = [1 3 11 41];
rectX = [-1.5 1.5];
k = 1.5;
%k = 4.5;
for iiT = 1:length(iT)
    iiT
    db = decisionBoundary(S, D(:,:,:,iT(iiT)));                     % Decision boundary in belief space
    [dbISTri, triPlain] = dbIntersectionTri(db, rect(1:2));
    dbX = decisionBoundary({X{:,iT(iiT)}}, D(:,:,:,iT(iiT)));       % Decision boundary in diffusion space
    [dbISTriX,  triPlainX]  = dbIntersectionTri(dbX, rectX(1:2));
    [dbISCubeX, cubePlainX] = dbIntersectionCube(dbX, rectX(1:2));
    toc;
    
%     u0 = 1;  uOpt = fminsearch(@(u) errorAttVsOptimalDB(u, k, S, dbISCubeX, rectX), u0);
%     us = 1:.25:2.5;  for iU = length(us):-1:1;  J(iU) = errorAttVsOptimalDB(us(iU), k, S, dbISCubeX, rectX);  end:  uOpt = us(find(J==min(J),1));
    uOpt = (mean(sum(halfRectif(dbISCubeX{1}.vertices).^k, 2)) + mean(sum(halfRectif(dbISCubeX{2}.vertices).^k, 2)) + mean(sum(halfRectif(dbISCubeX{3}.vertices).^k, 2))) / 3;
    [~, attX, attISCubeX] = errorAttVsOptimalDB(uOpt, k, S, dbISCubeX, rectX);
    
    toc;
    %% Boundaries supreimposed:
    subplotXY(5,4,1,1);
        plotDecisionVolume(db, rect(1:2), [myCol, [0.3; 0.05; 0.05]]); hold on;
        plotTriPlain(dbISTri, triPlain, rect(1:2), [myCol, [0.3; 0.05; 0.05]]); hold on;
        xlabel('r_1^{hat}'); ylabel('r_2^{hat}'); zlabel('r_3^{hat}'); title('Belief space');
    subplotXY(5,4,1,3);
        plotDecisionVolume(dbX, rectX(1:2), [myCol, [0.3; 0.05; 0.05]]); hold on;
        plotTriPlain(dbISTriX, triPlainX, rectX(1:2), [myCol, [0.3; 0.05; 0.05]]); hold on;
        xlabel('x_1'); ylabel('x_2'); zlabel('x_3'); title('Diffusion space');
    %% Boundaries supreimposed (2D):
    colR = iiT/length(iT);
    subplotXY(5,4,1,2);
        plotISTri2d(dbISTri, colR*myCol+(1-colR)*ones(3));
        plotTri2d(triPlain)
    subplotXY(5,4,1,4);
        plotISTri2d(dbISTriX, colR*myCol+(1-colR)*ones(3));
        plotTri2d(triPlainX)
    %% Boundaries & triangle:
    subplotXY(5,2*length(iT),2,iiT);
        plotDecisionVolume(db, rect(1:2), [myCol, [0.5; 0.5; 0.5]]); hold on;
        plotTriPlain(dbISTri, triPlain, rect(1:2), [myCol, [0.5; 0.5; 0.5]]); hold on;
        title(['Time=' num2str(t(iT(iiT)))]);
    subplotXY(5,2*length(iT),2,iiT+length(iT));
        plotDecisionVolume(dbX, rectX(1:2), [myCol, [0.5; 0.5; 0.5]]); hold on;
        plotTriPlain(dbISTriX, triPlainX, rectX(1:2), [myCol, [0.5; 0.5; 0.5]]); hold on;
        title(['Time=' num2str(t(iT(iiT)))]);
    %% Boundaries & cube, cube & attractor:
    subplotXY(5,2*length(iT),3,iiT+length(iT));
        plotDecisionVolume(dbX, rectX(1:2), [myCol, [0.5; 0.1; 0.1]]); hold on;
        plotCubePlain(dbISCubeX, cubePlainX, rectX(1:2), [myCol, [0.5; 0.5; 0.5]]); hold on;
    subplotXY(5,2*length(iT),4,iiT+length(iT));
        plotCubePlain(dbISCubeX, cubePlainX, rectX(1:2), [myCol, [0.5; 0.5; 0.5]]); hold on;
        plotAtt(attISCubeX, attX, rectX(1:2), [myCol, [0.5; 0.5; 0.5]]); hold on;
    %% 2D plots:
    subplotXY(5,2*length(iT),5,iiT);
        plotISTri2d(dbISTri, myCol);
        plotTri2d(triPlain)
    subplotXY(5,2*length(iT),5,iiT+length(iT));
        plotISTri2d(dbISTriX, myCol);
        plotISCube2d(dbISCubeX, myCol);
        plotISAtt2d(attISCubeX, myCol);
        plotTri2d(triPlainX);
        title(['u = ' num2str(uOpt,3)]);
end
toc;

%% Functions for dynamic programming:
function [V0, V, D, EVnext] = backwardInduction(c,g,Rh,t,dt,iS0)
[V(:,:,:,length(t)), D(:,:,:,length(t))] = max_({Rh{1}, Rh{2}, Rh{3}});                     % Max V~ at time tmax
for iT = length(t)-1:-1:1
    EVnext(:,:,:,iT) = E(V(:,:,:,iT+1), g.PTrans{iT});                                       % <V~(t+1)|S(t)> for waiting
    [V(:,:,:,iT), D(:,:,:,iT)] = max_({Rh{1}, Rh{2}, Rh{3}, EVnext(:,:,:,iT)-c*dt});        % [Average-adjusted value (V~), decision] at time t
%     fprintf('%d/%d\t',iT,length(t)-1); toc;
end
V0 = mean(vector(V(iS0(1),iS0(2),1)));
D(D==0) = 4;
fprintf('V0 = %d\t', V0); toc;

function [EV] = E(V,PTrans)
%% Expected value:
% mgn = ceil(size(g.PTrans)/2);
% V = extrap(V,mgn,[5 5 5]);
EV = convn(V,PTrans,'same') ./ convn(ones(size(V)),PTrans,'same');
% EV = EV(mgn(1)+1:end-mgn(1), mgn(2)+1:end-mgn(2), mgn(3)+1:end-mgn(3));

function CTrans = covarianceOfTranslation(invCRh, invCX, dt)
CRhNext = inv(invCRh + dt*invCX);
% CTrans = dt * CRhNext * invCX * CRhNext;
CTrans = dt * CRhNext * invCX * CRhNext + dt^2 * CRhNext * invCX * inv(invCRh) * invCX * CRhNext;

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

%% Functions for computing surfaces and intersections:
function [db] = decisionBoundary(S, D)
for iD = 3:-1:1
    switch iD
        case 1
            idx = D==1 | D==12 | D==13 | D==123;
        case 2
            idx = D==2 | D==12 | D==23 | D==123;
        case 3
            idx = D==3 | D==23 | D==13 | D==123;
    end
    db{iD}.vertices = [vector(S{1}(idx)), vector(S{2}(idx)), vector(S{3}(idx))];
    db{iD}.faces = convhull(vector(S{1}(idx)), vector(S{2}(idx)), vector(S{3}(idx)));
end

function [att] = makeAttractor(S, u, k)
idx = (halfRectif(S{1}).^k + halfRectif(S{2}).^k + halfRectif(S{3}).^k) < u;
att.vertices = [vector(S{1}(idx)), vector(S{2}(idx)), vector(S{3}(idx))];
att.faces = convhull(vector(S{1}(idx)), vector(S{2}(idx)), vector(S{3}(idx)));
att.idx = idx;

function y = halfRectif(x)
y = x;
y(y<0) = 0;

function [dbIS, triPlain] = dbIntersectionTri(db, plotRng)
a = plotRng(1);  b = plotRng(2);
triPlain.vertices = [[b a a];[a b a];[a a b]];
triPlain.faces = [1 2 3];
projMat = [1/sqrt(2) -1/sqrt(2) 0; -1/sqrt(6) -1/sqrt(6) 2/sqrt(6); 0 0 0]';
for iD = 3:-1:1
    [~, dbIS{iD}] = SurfaceIntersection(db{iD}, triPlain);
    if ~isempty(dbIS{iD}.vertices)
        dbIS{iD}.vertices2d = dbIS{iD}.vertices * projMat;
    else
        dbIS{iD}.vertices2d = [];
    end
end
triPlain.vertices2d = triPlain.vertices * projMat;

function [dbIS, cubePlain] = dbIntersectionCube(db, plotRng)
a = plotRng(1);  b = plotRng(2);
th = 1;
cubePlain{1}.vertices = [[th a a];[th a th];[th th a];[th th th]];
cubePlain{1}.faces = [1 2 3; 2 3 4];
cubePlain{2}.vertices = [[a th a];[a th th];[th th a];[th th th]];
cubePlain{2}.faces = [1 2 3; 2 3 4];
cubePlain{3}.vertices = [[a a th];[a th th];[th a th];[th th th]];
cubePlain{3}.faces = [1 2 3; 2 3 4];

projMat = [1/sqrt(2) -1/sqrt(2) 0; -1/sqrt(6) -1/sqrt(6) 2/sqrt(6); 0 0 0]';
for iD = 3:-1:1
    [~, dbIS{iD}] = SurfaceIntersection(db{iD}, cubePlain{iD});
    if ~isempty(dbIS{iD}.vertices)
        dbIS{iD}.vertices2d = dbIS{iD}.vertices * projMat;
    else
        dbIS{iD}.vertices2d = [];
    end
    cubePlain{iD}.vertices2d = cubePlain{iD}.vertices * projMat;
end

function [attIS, cubePlain] = attIntersectionCube(att, plotRng)
a = plotRng(1);  b = plotRng(2);
th = 1;
cubePlain{1}.vertices = [[th a a];[th a th];[th th a];[th th th]];
cubePlain{1}.faces = [1 2 3; 2 3 4];
cubePlain{2}.vertices = [[a th a];[a th th];[th th a];[th th th]];
cubePlain{2}.faces = [1 2 3; 2 3 4];
cubePlain{3}.vertices = [[a a th];[a th th];[th a th];[th th th]];
cubePlain{3}.faces = [1 2 3; 2 3 4];

projMat = [1/sqrt(2) -1/sqrt(2) 0; -1/sqrt(6) -1/sqrt(6) 2/sqrt(6); 0 0 0]';
for iD = 3:-1:1
    [~, attIS{iD}] = SurfaceIntersection(att, cubePlain{iD});
    if ~isempty(attIS{iD}.vertices)
        attIS{iD}.vertices2d = attIS{iD}.vertices * projMat;
    else
        attIS{iD}.vertices2d = [];
    end
    cubePlain{iD}.vertices2d = cubePlain{iD}.vertices * projMat;
end

function [J, attX, attISCubeX] = errorAttVsOptimalDB(u, k, S, dbISCubeX, rectX)
attX = makeAttractor(S, u, k);                                  % Attractor surface in diffusion space
[attISCubeX, ~] = attIntersectionCube(attX, rectX(1:2));
for iD = 3:-1:1
    if ~isempty(attISCubeX{iD}.vertices2d)
        Dnn(iD) = mean(minN(LnDist(attISCubeX{iD}.vertices2d',dbISCubeX{iD}.vertices2d',2),1).^2);
    else
        Dnn(iD) = Inf;
    end
end
J = mean(Dnn);

%% Functions for making figures:
function plotDecisionVolume(db, plotRng, myCol)
if nargin < 3;  myCol = [1 0 0 0.5; 0 1 0 0.5; 0 0 1 0.5];  end;
shiftMin = 0.02 * diff(plotRng) * [1 0 0; 0 1 0; 0 0 1];
for iD = 3:-1:1
%     plot3(vector(S{1}(idx)), vector(S{2}(idx)), vector(S{3}(idx)), '.', 'Color', myCol(iD,:));
    trisurf(db{iD}.faces, db{iD}.vertices(:,1)+shiftMin(iD,1), db{iD}.vertices(:,2)+shiftMin(iD,2), db{iD}.vertices(:,3)+shiftMin(iD,3), 'FaceColor',myCol(iD,1:3),'FaceAlpha',myCol(iD,4),'EdgeColor','none'); hold on;
end
a = plotRng(1)+0.01;  b = plotRng(2)-0.01;
line([a b; a a; a b; b b;   a a; a a; b b; b b;   a b; a a; a b; b b]', ...
     [a a; a b; b b; a b;   a a; b b; a a; b b;   a a; a b; b b; a b]', ...
     [b b; b b; b b; b b;   a b; a b; a b; a b;   a a; a a; a a; a a]', 'Color',.7*[1 1 1]);
axis square; camproj perspective; grid on; axis([plotRng plotRng plotRng]); view([-25 15]); camlight left; lighting phong;
set(gca,'XTick',-100:100,'YTick',-100:100,'ZTick',-100:100);

function plotTriPlain(dbIS, triPlain, plotRng, myCol)
if nargin < 4;  myCol = [1 0 0 0.5; 0 1 0 0.5; 0 0 1 0.5];  end;
shiftMin = 0.02 * diff(plotRng) * [1 0 0; 0 1 0; 0 0 1];
trisurf(triPlain.faces, triPlain.vertices(:,1), triPlain.vertices(:,2), triPlain.vertices(:,3), 'FaceColor',[0 0 0],'FaceAlpha',0.1,'EdgeColor','none'); hold on;
for iD = 3:-1:1
    if ~isempty(dbIS{iD}.vertices)
        line([dbIS{iD}.vertices(dbIS{iD}.edges(:,1),1), dbIS{iD}.vertices(dbIS{iD}.edges(:,2),1)]'+shiftMin(iD,1),...
             [dbIS{iD}.vertices(dbIS{iD}.edges(:,1),2), dbIS{iD}.vertices(dbIS{iD}.edges(:,2),2)]'+shiftMin(iD,2),...
             [dbIS{iD}.vertices(dbIS{iD}.edges(:,1),3), dbIS{iD}.vertices(dbIS{iD}.edges(:,2),3)]'+shiftMin(iD,3), ...
             'Color',myCol(iD,1:3),'LineWidth',1.5); hold on;
    end
end
axis square; camproj perspective; grid on; axis([plotRng plotRng plotRng]); view([-25 15]); camlight left; lighting phong;
set(gca,'XTick',-100:100,'YTick',-100:100,'ZTick',-100:100);

function plotCubePlain(dbIS, cubePlain, plotRng, myCol)
if nargin < 4;  myCol = [1 0 0 0.5; 0 1 0 0.5; 0 0 1 0.5];  end;
shiftMin = 0.02 * diff(plotRng) * [1 0 0; 0 1 0; 0 0 1];
for iD = 3:-1:1
    trisurf(cubePlain{iD}.faces, cubePlain{iD}.vertices(:,1), cubePlain{iD}.vertices(:,2), cubePlain{iD}.vertices(:,3), 'FaceColor',myCol(iD,1:3),'FaceAlpha',0.3,'EdgeColor','none'); hold on;
    if ~isempty(dbIS{iD}.vertices)
        line([dbIS{iD}.vertices(dbIS{iD}.edges(:,1),1), dbIS{iD}.vertices(dbIS{iD}.edges(:,2),1)]'+shiftMin(iD,1),...
             [dbIS{iD}.vertices(dbIS{iD}.edges(:,1),2), dbIS{iD}.vertices(dbIS{iD}.edges(:,2),2)]'+shiftMin(iD,2),...
             [dbIS{iD}.vertices(dbIS{iD}.edges(:,1),3), dbIS{iD}.vertices(dbIS{iD}.edges(:,2),3)]'+shiftMin(iD,3), ...
             'LineStyle',':','Color',myCol(iD,1:3),'LineWidth',1.5); hold on;
    end
end
axis square; camproj perspective; grid on; axis([plotRng plotRng plotRng]); view([-25 15]); camlight left; lighting phong;
set(gca,'XTick',-100:100,'YTick',-100:100,'ZTick',-100:100);

function plotAtt(attIS, att, plotRng, myCol)
if nargin < 4;  myCol = [1 0 0 0.5; 0 1 0 0.5; 0 0 1 0.5];  end;
shiftMin = 0.02 * diff(plotRng) * [1 0 0; 0 1 0; 0 0 1];
for iD = 3:-1:1
    trisurf(att.faces, att.vertices(:,1), att.vertices(:,2), att.vertices(:,3), 'FaceColor',.5*[1 1 1],'FaceAlpha',0.3,'EdgeColor','none'); hold on;
    if ~isempty(attIS{iD}.vertices)
        line([attIS{iD}.vertices(attIS{iD}.edges(:,1),1), attIS{iD}.vertices(attIS{iD}.edges(:,2),1)]'+shiftMin(iD,1),...
             [attIS{iD}.vertices(attIS{iD}.edges(:,1),2), attIS{iD}.vertices(attIS{iD}.edges(:,2),2)]'+shiftMin(iD,2),...
             [attIS{iD}.vertices(attIS{iD}.edges(:,1),3), attIS{iD}.vertices(attIS{iD}.edges(:,2),3)]'+shiftMin(iD,3), ...
             'LineStyle',':','Color',myCol(iD,1:3),'LineWidth',1.5); hold on;
    end
end
axis square; camproj perspective; grid on; axis([plotRng plotRng plotRng]); view([-25 15]); camlight left; lighting phong;
set(gca,'XTick',-100:100,'YTick',-100:100,'ZTick',-100:100);

function plotISTri2d(dbISTri, myCol)
for iD = 1:3
    if ~isempty(dbISTri{iD}.vertices2d)
        line([dbISTri{iD}.vertices2d(dbISTri{iD}.edges(:,1),1), dbISTri{iD}.vertices2d(dbISTri{iD}.edges(:,2),1)]',...
             [dbISTri{iD}.vertices2d(dbISTri{iD}.edges(:,1),2), dbISTri{iD}.vertices2d(dbISTri{iD}.edges(:,2),2)]',...
             [dbISTri{iD}.vertices2d(dbISTri{iD}.edges(:,1),3), dbISTri{iD}.vertices2d(dbISTri{iD}.edges(:,2),3)]', ...
             'Color',myCol(iD,1:3),'LineWidth',1.5); hold on;
    end
end
axis equal; view([0 90]); axis off;

function plotISCube2d(dbISCube, myCol)
for iD = 1:3
    if ~isempty(dbISCube{iD}.vertices2d)
        line([dbISCube{iD}.vertices2d(dbISCube{iD}.edges(:,1),1), dbISCube{iD}.vertices2d(dbISCube{iD}.edges(:,2),1)]',...
             [dbISCube{iD}.vertices2d(dbISCube{iD}.edges(:,1),2), dbISCube{iD}.vertices2d(dbISCube{iD}.edges(:,2),2)]',...
             [dbISCube{iD}.vertices2d(dbISCube{iD}.edges(:,1),3), dbISCube{iD}.vertices2d(dbISCube{iD}.edges(:,2),3)]', ...
             'LineStyle',':','Color',myCol(iD,1:3),'LineWidth',1); hold on;
    end
end
axis equal; view([0 90]); axis off;

function plotISAtt2d(attISCube, myCol)
for iD = 1:3
    if ~isempty(attISCube{iD}.vertices2d)
        line([attISCube{iD}.vertices2d(attISCube{iD}.edges(:,1),1), attISCube{iD}.vertices2d(attISCube{iD}.edges(:,2),1)]',...
             [attISCube{iD}.vertices2d(attISCube{iD}.edges(:,1),2), attISCube{iD}.vertices2d(attISCube{iD}.edges(:,2),2)]',...
             [attISCube{iD}.vertices2d(attISCube{iD}.edges(:,1),3), attISCube{iD}.vertices2d(attISCube{iD}.edges(:,2),3)]', ...
             'LineStyle',':','Color',myCol(iD,1:3),'LineWidth',2); hold on;
    end
end
axis equal; view([0 90]); axis off;

function plotTri2d(triPlain)
line(triPlain.vertices2d([1:3 1],1), triPlain.vertices2d([1:3 1],2), 'Color',0.5*[1 1 1]);
