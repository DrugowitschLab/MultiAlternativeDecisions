function rModel = simulateDiffusion3(r, p, model)
rModel.X    = zeros(p.nTrial,p.N,length(p.t));
switch upper(model)
    case 'RM'
        rModel.X = r.base.X + p.rm.X0mean;
        rModel = performance(rModel, r, p);
        
    case 'URM'
        urgency   = @(t) p.ur.b*t + p.ur.u0;
        rModel.X = r.base.X + shiftdim(repmat(urgency(p.t(:)),[1 p.nTrial p.N]),1);
        rModel = performance(rModel, r, p);

    case 'NRM'
        rModel.X = r.normalized.X + p.ms.u;
        rModel = performance(rModel, r, p);

    case 'NRMDIV'
        rModel.X = r.normalizedDiv.X;
        rModel = performance(rModel, r, p);

    case 'LUCRM'
        urgency   = @(t) p.uc.b*t + p.uc.u0;
        rModel.X = r.normalized.X + shiftdim(repmat(urgency(p.t(:)),[1 p.nTrial p.N]),1);
        rModel = performance(rModel, r, p);
        
    case 'UCRM'
        f = @(X) halfRectify(X).^p.uc.a;
        urgency     = @(t) p.uc.b*t + p.uc.u0;
        errorUCR    = @(X,t,a) (urgency(t) - repmat(mean(X, 2), [1 size(X,2)]));
%         errorUCR    = @(X,t,a) (urgency(t) - repmat(mean(f(X), 2), [1 size(X,2)]));
        addNoise = @(x) x + p.noiseGain*sqrt(p.dt/p.uc.nIterConstrain*abs(x).*p.fano) .* randn(size(x));
        rModel.u = urgency(p.t);
        rModel.X(:,:,1) = repmat(urgency(0)*(ones(1,p.N)/3).^(1/p.uc.a), [p.nTrial 1]) + r.X0;
        for iT = 2:length(p.t)
            rModel.X(:,:,iT) = f(rModel.X(:,:,iT-1) + r.dX(:,:,iT));
            %% Constrain state on a surface:
            for iIt = 1:p.uc.nIterConstrain
                err_ = errorUCR(rModel.X(:,:,iT), p.t(iT), p.uc.a);
                
%                 rModel.X(:,:,iT) = rModel.X(:,:,iT) + 0.4*err_.*repmat(mean(p.uc.a * halfRectify(rModel.X(:,:,iT)).^(p.uc.a-1),2),[1 p.N]);
                rModel.X(:,:,iT) = rModel.X(:,:,iT) + 0.4*err_;
                
                rModel.X(:,:,iT) = f(rModel.X(:,:,iT));
                
                %% Add noise:
                rModel.X(:,:,iT) = addNoise(rModel.X(:,:,iT));
            end
            %% Constraining error:
            rModel.err(:,iT) = sqrt(mean(err_.^2,2));
        end
        rModel = performance(rModel, r, p);
        
    case 'RANDOM'
        rModel.Reward = mean(p.meanZ);
        rModel.RT     = 0;
        rModel.RR     = (rModel.Reward - p.c(1)*rModel.RT) / (rModel.RT + p.tNull);
        rModel.correct= 1/p.N;
        rModel.CR     = 1/p.N/p.tNull;
        
    case 'ORACLE'
        rModel.Reward = mean(max(r.Z,[],2));
        rModel.RT     = 0;
        rModel.RR     = (rModel.Reward - p.c(1)*rModel.RT) / (rModel.RT + p.tNull);
        rModel.correct= 1;
        rModel.CR     = 1/p.tNull;
    
    otherwise
        display('ERROR: A wrong model-type parameter for function ''simulateDiffusion3''.');
end

function Y = halfRectify(X)
Y = max(0, X);

function rModel = performance(rModel, r, p)
rModel.RT    = max(p.t) * ones(p.nTrial,1);
rModel.iTEnd = ones(p.nTrial,1) * length(p.t);
rModel.D     = zeros(p.nTrial,1);
rModel.cost  = zeros(p.nTrial,1);
for iT = 1:length(p.t)
    [Xmax, iXmax] = max(rModel.X(:,:,iT),[],2);
    iTrialEnd = (Xmax >= p.threshold) & ~rModel.D;
    rModel.RT(iTrialEnd) = p.t(iT);
    rModel.cost(iTrialEnd) = p.cTotal(iT);
    rModel.iTEnd(iTrialEnd) = iT;
    rModel.D(iTrialEnd) = iXmax(iTrialEnd);
end
iTrialEnd = ~rModel.D;
rModel.D(iTrialEnd) = iXmax(iTrialEnd);

for iT = length(p.t):-1:1
    [~, iXmax] = max(rModel.X(:,:,iT),[],2);
    rModel.DFixedTime(:,iT)       = iXmax;
    rModel.RewardFixedTime(:,iT)  = varSelected(r.Z, rModel.DFixedTime(:,iT));
    rModel.RRFixedTime(:,iT)      = mean(rModel.RewardFixedTime(:,iT) - rModel.cost) ./ mean(p.t(iT) + p.tNull);
    rModel.correctFixedTime(:,iT) = (rModel.DFixedTime(:,iT) == r.iZmax);
end
rModel.Reward = varSelected(r.Z, rModel.D);
rModel.RR  = mean(rModel.Reward - rModel.cost) ./ mean(rModel.RT + p.tNull);
rModel.correct = (rModel.D == r.iZmax);
rModel.CR  = mean(rModel.correct - rModel.cost) ./ mean(rModel.RT + p.tNull);

function y = varSelected(x, d)
% for iT = size(x,3):-1:1
%     y(:,iT) = (d==1).*x(:,1,iT) + (d==2).*x(:,2,iT) + (d==3).*x(:,3,iT);
% end
nT = size(x,3);
y = repmat((d==1),[1 nT]).*squeeze(x(:,1,:));
for iD = 2:size(x,2)
    y = y + repmat((d==iD),[1 nT]).*squeeze(x(:,iD,:));
end

