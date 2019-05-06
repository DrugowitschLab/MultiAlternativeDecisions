function r = commonDynamics(p, Z)
% outputs common randomness driving dynamics
% X0 - initial point [trials x N x 1]
% Z - true latent state [trials x N]
% Zmax - maximum latent state [trials]
% iZmax - index of maximum latent state [trials]
% dX - momentary evidence [trials x N x t]
% X - linearly accumulated evidence [trials x N x t]
% Zh - latent state estimate [trials x N x t]
% Xnorm - normalizes X [trials x N x t]

% initial integrator state, (slightly noisy to avoid too much regularity)
r.X0 = 0.00001 * randn(p.sim.nTrial, p.task.N, 1);
% latent state, maximum value, and index thereof
if nargin < 2
    r.Z = mvnrnd(p.task.meanZ, p.task.covZ, p.sim.nTrial);
else
    r.Z = Z;
end
[r.Zmax, r.iZmax] = max(r.Z, [], 2);
[r.Zmin, r.iZmin] = min(r.Z, [], 2);
r.iZmid = zeros(size(r.iZmin));
r.iZmid(r.iZmax~=1 & r.iZmin~=1) = 1;
r.iZmid(r.iZmax~=2 & r.iZmin~=2) = 2;
r.iZmid(r.iZmax~=3 & r.iZmin~=3) = 3;

% momentary evidece
for iT = length(p.sim.t):-1:2
    r.dX(:,:,iT) = mvnrnd(p.sim.dt*r.Z, p.sim.dt*p.task.covX, p.sim.nTrial);
end
r.dX(:,:,1) = r.X0;


% linearly accumulated evidence
% r.X(:,:,1) = cumsum();
% r.Zh(:,:,1) = repmat(p.task.meanZ, [p.sim.nTrial 1]);
% for iT = 2:length(p.sim.t);
%     r.X(:,:,iT) = r.X(:,:,iT-1) + r.dX(:,:,iT);
%     % hidden state estimate
%     icovZh = p.task.icovZ + p.sim.t(iT) * p.task.icovX;
%     r.Zh(:,:,iT) = (repmat(p.task.meanZ, [p.sim.nTrial 1]) * p.task.icovZ ...
%                     + r.X(:,:,iT) * p.task.icovX) / icovZh;
% end
% constrain = @(X) X - repmat(mean(X, 2), [1 p.task.N 1]);
% r.Xnorm = constrain(r.X);
