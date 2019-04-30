function p = baseParameters(p)
%% adds the default parameters to p (if given)
%
% If p is provided, then all the structures in p that depend on other
% settings are updated (e.g. cost vector that depends on cpers).

if nargin < 1
    p = struct();
    
    % task settings
    p.task.N = 3;
    p.task.priZ = 1;
    p.task.sigZ = 1;
    p.task.sigX = 1;
    p.task.cpers = 0;
    p.task.tNull = 0.5;
    p.task.threshold = 1;

    % model parameters
    % normalization network
    p.model.normTau = 0.05; % network time constant
    p.model.normBase = 0.4; % normalization base rate multiplier
    p.model.normGain = 1;   % normalization gain
    % integrator noise
    p.model.intNoise = 0;   % integrator noise
    p.model.intFano = 0;    % integrator fano factor
    % urgency signal
    p.model.u0 = 0.5;       % urgency offset
    p.model.b = 0.5;        % urgency slope
    % non-linearity
    p.model.a = 1.5;        % power of non-linearity
    % constraint (normalization)
    p.model.nIterConstraint = 5;  % constraint iterations
    p.model.iterAlpha = 0.4;      % constraint iterations learning rate

    % network simulation settings
    p.sim.nTrial = 1000000;
    p.sim.dt = 0.005;
    p.sim.maxt = 10;

    % dynamic programming settings (N-dependent are further below)
    p.dp.maxt = 3;
    p.dp.diffSDs = 4;        % extent in SDs of transition kernel

    % parameter optimization settings
    p.optim.nRestarts = 10;
    p.optim.opt.Display = 'iter';
    p.optim.opt.UncertaintyHandling = 1;
    
end

% complete dependent structures
p.sim.t = 0:p.sim.dt:p.sim.maxt;

p.task.c = p.task.cpers * p.sim.t;
p.task.accumc = cumsum(p.task.c) * p.sim.dt;
p.task.meanZ = p.task.priZ * ones(1, p.task.N);
p.task.covZ = p.task.sigZ^2 * eye(p.task.N);
p.task.covX = p.task.sigX^2 * eye(p.task.N);
p.task.icovZ = inv(p.task.covZ);
p.task.icovX = inv(p.task.covX);

p.model.normSigH = p.model.normBase * p.task.priZ;
p.model.normK = 1 + p.model.normGain * (p.model.normSigH + p.task.N * p.task.priZ);

% N-dependent dynamic programming parameters
% N     other   2       3      4
Rmax  = [NaN    4       4      4    ];
dt    = [NaN    0.001   0.005  0.005];
dtL   = [NaN    0.005   0.01   0.01 ];
resR  = [NaN 1001     201    105    ];
resRL = [NaN  401      51     33    ];
if p.task.N > 4
    iN = 1;
else
    iN = p.task.N;
end
p.dp.Rmax = Rmax(iN);                % maximum discretized reward
p.dp.dt = dt(iN);                    % time step for fine grid
p.dp.dtL = dtL(iN);                  % time step for coarse grid
p.dp.resR = resR(iN);                % fine R grid resolution
p.dp.resRL = resRL(iN);              % coarse R grid resolution
p.dp.t = 0:p.dp.dt:p.dp.maxt;
p.dp.tL = 0:p.dp.dtL:p.dp.maxt;

% test overrides
%p.sim.nTrial = 100000;
%p.optim.nRestarts = 2;
