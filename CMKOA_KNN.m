clear
close all
addpath('.\datasets');
addpath('.\funs');


%% ==================== Dataset ==========================
dataname = 'MSRC';lambda=50; r=5; p=0.1; k = 10;
% dataname = 'ORL'; lambda=100; r=7; p=0.1; k = 10;

load([dataname '.mat']);

gt=double(gt);
nV = size(X, 2);
[dv, nN] = size(X{1}'); 
nC = length(unique(gt)); 


Iter_max = 200;
max_mu = 12e12; 
rho = 1.1;
for k = k
% ------- Input Distance --------
D = cell(1, nV);
for v = 1:nV
    X{v} = data_process(X{v}, "max-min");
    D{v} = mydistance(X{v}, X{v}, "knn-L2", k);
end

for p = p
for r = r
for lambda = lambda

% initialize Y J Q H
Y = cell(1, nV);
J = cell(1, nV);
Q = cell(1, nV);
for i = 1:nN
    Y{1}(i,mod(i, nC)+1) = 1;
end

for v = 1:nV
    Y{v} = Y{1};
    J{v} = zeros(nN, nC);
    Q{v} = zeros(nN, nC);
end
alpha = ones(1,nV)./nV;
mu = 0.0001; 

% ---------- Iterative Update -----------
obj = [1];
obj_Y = [];
flag = 1;
iter = 1;
while flag == 1  
    % Solving Y
    for v = 1:nV
        H =  J{v} - Q{v}/mu;
        Y{v} = sol_discY(D{v}, mu / alpha(v)^r, H, Y{v});
    end
    % Solving J --Schatten p_norm
    [J, Q, mu] = sol_TSP(lambda, mu, p, Y, Q, rho, max_mu);
    
    % Solving alpha 
    for v = 1:nV
        h(v) = trace(Y{v}'*D{v}*Y{v});
    end
    alpha = auto_weight(h, r);

    oo = 0;
    for v = 1:nV
        oo = oo + norm(Y{v} - J{v},'fro');
    end

    obj_Y = [obj_Y oo];
    % ShouLian
    if  obj_Y(iter) < 1e-8 || iter == Iter_max
        flag = 0;
    end
    iter = iter + 1;
end
%% ================== Perfermance Calculate=======================
Y_sum = zeros(nN, nC);

for v=1:nV
    Y_sum = Y_sum + alpha(v)^r*Y{v};                                  
end
[~, label] = max(Y_sum');
result = ClusteringMeasure(gt,label)
end
end
end
end