function [Y] = sol_discY(D, mu, H, Y_pre)
% Solving the following model:
%    min_Y tr(Y'DY) + mu/2 * ||Y - H||_F^2
%    s.t. Y \in Ind, D_ii = 0
%

[nN, ~] = size(H);
Y = Y_pre;
for i = 1:nN 
    M = (2*(D(i,:)*Y) - mu * H(i,:))';%*lambda
    [~, m] = min(M);
    Y(i,:) = 0;
    Y(i,m) = 1;
end

end

