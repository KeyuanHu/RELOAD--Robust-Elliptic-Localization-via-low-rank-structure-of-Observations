function x0_est = RELOAD(r_bar,Xt,Xr,U,D)

[M,L] = size(r_bar);
iter = 500;
thre = 0.00000001;
lambda1_0 = 10000;
lambda1 = lambda1_0;
epsilon1 = lambda1;
mu = 0.95;
v = [ones(M,1);-1*ones(L,1)];

x_initial = LLS(Xt,Xr,r_bar);
dt_initial = sqrt(sum((Xt - x_initial*ones(1,M)).^2,1)).';
dr_initial = sqrt(sum((Xr - x_initial*ones(1,L)).^2,1)).';
d_initial = [dt_initial;dr_initial];
e = zeros(M+L,1);
z = zeros(4,1);
d_est = d_initial;

a = sum(r_bar,2)/L;
b = sum(r_bar,1)/M;
b = b';
r_mean = sum(sum(r_bar,2))/(M*L);
a = a-r_mean;
y = [a;b]; 
t = mean((d_initial - y).*v);
y_square = y.^2;

for run = 1:iter
run;
z_old = z;
d_old = d_est;

%update z
y1 = (y+t*v).^2 - e;
z = inv(U'*U)*U'*y1;

%update e
y2 = (y+t*v).^2 - U*z;
e = max(prox_HOC(y2,lambda1,epsilon1),0);


%update t
p = y;
h = y.^2 -U*z - e;
t = solve_t_cardano(h, p, M, L);

%update lambda1
T = y2 - e; 
t_m_n = T(find(T));
lambda1_1 = 0.7413*(quantile(t_m_n,0.75)-quantile(t_m_n,0.25));
lambda1_1 = lambda1_0*mu^run + lambda1_1;
lambda1 = min(lambda1,lambda1_1);
epsilon1 = lambda1;

d_est = U*z;
d_est = sqrt(d_est);

if (norm(d_est - d_old)/norm(d_old)) < thre
    break;
end


end
d_est = U*z;
ds_est = sqrt(d_est);
D_est = zeros(M+L+1,M+L+1);
D_est(1:end-1,1:end-1) = D;
D_est(1:end-1,end) = d_est;
D_est(end,1:end-1) = d_est';
X_mds = classical_mds(D_est,2);
X_aligned = align_and_get_last([Xt,Xr]',X_mds);
x0_est = X_aligned(end, :);



end

function prox_u = prox_HOC(u, lambda, epsilon)

    term = abs(u) - (abs(u).*(epsilon^2+lambda^2)) ./ (epsilon^2+u.^2);
    prox_u = max(0, term) .* sign(u);
end

function t_opt = solve_t_cardano(x, y, M, L)

    v = [ones(M,1); -ones(L,1)];
    x = x(:); y = y(:); v = v(:);

    n   = numel(x);
    S1  = v.'*y;
    S2  = sum(y.^2);
    Sx  = sum(x);
    Sxy = sum((v.*y).*x);

    A = n;
    B = 3*S1;
    C = 2*S2 + Sx;
    D = Sxy;

    shift = B/(3*A);
    p = (3*A*C - B^2)/(3*A^2);
    q = (2*B^3 - 9*A*B*C + 27*A^2*D)/(27*A^3);

    Delta = (q/2)^2 + (p/3)^3;

    f = @(t) sum((t.^2 + 2*(v.*y)*t + x).^2);

    if Delta > 0
        u1 = nthroot(-q/2 + sqrt(Delta), 3);
        u2 = nthroot(-q/2 - sqrt(Delta), 3);
        u  = u1 + u2;
        t_opt = u - shift;
        f_min = f(t_opt);
    else
        r = sqrt(-p/3);
        phi = acos(-q/(2*r^3));
        u = zeros(3,1);
        t_candidates = zeros(3,1);
        fvals = zeros(3,1);
        for k = 0:2
            u(k+1) = 2*r*cos((phi+2*pi*k)/3);
            t_candidates(k+1) = u(k+1) - shift;
            fvals(k+1) = f(t_candidates(k+1));
        end
        [f_min, idx] = min(fvals);
        t_opt = t_candidates(idx);
    end
end

function X = classical_mds(D, d)

    n = size(D,1);

    D2 = D;

    J = eye(n) - ones(n)/n;
    B = -0.5 * J * D2 * J;   

    [V, Lambda] = eig(B);

    [eigvals, idx] = sort(diag(Lambda), 'descend');
    V = V(:, idx);

    eigvals_d = eigvals(1:d);
    eigvals_d(eigvals_d < 0) = 0;
    V_d = V(:, 1:d);

    X = V_d * diag(sqrt(eigvals_d));

end

function x_last = align_and_get_last(X_true, X_mds)

    [n, d] = size(X_true);

    X_mds_known = X_mds(1:n, :);

    [~, Z, transform] = procrustes(X_true, X_mds_known);

    X_mds_all = X_mds;  

    X_aligned = transform.b*X_mds_all*transform.T + transform.c(1,:);

    x_last = X_aligned;
end
