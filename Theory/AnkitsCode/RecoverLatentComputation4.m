function [cvMCRPolicySmooth,cvMCRResponseSmooth, cvMCRPCAResponseSmooth] = RecoverLatentComputation4(dN,cvNumFolds,T, DoActionProbs)
seed = rng
SaveFigs = false;

%% Low-dimensional Dynamics
d = 3; % dimension of latent factor (computational state)
%P = 0.99 * eye(d); % dynamics matrix (should be related to computation/policy)
%P = eye(3); P = P(randperm(3),:)
ShiftRate = 0.025;
P = eye(d); P = P([2 3 1],:) * ShiftRate + eye(d) * (1-ShiftRate); P = 0.95 * P
DynamicsSigma = 1.0;

%% High-dimensional Embedding
D = 100; % number of neurons
E = randn(D,d); % embedding from latent computation space R^d to neural firing space R^D
ResponseSigma = 1.0;
NumIters = T;

%% Simulate AR(1) Latent Policy process
spec = struct();
spec.seed = seed;
spec.d = d;
spec.P = P;
spec.DynamicsSigma = DynamicsSigma;
spec.NumIters = NumIters
x = simAutoRegressiveProcess(spec);

% Random Embedding into high-dimensional neural response space
y = E * x + ResponseSigma * randn(D,NumIters);

% Simulate AR(1) Latent Nuisance process if desired
% This introduces structured noise that interferes with the latent space
% we care about. It models e.g. motor variability, attention, thoughts,
% circuits noise, etc.
AddNuisanceProcess = true
if AddNuisanceProcess
    % Choose Nuisance process params s.t. they have structure and they are
    % in a larger latent space than the computation we care about.
    dN = 30; % Choose latent dimension high relative to dim of latent subspace
    NP = randpermmtx(dN);
    NP = NP * ShiftRate + eye(dN) * (1-ShiftRate); 
    NP = 0.95 * NP;
    NE = randn(D,dN);

    % Simulate Latent Nuisance process
    specN = struct();
    specN.d = dN;
    specN.P = NP;
    specN.DynamicsSigma = DynamicsSigma;
    specN.NumIters = NumIters;
    [xN] = simAutoRegressiveProcess(specN);
    
    % Embed into high-dim neural response space
    yN = NE * xN; % no obs noise since already included in latent process
    y = y + yN;
end

%% Map latent policy process to actions
EMATimePeriod = 1
%EMATimePeriod = 0.10 * NumIters
%xsmooth = tsmovavg(x,'e',EMATimePeriod,2);
%ysmooth = tsmovavg(y,'e',EMATimePeriod,2);
xsmooth = x;
ysmooth = y;
[xmax,aOld] = max(xsmooth);
prob_a = softmax(xsmooth);
[pmax,a] = max(prob_a,[],1);

% Plot simulated dynamics
PlotLatentDynamics = true;
if PlotLatentDynamics
    figure;
    plot(x(1,:), 'r-');
    hold on;
    plot(x(2,:), 'g-');
    plot(x(3,:), 'b-');
    hold off;
    xlabel('Time (iters)');
    ylabel('Latent State (a.u.)');
    title('Raw Latent State vs Time');
    if SaveFigs, axis off; print -dpdf latent_dyn.pdf; end

    figure;
    plot(xsmooth', '-');
    hold on;
    plot(xmax, 'k-');
    hold off;
    xlabel('Time (iters)');
    ylabel('Smoothened Latent State (a.u.)');
    title('Smoothened Latent State vs Time');
    if SaveFigs, axis off; print -dpdf latent_dyn_smooth.pdf; end

    figure;
    plot(prob_a', '-');
    ylim([0 1]);
    xlabel('Time (iters)');
    ylabel('Action Probability');
    grid on;
    if SaveFigs, axis off; print -dpdf action_prob_dyn.pdf; end

    figure;
    imagesc(prob_a); 
    colormap(gray); colorbar;
    xlabel('Time (iters)');
    ylabel('Action or Choice');
    title('Action/Choice Probability vs Time');
    if SaveFigs, axis off; print -dtiff action_prob_dyn_imgsc.tiff; end
    
    
    figure;
    plot(a, 'k-');
    ylim([0 4]);
    xlabel('Time (iters)');
    ylabel('Actions or Choices');
    if SaveFigs, axis off; print -dpdf actions_dyn.pdf; end
    
    
    figure;
    plot(xN');
    xlabel('Time (iters)');
    ylabel('Latent Nuisance State (a.u.)');
    if SaveFigs, axis off; print -dpdf latent_dyn_nuisance.pdf; end
end

PlotResponseDynamics = true;
if PlotResponseDynamics
    figure; 
    imagesc(tanh(y ./ std(y(:)))); colormap(gray)
    xlabel('Time(iters)'); 
    ylabel('Neuron');
    title('Neural Response Rate Raster');
    if SaveFigs, axis off; print -dtiff response_dyn.tiff; end
    if SaveFigs, axis off; print -dpdf response_dyn.pdf; end

end

%% Estimate E = embedding matrix
UseLM = false
if UseLM
    E_est = nan(size(E));
    for i = 1:D
        mdl = LinearModel.fit(x',y(i,:)','Intercept',false);
        E_est(i,:) = single(mdl.Coefficients(:,1))';
    end
else
    E_est = y / x;
end
errE = mae(E, E_est)

% Estimate projection matrix = pseudoinverse of E
% TODO: Show as colored connections from neurons to latent factors
Einv_est = pinv(E_est);
figure; 
imagesc(Einv_est'); colormap(gray);
xlabel('Latent Factor'); 
ylabel('Neuron'); 
title('Projection to Latent Space')
if SaveFigs, axis off; print -dtiff response_to_latent_weights.tiff; end

%% Recover P = latent factor transition mtx (e.g. policy/computation)
x_est = Einv_est * y;
figure;
plot(x(1,:), 'r-');
hold on;
plot(x_est(1,:), 'r--');
hold off;
xlabel('Time (iters)');
ylabel('Latent State (a.u.)');
title('Estimated Latent State vs. Ground Truth');
if SaveFigs, axis off; print -dpdf latent_state_est_vs_truth; end

if UseLM
    P_est = nan(size(P));
    for i = 1:d
        %mdl = LinearModel.fit(x(:,1:(NumIters-1))',x(i,2:NumIters)','Intercept',false);
        mdl = LinearModel.fit(x_est(:,1:(NumIters-1))',x_est(i,2:NumIters)','Intercept',false);
        P_est(i,:) = single(mdl.Coefficients(:,1))';
    end
else
    P_est = x_est(:,2:end) / x_est(:,1:end-1);
end

% Estimate error
errP = mae(P, P_est)

% Show comparison between true and estimate policy matrix
P
P_est
figure;
subplot(2,1,1); imagesc(P); title('True Policy Matrix');
subplot(2,1,2); imagesc(P_est); title('Est. Policy Matrix');
if SaveFigs, axis off; print -dtiff policy_est_vs_truth; end


%% Recover T = neural response transition matrix
T = E * P * pinv(E);
T_est_comp = E_est * P_est * Einv_est;
if UseLM
    T_est_null = nan(D);
    for i = 1:D
        mdl = LinearModel.fit(y(:,1:(NumIters-1))',y(i,2:NumIters)','Intercept',false);
        T_est_null(i,:) = single(mdl.Coefficients(:,1))';
    end
else
    T_est_null = y(:,2:end) / y(:,1:end-1);
end

% Estimate error
errT_comp = mae(T, T_est_comp)
errT_null = mae(T, T_est_null)

% Show comparison between response dynamics estimated w/ and w/o Latent
% (Policy) information. Shows value-add of having a computational model
% of the task (assuming monkey has learned it well).
figure;
subplot(3,1,1); imagesc(T); title('True Neural Transition Matrix');
subplot(3,1,2); imagesc(T_est_null); title('Est. Neural Transition Matrix (No Info about Computation)');
subplot(3,1,3); imagesc(T_est_comp); title('Est. Neural Transition Matrix (With Computational Model)');
if SaveFigs, axis off; print -dtiff response_txn_est_vs_truth; end


%% THE STATISTICAL STRAW MAN

% How well would PCA have estimated the latent space's dimension?
[coeffs,y_pca_full,u_latent,u_tsquared,explained,u_mu] = pca(y');
y_pca = y_pca_full(:,1:33);
y_pca_smooth = tsmovavg(y_pca,'e',EMATimePeriod,1);
figure;
plot(explained, 'b.', 'MarkerSize', 12);
grid on;
xlabel('Principal Component');
ylabel('Pct Variance Explained');
title('Can PCA Recover the Latent Dimension?');
if SaveFigs, print -dpdf response_pca_var_explained; end

fprintf('ACTIONS CLASSIFICATION + CROSS-VALIDATION:\n')
%cvNumFolds = 2
classifierType = 'mnlr'
DoConfusion = false;
%DoConfusion = true;

% How well can we predict actions a_t from just neural responses y_t?
% We compute misclassification rate (MCR) and confusion matrices (CONF)
fprintf('CLASSIFY: Actions ~ Neural Responses\n');
%[cvMCRResponse,cvConfusionResponse] = cvclassify(y',a',cvNumFolds, DoConfusion)
[cvMCRResponseSmooth,cvConfusionResponseSmooth] = cvclassify(ysmooth',a',cvNumFolds, DoConfusion)


fprintf('CLASSIFY: Actions ~ PCA(Neural Responses)\n');
[cvMCRPCAResponseSmooth,cvConfusionPCAResponseSmooth] = cvclassify(y_pca_smooth,a',cvNumFolds, DoConfusion)

% How well can we predict actions a_t from just policy estimates xhat_t?
% We compute misclassification rate (MCR) and confusion matrices (CONF)
fprintf('CLASSIFY: Actions ~ Latent Policy Factors\n');
%[cvMCRPolicy,cvConfusionPolicy] = cvclassify(x',a',cvNumFolds, DoConfusion)
[cvMCRPolicySmooth,cvConfusionPolicySmooth] = cvclassify(xsmooth',a',cvNumFolds, DoConfusion)


% Show true actions and all posterior probs from truth, us, straw man
if DoActionProbs
    prob_a = prob_a'; % true generating probabilities of actions
    prob_a_policy = nan(size(prob_a));
    prob_a_resp = nan(size(prob_a));
    prob_a_resp_pca = nan(size(prob_a));
    indices = crossvalind('Kfold',a',cvNumFolds);
    for i = 1:cvNumFolds
        test = (indices == i); train = ~test;
        prob_a_policy(test,:) = post_probs_mnlr(xsmooth(:,train)', a(:,train)', xsmooth(:,test)');
        prob_a_resp(test,:) = post_probs_mnlr(ysmooth(:,train)', a(:,train)', ysmooth(:,test)');
        prob_a_resp_pca(test,:) = post_probs_mnlr(y_pca_smooth(train,:), a(:,train)', y_pca_smooth(test,:));
    end

    %[pmax,a] = max(prob_a,[],1);
    row_a = zeros(1,NumIters,3);
    for t = 1:NumIters
        row_a(1,t,a(t)) = 1;
    end
    row_prob_a = reshape(prob_a,[1 NumIters 3]);
    row_prob_a_policy = reshape(prob_a_policy,[1 NumIters 3]);
    row_prob_a_resp = reshape(prob_a_resp,[1 NumIters 3]);
    row_prob_a_resp_pca = reshape(prob_a_resp_pca,[1 NumIters 3]);
    %row_zeros = zeros(size(row_prob_a));
    rows = [ row_a; row_prob_a; row_prob_a_policy; row_prob_a_resp; row_prob_a_resp_pca ];
    figure; image(rows)
    xlabel('Time (iters)');
    ylabel('pca resp policy true actions');
    title('True Actions and Probabilities vs Predictions');
    if SaveFigs, axis off; print -dtiff action_probs_true_vs_est; end
end

%% Compare error covariance of y_t autoregression to derived
DoYErrCovAnalysis = false;
if DoYErrCovAnalysis
    yerrcov = DynamicsSigma^2 * E * E' + ResponseSigma * eye(D);
    yhat = T_est_null * y(:,1:end-1);
    yerr = yhat - y(:,2:end);
    yerrcov_est = cov(yerr');
    yerrcov_diff = mae(yerrcov,yerrcov_est)
    yerrcov_rdiff = mare(yerrcov,yerrcov_est)
    yerrcov(1:5,1:5)
    yerrcov_est(1:5,1:5)
end

%% Save important data to file
fprintf('SAVING VARIABLES TO FILE\n');
save seed_figures_xaq_0316.mat seed spec specN;

fprintf('Done.\n');

end % main function

%% Helper functions
function [y] = logistic(x)
    y = 1 ./ (1 + exp(-x));
end

function [err] = mae(M, Mest)
    err = mean(abs(Mest(:)-M(:)));
end

function [relerr] = mare(M, Mest)
    relerr = mean(abs( (Mest(:)-M(:)) ./ M(:)));
end

function [x] = simAutoRegressiveProcess(P)
    x = zeros(P.d, P.NumIters);

    % Initialize states and observables
    x(:,1) = P.DynamicsSigma * randn(P.d,1);

    % Simulate dynamics
    for t = 2:P.NumIters
        x(:,t) = P.P * x(:,t-1) + P.DynamicsSigma * randn(P.d,1);
    end
end

function [P] = randpermmtx(d)
    P = eye(d);
    P = P(randperm(d),:);
end

function [y] = simEmbeddingProcess(P, x)
    y = zeros(P.D, P.NumIters);

    % Initialize states and observables
    y(:,1) = P.E * x(:,1) + P.ResponseSigma * randn(P.D,1);

    % Simulate dynamics
    for t = 2:P.NumIters
        y(:,t) = P.E * x(:,t) + P.ResponseSigma * randn(P.D,1);
    end
end

function [pihat] = post_probs_mnlr(Xtrain, ytrain, Xtest)
    % IMPORTANT: ytrain might be nominal, must cast to double
    B = mnrfit(Xtrain, double(ytrain));
    pihat = mnrval(B,Xtest);
    %[~,yhat] = max(pihat,[],2);
end
