close all, clc;

T = 500
cvNumFolds = 2
dNuis = [3 5 10 15 20 30 40 50 60 70 80 90 ]
DoActionProbs = false
SaveFigs = true;

for i = 1:length(dNuis)
    i, dNuis(i)
    close all;
    [cvMCRPolicy,cvMCRResp,cvMCRPCAResp] = RecoverLatentComputation4(dNuis(i),cvNumFolds,T,DoActionProbs);
    mcrPolicy(i) = cvMCRPolicy;
    mcrResp(i) = cvMCRResp;
    mcrPCAResp(i) = cvMCRPCAResp;
end

save acc_vs_nuis_dim_pcaresp_T=500_Kf=2.mat T dNuis mcrPolicy mcrResp;

dNuis
mcrPolicy
mcrResp

figure;
plot(dNuis, 100*(1-mcrPolicy), 'b.', 'MarkerSize', 10);
hold on;
plot(dNuis, 100*(1-mcrResp), 'r.', 'MarkerSize', 10);
plot(dNuis, 100*(1-mcrPCAResp), 'g.', 'MarkerSize', 10);
hold off;
ylim([0 100.0]);
grid on;
xlabel('Dimension of Latent Nuisance Process');
ylabel('Cross-validated Classification Accuracy for Actions')
title('Action Prediction Accuracy vs Nuisance Dimension');

if SaveFigs, axis off; print -dpdf action_pred_acc_vs_nuis_dim; end
