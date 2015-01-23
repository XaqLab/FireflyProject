%load acc_vs_nuis_dim_T=500_Kf=2.mat
load acc_vs_nuis_dim_pcaresp_T=500_Kf=2.mat

dNuis
mcrPolicy
mcrResp
mcrPCAResp

figure;
plot(dNuis, 100*(1-mcrPolicy), 'b.', 'MarkerSize', 10);
hold on;
plot(dNuis, 100*(1-mcrResp), 'r.', 'MarkerSize', 10);
plot(dNuis, 100*(1-mcrPCAResp), 'g.', 'MarkerSize', 10);
hold off;
ylim([0 100.0]);
grid on;
xlabel('Dimension of Latent Nuisance Process');
ylabel('Cross-validated Classification Accuracy for Actions(%)')
title('Action Prediction Accuracy vs Nuisance Dimension');
legend('Policy Factors','Neural Responses')

SaveFigs = true;
if SaveFigs
    print -dpdf action_pred_acc_vs_nuis_dim; 
end
