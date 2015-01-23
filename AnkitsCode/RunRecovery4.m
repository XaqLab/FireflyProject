close all, clc;

T = 500
cvNumFolds = 2
dNuis = [30]
DoActionProbs = true

for i = 1:length(dNuis)
    i, dNuis(i)
    close all;
    [cvMCRPolicy,cvMCRResp, cvMCRPCAResp] = RecoverLatentComputation4(dNuis(i),cvNumFolds,T,DoActionProbs);
end


fprintf('Done.\n')