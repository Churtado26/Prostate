import torch
from pytorch_metric_learning import losses
from pytorch_metric_learning import miners

class SupervisedContrastiveLoss(torch.nn.Module):
    def __init__(self, temperature=0.1):
        super(SupervisedContrastiveLoss, self).__init__()
        self.temperature = temperature
        self.miner_function = miners.BatchEasyHardMiner(
            pos_strategy = miners.BatchEasyHardMiner.EASY,
            neg_strategy = miners.BatchEasyHardMiner.HARD
        )

    def forward(self, feature_vectors, labels):
        # Normalize feature vectors
        mined_hard_embeddings = self.miner_function(feature_vectors, torch.squeeze(labels))
        return losses.SupConLoss(temperature=self.temperature)(feature_vectors, torch.squeeze(labels), mined_hard_embeddings )
    
    

class TripletLoss(torch.nn.Module):
    def __init__(self, margin=0.05, positives = None, negatives = None):
        super(TripletLoss, self).__init__()
        self.margin = margin
        rules = {
            'easy': miners.BatchEasyHardMiner.EASY,
            'semihard': miners.BatchEasyHardMiner.SEMIHARD,
            'hard': miners.BatchEasyHardMiner.HARD
        }
        if positives is None or negatives is None:
            self.miner_function = None
            print('No mining will be applied')
        else:
            self.miner_function = miners.BatchEasyHardMiner(
                pos_strategy = rules[positives],
                neg_strategy = rules[negatives]
            )


    def forward(self, feature_vectors, labels):
        # Normalize feature vectors
        if self.miner_function is None:
            return losses.TripletMarginLoss(
                margin = self.margin,
                swap = False,
                smooth_loss = False, 
                triplets_per_anchor = 'all'
            )(feature_vectors, torch.squeeze(labels))
        else:
            mined_hard_embeddings = self.miner_function(feature_vectors, torch.squeeze(labels))
            return losses.TripletMarginLoss(
                margin=self.margin,
                swap = False,
                smooth_loss = False, 
                triplets_per_anchor = 'all'
            )(feature_vectors, torch.squeeze(labels), mined_hard_embeddings )
        
    
    
class NTXentLoss(torch.nn.Module):
    def __init__(self, temperature=0.07, positives = None, negatives = None):
        super(NTXentLoss, self).__init__()
        self.temperature = temperature
        rules = {
            'easy': miners.BatchEasyHardMiner.EASY,
            'semihard': miners.BatchEasyHardMiner.SEMIHARD,
            'hard': miners.BatchEasyHardMiner.HARD
        }
        if positives is None or negatives is None:
            self.miner_function = None
            print('No mining will be applied')
        else:
            self.miner_function = miners.BatchEasyHardMiner(
                pos_strategy = rules[positives],
                neg_strategy = rules[negatives]
            )

    def forward(self, feature_vectors, labels):
        # Normalize feature vectors
        if self.miner_function is None:
            return losses.NTXentLoss(
                temperature = self.temperature
            )(feature_vectors, torch.squeeze(labels))
        else:
            mined_hard_embeddings = self.miner_function(feature_vectors, torch.squeeze(labels))
            return losses.NTXentLoss(
                temperature=self.temperature
            )(feature_vectors, torch.squeeze(labels), mined_hard_embeddings )