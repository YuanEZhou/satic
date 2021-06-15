import torch
import torch.nn as nn
import torch.nn.functional as  F
import misc.utils as utils
from misc.rewards import init_scorer, get_self_critical_reward
import pdb

class LossWrapper(torch.nn.Module):
    #def __init__(self, model, opt):
    def __init__(self, model, teacher, opt):                    # add
        super(LossWrapper, self).__init__()
        self.opt = opt
        self.model = model
        self.teacher = teacher                                 # add
        if opt.label_smoothing > 0:
            self.crit = utils.LabelSmoothing(smoothing=opt.label_smoothing)
        else:
            self.crit = utils.LanguageModelCriterion()
        self.rl_crit = utils.RewardCriterion()
        #pdb.set_trace()
        self.kl_div = nn.KLDivLoss(log_target=True, reduction='batchmean')                               # add

    def forward(self, fc_feats, att_feats, labels, masks, att_masks, gts, gt_indices,
                sc_flag):
        out = {}
        if not sc_flag:
            # loss = self.crit(self.model(fc_feats, att_feats, labels, att_masks), labels[:,1:], masks[:,1:])
            #==================================================
            # pdb.set_trace()
            student_output = self.model(fc_feats, att_feats, labels, att_masks)
            with torch.no_grad():
                teacher_output = self.teacher(fc_feats, att_feats, labels, att_masks)
            loss1 = self.crit(student_output, labels[:,1:], masks[:,1:])
            
            if student_output.size(1) != teacher_output.size(1):
                loss2 = self.kl_div(student_output[masks[:,1:student_output.size(1)+1].bool()],teacher_output[masks[:,1:teacher_output.size(1)+1].bool()])
            else:
                loss2 = self.kl_div(student_output[masks[:,1:student_output.size(1)+1].bool()],teacher_output[masks[:,1:student_output.size(1)+1].bool()])
            
            loss = loss1+ self.opt.kd_weight * loss2
            #==================================================
        else:
            gen_result, sample_logprobs = self.model(fc_feats, att_feats, att_masks, opt={'sample_max':0}, mode='sample')
            gts = [gts[_] for _ in gt_indices.tolist()]
            reward = get_self_critical_reward(self.model, fc_feats, att_feats, att_masks, gts, gen_result, self.opt)
            reward = torch.from_numpy(reward).float().to(gen_result.device)
            loss = self.rl_crit(sample_logprobs, gen_result.data, reward)
            out['reward'] = reward[:,0].mean()
        out['loss'] = loss
        return out
