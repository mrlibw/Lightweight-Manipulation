import torch
import torch.nn as nn

import numpy as np
from miscc.config import cfg

from attention import func_attention
import torch.nn.functional as F
import torchvision.models as models
from torch.autograd import Variable
# ##################Loss for matching text-image###################
def cosine_similarity(x1, x2, dim=1, eps=1e-8):
    """Returns cosine similarity between x1 and x2, computed along dim.
    """
    w12 = torch.sum(x1 * x2, dim)
    w1 = torch.norm(x1, 2, dim)
    w2 = torch.norm(x2, 2, dim)
    return (w12 / (w1 * w2).clamp(min=eps)).squeeze()


def sent_loss(cnn_code, rnn_code, labels, class_ids,
              batch_size, eps=1e-8):
    # ### Mask mis-match samples  ###
    # that come from the same class as the real sample ###
    masks = []
    if class_ids is not None:
        for i in range(batch_size):
            mask = (class_ids == class_ids[i]).astype(np.uint8)
            mask[i] = 0
            masks.append(mask.reshape((1, -1)))
        masks = np.concatenate(masks, 0)
        masks = torch.BoolTensor(masks)
        if cfg.CUDA:
            masks = masks.cuda()

    if cnn_code.dim() == 2:
        cnn_code = cnn_code.unsqueeze(0)
        rnn_code = rnn_code.unsqueeze(0)

    cnn_code_norm = torch.norm(cnn_code, 2, dim=2, keepdim=True)
    rnn_code_norm = torch.norm(rnn_code, 2, dim=2, keepdim=True)
    scores0 = torch.bmm(cnn_code, rnn_code.transpose(1, 2))
    norm0 = torch.bmm(cnn_code_norm, rnn_code_norm.transpose(1, 2))
    scores0 = scores0 / norm0.clamp(min=eps) * cfg.TRAIN.SMOOTH.GAMMA3

    scores0 = scores0.squeeze()
    if class_ids is not None:
        scores0.data.masked_fill_(masks, -float('inf'))
    scores1 = scores0.transpose(0, 1)
    if labels is not None:
        loss0 = nn.CrossEntropyLoss()(scores0, labels)
        loss1 = nn.CrossEntropyLoss()(scores1, labels)
    else:
        loss0, loss1 = None, None
    return loss0, loss1


def words_loss(img_features, words_emb, labels,
               cap_lens, class_ids, batch_size):

    masks = []
    att_maps = []
    similarities = []
    cap_lens = cap_lens.data.tolist()
    for i in range(batch_size):
        if class_ids is not None:
            mask = (class_ids == class_ids[i]).astype(np.uint8)
            mask[i] = 0
            masks.append(mask.reshape((1, -1)))
        
        words_num = cap_lens[i]
        word = words_emb[i, :, :words_num].unsqueeze(0).contiguous()
        word = word.repeat(batch_size, 1, 1)
        context = img_features

        weiContext, attn = func_attention(word, context, cfg.TRAIN.SMOOTH.GAMMA1)

    
        att_maps.append(attn[i].unsqueeze(0).contiguous())
        word = word.transpose(1, 2).contiguous()
        weiContext = weiContext.transpose(1, 2).contiguous()
        word = word.view(batch_size * words_num, -1)
        weiContext = weiContext.view(batch_size * words_num, -1)
        #
        row_sim = cosine_similarity(word, weiContext)
        row_sim = row_sim.view(batch_size, words_num)

        row_sim.mul_(cfg.TRAIN.SMOOTH.GAMMA2).exp_()
        row_sim = row_sim.sum(dim=1, keepdim=True)
        row_sim = torch.log(row_sim)

        similarities.append(row_sim)

    similarities = torch.cat(similarities, 1)
    if class_ids is not None:
        masks = np.concatenate(masks, 0)
        masks = torch.BoolTensor(masks)
        if cfg.CUDA:
            masks = masks.cuda()

    similarities = similarities * cfg.TRAIN.SMOOTH.GAMMA3

    if class_ids is not None:
        similarities.data.masked_fill_(masks, -float('inf'))
    similarities1 = similarities.transpose(0, 1)

    if labels is not None:
        loss0 = nn.CrossEntropyLoss()(similarities, labels)
        loss1 = nn.CrossEntropyLoss()(similarities1, labels)
    else:
        loss0, loss1 = None, None
    return loss0, loss1, att_maps

# ##################Loss for G and Ds##############################
def discriminator_loss(netD, real_imgs, fake_imgs, conditions,
                       real_labels, fake_labels, words_embs, cap_lens, image_encoder, class_ids,
                        w_words_embs, wrong_caps_len, wrong_cls_id, word_labels):
    
    real_features = netD(real_imgs)
    fake_features = netD(fake_imgs.detach())
    
    cond_real_logits = netD.COND_DNET(real_features, conditions)
    cond_real_errD = nn.BCELoss()(cond_real_logits, real_labels)
    cond_fake_logits = netD.COND_DNET(fake_features, conditions)
    cond_fake_errD = nn.BCELoss()(cond_fake_logits, fake_labels)
    #
    batch_size = real_features.size(0)
    cond_wrong_logits = netD.COND_DNET(real_features[:(batch_size - 1)], conditions[1:batch_size])
    cond_wrong_errD = nn.BCELoss()(cond_wrong_logits, fake_labels[1:batch_size])

    if netD.UNCOND_DNET is not None:
        real_logits = netD.UNCOND_DNET(real_features)
        fake_logits = netD.UNCOND_DNET(fake_features)
        real_errD = nn.BCELoss()(real_logits, real_labels)
        fake_errD = nn.BCELoss()(fake_logits, fake_labels)
        errD = ((real_errD + cond_real_errD) / 2. +
                (fake_errD + cond_fake_errD + cond_wrong_errD) / 3.)
    else:
        errD = cond_real_errD + (cond_fake_errD + cond_wrong_errD) / 2.


    region_features_real, cnn_code_real = image_encoder(real_imgs)


    #w_result = word_level_correlation(region_features_real, w_words_embs, wrong_caps_len,
    #                                        batch_size, wrong_cls_id, fake_labels, word_labels)

    result = word_level_correlation(region_features_real, words_embs,
                                        cap_lens, batch_size, class_ids, real_labels, word_labels)

    errD += result

    return errD, result


def generator_loss(netsD, image_encoder, fake_imgs, real_labels,
                   words_embs, sent_emb, match_labels,
                   cap_lens, class_ids, style_loss, real_imgs):
    numDs = len(netsD)
    batch_size = real_labels.size(0)
    logs = ''
    # Forward
    errG_total = 0
    feature_loss = 0
    ## numDs: 3
    for i in range(numDs):
        features = netsD[i](fake_imgs[i])
        cond_logits = netsD[i].COND_DNET(features, sent_emb)
        cond_errG = nn.BCELoss()(cond_logits, real_labels)
        if netsD[i].UNCOND_DNET is  not None:
            logits = netsD[i].UNCOND_DNET(features)
            errG = nn.BCELoss()(logits, real_labels)
            g_loss = errG + cond_errG
        else:
            g_loss = cond_errG
        errG_total += g_loss
        logs += 'g_loss%d: %.2f ' % (i, g_loss)

    
        region_features, cnn_code = image_encoder(fake_imgs[i])
        w_loss0, w_loss1, _ = words_loss(region_features, words_embs,
                                            match_labels, cap_lens,
                                            class_ids, batch_size)
        w_loss = (w_loss0 + w_loss1) * \
                cfg.TRAIN.SMOOTH.LAMBDA

        s_loss0, s_loss1 = sent_loss(cnn_code, sent_emb,
                                         match_labels, class_ids, batch_size)
        s_loss = (s_loss0 + s_loss1) * \
                cfg.TRAIN.SMOOTH.LAMBDA

        errG_total += w_loss + s_loss
        logs += 'w_loss: %.2f s_loss: %.2f ' % (w_loss, s_loss)

        fake_img = fake_imgs[i]
        real_img = real_imgs[i]

        real_Gmatrix = style_loss(real_img)
        fake_Gmatrix = style_loss(fake_img)

        for i in range(len(real_Gmatrix)):
            cur_real_Gmatrix = real_Gmatrix[i]
            cur_fake_Gmatrix = fake_Gmatrix[i]
            feature_loss += F.mse_loss(cur_real_Gmatrix, cur_fake_Gmatrix) 
            
    errG_total += feature_loss / 2.
    logs += 'feature_loss: %.2f ' % (feature_loss / 2.)
    return errG_total, logs


##################################################################
def KL_loss(mu, logvar):
    # -0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD_element = mu.pow(2).add_(logvar.exp()).mul_(-1).add_(1).add_(logvar)
    KLD = torch.mean(KLD_element).mul_(-0.5)
    return KLD


##################################################################

def word_level_correlation(img_features, words_emb,
               cap_lens, batch_size, class_ids, labels, word_labels):
    
    masks = []
    att_maps = []
    result = 0
    cap_lens = cap_lens.data.tolist()
    similar_list = []

    for i in range(batch_size):
        if class_ids is not None:
            mask = (class_ids == class_ids[i]).astype(np.uint8)
            mask[i] = 0
            masks.append(mask.reshape((1, -1)))

        words_num = cap_lens[i]
        word = words_emb[i, :, :words_num].unsqueeze(0).contiguous()
        cur_word_labels = word_labels[i, :words_num]
        
        context = img_features[i, :, :, :].unsqueeze(0).contiguous()        
        weiContext, attn = func_attention(word, context, cfg.TRAIN.SMOOTH.GAMMA1)

        cur_weiContext = weiContext[0, :, :]
        cur_weiContext = cur_weiContext.transpose(0, 1)
        sum_weiContext = cur_weiContext.sum(dim=1, keepdim=False)
        soft_weiContext = nn.Softmax()(sum_weiContext)
        cur_result = nn.BCELoss()(soft_weiContext, cur_word_labels.float())

        result += cur_result

    return result


