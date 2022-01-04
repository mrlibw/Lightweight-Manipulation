import torch
import torch.nn as nn

import numpy as np
from miscc.config import cfg

from attention import func_attention
import torch.nn.functional as F
import torchvision.models as models
from torch.autograd import Variable


def blurring(img, k_size=5, s=1, pad=2):
    k_size = 9  # 7
    pad= 4  # 3
    _blur_filter = torch.ones([3, 3, k_size, k_size]).to('cuda')
    blur_filter = _blur_filter.view(3, 3, k_size, k_size) / (k_size**2 * 3)  # first element is output ch. https://pytorch.org/docs/stable/generated/torch.nn.functional.conv2d.html
    # gray = getGrayImage(img)
    out = torch.nn.functional.conv2d(input=img,
                                    weight=Variable(blur_filter),
                                    stride=s,
                                    padding=pad)
    return out

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
        if(len(mask.shape)==1):
            mask = mask[None,:]
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
                        w_words_embs, wrong_caps_len, wrong_cls_id, word_labels, epoch, cfg):
    
    real_features = netD(real_imgs, epoch)  # shape: [10, 512, 4, 4]
    fake_features = netD(fake_imgs.detach(), epoch)

    
    cond_real_logits = netD.COND_DNET(real_features, conditions)  # shape : [10]
    cond_real_errD = nn.BCELoss()(cond_real_logits, real_labels)
    cond_fake_logits = netD.COND_DNET(fake_features, conditions)
    cond_fake_errD = nn.BCELoss()(cond_fake_logits, fake_labels)
    # ------ blurr real image loss ------
    if cfg.TRAIN.USE_BLUR_REAL_IMAGE:
        blurred_real_img = blurring(real_imgs)
        blur_real_features = netD(blurred_real_img, epoch)
        # print("what is conditions??")
        cond_blurred_real_logits  = netD.COND_DNET(blur_real_features, conditions)  # conditions is sentence embeddings
        cond_blurred_real_errD = nn.BCELoss()(cond_blurred_real_logits, fake_labels)
    # ------------- done ----------------
    #
    batch_size = real_features.size(0)
    cond_wrong_logits = netD.COND_DNET(real_features[:(batch_size - 1)], conditions[1:batch_size])
    cond_wrong_errD = nn.BCELoss()(cond_wrong_logits, fake_labels[1:batch_size])

    if netD.UNCOND_DNET is not None:
        real_logits = netD.UNCOND_DNET(real_features)
        fake_logits = netD.UNCOND_DNET(fake_features)
        real_errD = nn.BCELoss()(real_logits, real_labels)
        fake_errD = nn.BCELoss()(fake_logits, fake_labels)

        if cfg.TRAIN.USE_BLUR_REAL_IMAGE:
            errD = ((real_errD + cond_real_errD) / 2. +
                (fake_errD + cond_fake_errD + cond_wrong_errD) / 3.)  + cond_blurred_real_errD
        else:
            errD = ((real_errD + cond_real_errD) / 2. +
                (fake_errD + cond_fake_errD + cond_wrong_errD) / 3.)  # original
    else:

        if cfg.TRAIN.USE_BLUR_REAL_IMAGE:
            errD = cond_real_errD + (cond_fake_errD + cond_wrong_errD) / 2. + cond_blurred_real_errD
        else:
            errD = cond_real_errD + (cond_fake_errD + cond_wrong_errD) / 2.  # original



    region_features_real, cnn_code_real = image_encoder(real_imgs)


    #w_result = word_level_correlation(region_features_real, w_words_embs, wrong_caps_len,
    #                                        batch_size, wrong_cls_id, fake_labels, word_labels)

    result = word_level_correlation(region_features_real, words_embs,
                                        cap_lens, batch_size, class_ids, real_labels, word_labels)
    
    # # UNCOMMENT HERE to activate NSL
    # result = sharpness_loss(fake_imgs)

    errD += result

    return errD, result


def generator_loss(netsD, image_encoder, fake_imgs, real_labels,
                   words_embs, sent_emb, match_labels,
                   cap_lens, class_ids, style_loss, real_imgs, epoch):
    numDs = len(netsD)
    batch_size = real_labels.size(0)
    logs = ''
    # Forward
    errG_total = 0
    feature_loss = 0
    ## numDs: 3
    for i in range(numDs):
        features = netsD[i](fake_imgs[i], epoch)
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

def M_loss(real, fake, mask):
    m_loss = torch.mean((torch.abs(real - fake))*mask) * 25
    return m_loss


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



def sharpness_loss(img, percent_grad=0.4, gamma=1, kmean_step = 5):
    laplacian_filter = torch.FloatTensor([[0, 1, 0], [1, -4, 1], [0, 1, 0]]).view(1, 1, 3, 3).cuda()
    
    diff = torch.mean(img,dim=1).unsqueeze(1) #rgb to greyscale
    diff = torch.nn.functional.pad(diff,(1,1,1,1),mode='replicate')
    diff = torch.nn.functional.conv2d(input=diff, weight=Variable(laplacian_filter), stride=1, padding=0) # the gradient image
    diff = torch.abs(diff)
    batch_size = diff.size(0)

    # # method 1
    # with torch.no_grad():
    #     caculate the threshold
    #     cum_val,_ = torch.sort(diff.view(batch_size,-1), dim=1, descending=True) # sorting
    #     cum_val = torch.cumsum(cum_val,dim=1)
    #     cum_val = cum_val/(cum_val[:,-1].unsqueeze(1))
    #     thrs_idx = torch.sum(cum_val<percent_grad, dim=1)
    #     thrs = cum_val[np.arange(batch_size), thrs_idx]
    # 
    # diff = torch.tanh(gamma*diff) # activation function to settle the value 
    # target = torch.ones_like(diff)
    # loss = torch.sum(F.binary_cross_entropy(diff,target,reduction='none')*high_grad)/n_high_grad
    
    # method 2
    with torch.no_grad():
        thrs = torch.mean(diff.view(batch_size,-1), dim=1)
        for i in range(kmean_step):
            mask = diff>thrs[:,None,None,None]
            m1 = torch.sum(diff*mask,dim=[1,2,3])/torch.sum(mask,dim=[1,2,3])
            mask = torch.logical_not(mask)
            m2 = torch.sum(diff*mask,dim=[1,2,3])/torch.sum(mask,dim=[1,2,3])
            thrs = (m1+m2)/2.0
          
        # get the mask indicating high gradient value
        high_grad = diff>thrs[:,None,None,None]
        n_high_grad = torch.sum(high_grad)
    
    diff = torch.tanh(gamma*diff) # activation function to settle the value 
    target = torch.ones_like(diff)
    loss = torch.sum(F.binary_cross_entropy(diff,target,reduction='none')*high_grad)/n_high_grad
    
    # # method 3
    # with torch.no_grad():
        # target = torch.tanh(diff*10)
    # loss = F.binary_cross_entropy(diff,target,reduction='mean')
        
    return loss
