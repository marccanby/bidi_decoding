import torch
import random

def reversePaddedArray(array, pad_idx=1): # good function: reverse sequences when there is padding on end (leave padding on end)
    assert len(array.shape) == 2
    shifts = array.shape[1] - (array != pad_idx).sum(dim=1)
    myrange = torch.arange(array.shape[1]-1, -1, -1, device=array.device).unsqueeze(0).repeat(array.shape[0], 1)
    myrange = myrange - shifts.unsqueeze(1)
    myrange[myrange < 0] = array.shape[1]-1
    res = torch.gather(array, 1, index=myrange)
    assert torch.all(res > 0), 'Did not support this case.'
    return res

def reverseSequence(input_l2r, trg_l2r, pad_idx=1):
    input_r2l = reversePaddedArray(trg_l2r)
    trg_r2l = reversePaddedArray(input_l2r)

    return input_r2l, trg_r2l

def createGatherers(length):
    '''
    Returns indices of all cells relevant for this: (0,0), (0,1), ... (basically the upper diagonal table)
    '''
    idxes = torch.triu_indices(length-1, length-1).T # -1 b/c in either direction the last item doesn't get fed in
    idxes[:,1] = idxes[:,1] - idxes[:,0]
    join_answers = (idxes.sum(dim=1) == length-2).long()
    return idxes, join_answers

def makeJaggedMask(length, upto):
    # length is the num cols in the tensor; upto is the index (inclusive) up to which you want to put a 1
    # note upto may have -1's in it, indicating that that row should be all 0's
    # note: it fills up to upto with 1's, everything after that is 0's
    result2 = torch.zeros((upto.shape[0], length+1), device=upto.device)
    result2[(torch.arange(result2.shape[0]), upto+1)] = 1
    result2 = 1 - result2.cumsum(dim=-1)
    result2 = result2[:,:-1]
    return result2

def subsetTensorToSpecificIdxes(tensor, idxes, pad,length=None):
    # idxes are inclusive
    assert len(tensor.shape) == 2
    assert len(idxes.shape) == 1
    assert tensor.shape[0] == idxes.shape[0]
    mask = makeJaggedMask(tensor.shape[1], idxes.long())
    t = (tensor+1) * mask - 1 # +-1 business is to deal with case if there was a 0 in the original matrix taht shouldnt be padding
    length = length if length is not None else idxes.max().int()+1
    t = t[:,:length]
    t[t==-1] = pad

    return t

def compute_advanced_loss(probs_l2r, trg_l2r, probs_r2l, trg_r2l, probs_join, gold_join, probs_order,
                          gath_concat, gath_wd_idxes, reduce, return_each_prob):

    assert probs_l2r.shape[0] == probs_r2l.shape[0] # no matter what
    is_full_bidi = probs_l2r.shape[0] == probs_join.shape[0]
    if not is_full_bidi:
        num_words = probs_l2r.shape[0]  # len(dp_getter_idxes_diag[0])  # i.e. bs
        assert num_words == trg_l2r.shape[0]
    else:
        num_words = trg_l2r.shape[0]
    max_len = torch.sum(gath_concat, dim=1).max() + 2

    final_probabilities = torch.ones(num_words, device=probs_l2r.device)

    for i in range(0, max_len - 1):
        # Get all the info related to this diagonal
        diag_bools = gath_concat.sum(dim=1) == i
        diag_idxes = diag_bools.nonzero().squeeze(1)
        diag_wds = gath_wd_idxes[diag_bools]
        diag_vals = gath_concat[diag_bools]

        # Important: the order of the columns (dim=1) here, currentlty starts at the top of the diagonal
        join_prob = probs_join[diag_idxes, :].reshape(-1, i + 1, 2)
        order_prob = probs_order[diag_idxes, :].reshape(-1, i + 1, 2)
        if not is_full_bidi:
            left_wd_prob = probs_l2r[diag_wds, diag_vals[:, 0], :].reshape(-1, i + 1, probs_l2r.shape[2])
            rt_wd_prob = probs_r2l[diag_wds, diag_vals[:, 1], :].reshape(-1, i + 1, probs_r2l.shape[2])
        else:
            left_wd_prob = probs_l2r[diag_idxes,:].reshape(-1, i + 1, probs_l2r.shape[1])
            rt_wd_prob = probs_r2l[diag_idxes,:].reshape(-1, i + 1, probs_r2l.shape[1])
        correct_wd_left = trg_l2r[diag_wds, diag_vals[:, 0]].reshape(-1, i + 1, 1).type(torch.int64)
        correct_wd_rt = trg_r2l[diag_wds, diag_vals[:, 1]].reshape(-1, i + 1, 1).type(torch.int64)
        assert len({join_prob.shape[0], order_prob.shape[0], left_wd_prob.shape[0], rt_wd_prob.shape[0],
                    correct_wd_left.shape[0], correct_wd_rt.shape[0]}) == 1
        assert len({join_prob.shape[1], order_prob.shape[1], left_wd_prob.shape[1], rt_wd_prob.shape[1],
                    correct_wd_left.shape[1], correct_wd_rt.shape[1]}) == 1

        left_wd_prob = torch.gather(left_wd_prob, dim=2, index=correct_wd_left)  # ~bs x diag x 1
        rt_wd_prob = torch.gather(rt_wd_prob, dim=2, index=correct_wd_rt)

        # Get joining and not joinging idxes (not actually idxes, they're bools)
        joining_idxes = gold_join[diag_bools].reshape(-1, i + 1).bool()
        joining_idxes = joining_idxes[:, 0]  # all rows should be the same all the way across ( maybe add an assert?)
        not_joining_idxes = ~joining_idxes

        # Get join probabilities for this diag
        # if self.exp_join:
        jidxes = (joining_idxes).type(torch.int64).unsqueeze(1)  # this gives us 0's and 1's
        jidxes = jidxes.repeat(1, join_prob.shape[1]).unsqueeze(2)
        join_prob = torch.gather(join_prob, dim=2, index=jidxes).squeeze(
            2)  # join_prob here needs to be the probability of the correct action (join or not join)

        # Now, compute this cell
        if i == 0:
            new_prob = torch.zeros(join_prob.shape[0], 1, device=probs_l2r.device)  # ~bs x diag
        else:
            wd_ids = torch.unique_consecutive(diag_wds)
            assert new_prob[wd_ids].shape == to_pass_on_rt.shape

            new_prob_left = new_prob[wd_ids] + to_pass_on_rt + join_prob[:, :-1]
            new_prob_above = new_prob[wd_ids] + to_pass_on_left + join_prob[:, 1:]
            # be careful about which side inf is concatted on; this has the diags ordered starting from top, so inf needs to be as follows
            inf = torch.zeros(new_prob_left.shape[0], 1, device=probs_l2r.device)
            inf[:] = -float('inf')  # -infinity is the null value for logsumexp operation
            new_prob_left = torch.cat([new_prob_left, inf], dim=1)
            inf = torch.zeros(new_prob_above.shape[0], 1, device=probs_l2r.device)
            inf[:] = -float('inf')
            new_prob_above = torch.cat([inf, new_prob_above], dim=1)
            stacked = torch.stack([new_prob_left, new_prob_above], 2)

            new_prob_ = torch.logsumexp(stacked, dim=2)
            assert torch.all(new_prob_ <= 0.001)

            # Now do the final logsumexp on the diagonal
            joining_wds = wd_ids[joining_idxes]
            # print(wd_ids, joining_idxes, torch.where(joining_idxes), joining_wds)
            probabilities_vector = new_prob_[joining_idxes, :]

            if len(probabilities_vector) > 0:
                assert reduce  # think don't want division here because of this
                final_probabilities[joining_wds] = torch.logsumexp(probabilities_vector,
                                                                       dim=1)  # / probabilities_vector_mml.shape[1]
                if torch.all(final_probabilities != 1.0):
                    break

            new_prob = torch.ones(num_words, new_prob_.shape[1], device=probs_l2r.device)
            new_prob[wd_ids[not_joining_idxes], :] = new_prob_[not_joining_idxes,
                                                     :]  # only send on the ones that are not joining

        # Now, prepare stuff to send on
        # Only do this if diag is NOT  a join, which can be different for different elements in the batch
        assert order_prob is not None, 'not sure why this was orignially coded with a chance for order prob to be none'
        left_wd = left_wd_prob[not_joining_idxes, :, :].squeeze(2)
        left_or = order_prob[not_joining_idxes, :, 0]
        rt_wd = rt_wd_prob[not_joining_idxes, :, :].squeeze(2)
        rt_or = order_prob[not_joining_idxes, :, 1]

        to_pass_on_left = left_wd + left_or
        to_pass_on_rt = rt_wd + rt_or

    # Final computation
    assert torch.all(final_probabilities <= 0.001)
    assert reduce  # if reduce, have to SUM (not mean)
    average_probability = -final_probabilities.sum()

    return average_probability if not return_each_prob else final_probabilities



def getOrderingStats(orderings):
    num_fully_left, num_fully_right, num_neither = 0, 0, 0
    num_left_moves, num_right_moves = [], []

    for ordering in orderings:
        length = len(ordering) - 1
        if ordering[-1] == 'L' and ordering[:-1] == list(range(1, length + 1, 1)):
            num_fully_left += 1
        elif ordering[-1] == 'R' and ordering[:-1] == list(range(length, 0, -1)):
            num_fully_right += 1
        else:
            num_neither += 1

        nl = ordering.index(length)
        if ordering[-1] == 'L': nl += 1
        num_left_moves += [nl / length]
        num_right_moves += [(length - nl) / length]
    assert num_fully_left + num_fully_right + num_neither == len(orderings)
    N = len(orderings)
    assert len(num_left_moves) == N and len(num_right_moves) == N

    return (num_fully_left, num_fully_right, num_neither, sum(num_left_moves), sum(num_right_moves))



def getDirections(exhere):
    directions_ = exhere[0]['directions'].tolist()
    directions = [x for x in directions_]
    assert directions[-1] in {-1, -2}
    directions[-1] = 'L' if directions[-1] == -1 else 'R'

    dir_string = str(directions)
    for jj in range(len(exhere)):
        directions_ = exhere[jj]['directions'].tolist()
        directions_jj = [x for x in directions_]
        assert directions_jj[-1] in {-1, -2}
        directions_jj[-1] = 'L' if directions_jj[-1] == -1 else 'R'
        dir_string += (';' + str(directions_jj))
    return directions, dir_string
