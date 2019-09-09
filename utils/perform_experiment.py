from __future__ import print_function

import torch

import math

import time
# -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=

# ======================================================================================================================
def experiment_vae(args, train_loader, val_loader, test_loader, model, optimizer, dir, log_dir, model_name='vae'):
    from utils.training import train_vae as train
    from utils.evaluation import evaluate_vae as evaluate

    # SAVING
    torch.save(args, dir + args.model_name + '.config')

    # best_model = model
    best_ndcg = 0.
    e = 0
    last_epoch = 0

    train_loss_history = []
    train_re_history = []
    train_kl_history = []

    val_loss_history = []
    val_re_history = []
    val_kl_history = []

    val_ndcg_history = []

    time_history = []

    for epoch in range(1, args.epochs + 1):
        time_start = time.time()
        model, train_loss_epoch, train_re_epoch, train_kl_epoch = train(epoch, args, train_loader, model,
                                                                             optimizer)

        val_loss_epoch, val_re_epoch, val_kl_epoch, val_ndcg_epoch = evaluate(args, model, train_loader, val_loader, epoch, dir, mode='validation')
        time_end = time.time()

        time_elapsed = time_end - time_start

        # appending history
        train_loss_history.append(train_loss_epoch), train_re_history.append(train_re_epoch), train_kl_history.append(
            train_kl_epoch)
        val_loss_history.append(val_loss_epoch), val_re_history.append(val_re_epoch), val_kl_history.append(
            val_kl_epoch), val_ndcg_history.append(val_ndcg_epoch)
        time_history.append(time_elapsed)

        # printing results
        print('Epoch: {}/{}, Time elapsed: {:.2f}s\n'
              '* Train loss: {:.2f}   (RE: {:.2f}, KL: {:.2f})\n'
              'o Val.  loss: {:.2f}   (RE: {:.2f}, KL: {:.2f}, NDCG: {:.5f})\n'
              '--> Early stopping: {}/{} (BEST: {:.5f})\n'.format(
            epoch, args.epochs, time_elapsed,
            train_loss_epoch, train_re_epoch, train_kl_epoch,
            val_loss_epoch, val_re_epoch, val_kl_epoch, val_ndcg_epoch,
            e, args.early_stopping_epochs, best_ndcg
        ))

        # early-stopping
        last_epoch = epoch
        if val_ndcg_epoch > best_ndcg:
            e = 0
            best_ndcg = val_ndcg_epoch
            # best_model = model
            print('->model saved<-')
            torch.save(model, dir + args.model_name + '.model')
        else:
            e += 1
            if epoch < args.warmup:
                e = 0
            if e > args.early_stopping_epochs:
                break

        # NaN
        if math.isnan(val_loss_epoch):
            break

    # FINAL EVALUATION
    best_model = torch.load(dir + args.model_name + '.model')
    test_loss, test_re, test_kl, test_ndcg, \
    eval_ndcg20, eval_ndcg10, eval_recall50, eval_recall20, \
    eval_recall10, eval_recall5, eval_recall1 = evaluate(args, best_model, train_loader, test_loader, 9999, dir, mode='test')

    print("NOTE: " + args.note)
    print('FINAL EVALUATION ON TEST SET\n'
          '- BEST VALIDATION NDCG: {:.5f} ({:} epochs) -\n'
          'NDCG@100: {:}  |  Loss: {:.2f}\n'
          'NDCG@20: {:}   |  RE: {:.2f}\n'
          'NDCG@10: {:}   |  KL: {:.2f}\n'
          'Recall@50: {:} |  Recall@5: {:}\n'
          'Recall@20: {:} |  Recall@1: {:}\n'
          'Recall@10: {:}'.format(
        best_ndcg, last_epoch,
        test_ndcg, test_loss,
        eval_ndcg20, test_re,
        eval_ndcg10, test_kl,
        eval_recall50, eval_recall5,
        eval_recall20, eval_recall1,
        eval_recall10
    ))
    print('-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-')

    if not args.no_log:
        with open(log_dir, 'a') as f:
            print(args, file=f)
            print("NOTE: " + args.note, file=f)
            print('FINAL EVALUATION ON TEST SET\n'
                  '- BEST VALIDATION NDCG: {:.5f} ({:} epochs) -\n'
                  'NDCG@100: {:}  |  Loss: {:.2f}\n'
                  'NDCG@20: {:}   |  RE: {:.2f}\n'
                  'NDCG@10: {:}   |  KL: {:.2f}\n'
                  'Recall@50: {:} |  Recall@5: {:}\n'
                  'Recall@20: {:} |  Recall@1: {:}\n'
                  'Recall@10: {:}'.format(
                best_ndcg, last_epoch,
                test_ndcg, test_loss,
                eval_ndcg20, test_re,
                eval_ndcg10, test_kl,
                eval_recall50, eval_recall5,
                eval_recall20, eval_recall1,
                eval_recall10
            ), file=f)
            print('-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-\n', file=f)

    # SAVING
    torch.save(train_loss_history, dir + args.model_name + '.train_loss')
    torch.save(train_re_history, dir + args.model_name + '.train_re')
    torch.save(train_kl_history, dir + args.model_name + '.train_kl')
    torch.save(val_loss_history, dir + args.model_name + '.val_loss')
    torch.save(val_re_history, dir + args.model_name + '.val_re')
    torch.save(val_kl_history, dir + args.model_name + '.val_kl')
    torch.save(val_ndcg_history, dir +args.model_name + '.val_ndcg')
    torch.save(test_loss, dir + args.model_name + '.test_loss')
    torch.save(test_re, dir + args.model_name + '.test_re')
    torch.save(test_kl, dir + args.model_name + '.test_kl')
    torch.save(test_ndcg, dir +args.model_name + '.test_ndcg')
