import ray
import logging

import numpy as np
import wandb


train_logger = logging.getLogger('train')
test_logger = logging.getLogger('train_test')



def _log(config, step_count, log_data, model, replay_buffer, lr, shared_storage, vis_result, timing_data=None):
    loss_data, td_data, priority_data = log_data
    total_loss, weighted_loss, loss, reg_loss, policy_loss, value_prefix_loss, value_loss, consistency_loss, \
        grad_norm, policy_entropy, target_policy_entropy = loss_data
    if vis_result:
        new_priority, target_value_prefix, target_value, trans_target_value_prefix, trans_target_value, target_value_prefix_phi, target_value_phi, \
        pred_value_prefix, pred_value, target_policies, predicted_policies, state_lst, other_loss, other_log, other_dist = td_data
        batch_weights, batch_indices = priority_data

    replay_episodes_collected, replay_buffer_size, priorities, total_num, worker_logs = ray.get([
        replay_buffer.episodes_collected.remote(), replay_buffer.size.remote(),
        replay_buffer.get_priorities.remote(), replay_buffer.get_total_len.remote(),
        shared_storage.get_worker_logs.remote()])

    worker_ori_reward, worker_reward, worker_reward_max, worker_eps_len, worker_eps_len_max, test_counter, test_dict, temperature, visit_entropy, priority_self_play, distributions = worker_logs

    _msg = '#{:<10} Total Loss: {:<8.3f} [weighted Loss:{:<8.3f} Policy Loss: {:<8.3f} Value Loss: {:<8.3f} ' \
           'Reward Sum Loss: {:<8.3f} Consistency Loss: {:<8.3f} ] ' \
           'Replay Episodes Collected: {:<10d} Buffer Size: {:<10d} Transition Number: {:<8.3f}k ' \
           'Batch Size: {:<10d} Lr: {:<8.3f}'
    _msg = _msg.format(step_count, total_loss, weighted_loss, policy_loss, value_loss, value_prefix_loss, consistency_loss,
                       replay_episodes_collected, replay_buffer_size, total_num / 1000, config.batch_size, lr)
    train_logger.info(_msg)

    if test_dict is not None:
        mean_score = np.mean(test_dict['mean_score'])
        max_score = np.mean(test_dict['max_score'])
        min_score = np.mean(test_dict['min_score'])
        std_score = np.mean(test_dict['std_score'])
        test_msg = '#{:<10} Test Mean Score of {}: {:<10} (max: {:<10}, min:{:<10}, std: {:<10})' \
                   ''.format(test_counter, config.env_name, mean_score, max_score, min_score, std_score)
        test_logger.info(test_msg)

    # Build a single log dict for wandb
    log_dict = {}

    tag = 'Train'
    if vis_result:
        target_value_prefix = target_value_prefix.flatten()
        pred_value_prefix = pred_value_prefix.flatten()
        target_value = target_value.flatten()
        pred_value = pred_value.flatten()
        new_priority = new_priority.flatten()

        log_dict['{}_statistics/new_priority_mean'.format(tag)] = new_priority.mean()
        log_dict['{}_statistics/new_priority_std'.format(tag)] = new_priority.std()
        log_dict['{}_statistics/target_value_prefix_mean'.format(tag)] = target_value_prefix.mean()
        log_dict['{}_statistics/target_value_prefix_std'.format(tag)] = target_value_prefix.std()
        log_dict['{}_statistics/pre_value_prefix_mean'.format(tag)] = pred_value_prefix.mean()
        log_dict['{}_statistics/pre_value_prefix_std'.format(tag)] = pred_value_prefix.std()
        log_dict['{}_statistics/target_value_mean'.format(tag)] = target_value.mean()
        log_dict['{}_statistics/target_value_std'.format(tag)] = target_value.std()
        log_dict['{}_statistics/pre_value_mean'.format(tag)] = pred_value.mean()
        log_dict['{}_statistics/pre_value_std'.format(tag)] = pred_value.std()

        for key, val in other_loss.items():
            if val >= 0:
                log_dict['{}_metric/{}'.format(tag, key)] = val

        for key, val in other_log.items():
            log_dict['{}_weight/{}'.format(tag, key)] = val

    # Core training metrics
    log_dict['{}/total_loss'.format(tag)] = total_loss
    log_dict['{}/loss'.format(tag)] = loss
    log_dict['{}/weighted_loss'.format(tag)] = weighted_loss
    log_dict['{}/reg_loss'.format(tag)] = reg_loss
    log_dict['{}/policy_loss'.format(tag)] = policy_loss
    log_dict['{}/value_loss'.format(tag)] = value_loss
    log_dict['{}/value_prefix_loss'.format(tag)] = value_prefix_loss
    log_dict['{}/consistency_loss'.format(tag)] = consistency_loss
    log_dict['{}/episodes_collected'.format(tag)] = replay_episodes_collected
    log_dict['{}/replay_buffer_len'.format(tag)] = replay_buffer_size
    log_dict['{}/total_node_num'.format(tag)] = total_num
    log_dict['{}/lr'.format(tag)] = lr

    # New metrics
    log_dict['{}/grad_norm'.format(tag)] = grad_norm
    log_dict['{}/policy_entropy'.format(tag)] = policy_entropy
    log_dict['{}/target_policy_entropy'.format(tag)] = target_policy_entropy

    # Worker metrics
    if worker_reward is not None:
        log_dict['workers/ori_reward'] = worker_ori_reward
        log_dict['workers/clip_reward'] = worker_reward
        log_dict['workers/clip_reward_max'] = worker_reward_max
        log_dict['workers/eps_len'] = worker_eps_len
        log_dict['workers/eps_len_max'] = worker_eps_len_max
        log_dict['workers/temperature'] = temperature
        log_dict['workers/visit_entropy'] = visit_entropy
        log_dict['workers/priority_self_play'] = priority_self_play
    # Test metrics
    if test_dict is not None:
        for key, val in test_dict.items():
            log_dict['train/{}'.format(key)] = np.mean(val)
        log_dict['train/test_counter'] = test_counter

    # Timing data from main training loop
    if timing_data is not None:
        for key, val in timing_data.items():
            log_dict['timing/{}'.format(key)] = val

    # Timing data from workers
    worker_timing = ray.get(shared_storage.get_worker_timing.remote())
    if worker_timing:
        for key, val in worker_timing.items():
            log_dict['timing/{}'.format(key)] = val

    wandb.log(log_dict, step=step_count)
