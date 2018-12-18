# coding: utf-8

import warnings

import chainer
import chainer.functions as F


def zeroth_only(model, properties, **kwargs):
    assert model.order >= 0
    if model.order >= 1:
        warnings.warn('In this loss function `zeroth_only`,'
                      ' calculated differential value is discarded.')
    if kwargs:
        warnings.warn('In this loss function, `zeroth_only`'
                      ' keyword arguments are ignored.')
    observation_keys = [f'RMSE/{properties[0]}', 'RMSE/total']

    def loss_function(inputs, labels, train=True):
        predictions = model.predict(inputs, train=train)
        pred0, *_ = predictions
        true0, *_ = labels
        loss0 = F.mean_squared_error(pred0, true0)
        RMSE0 = F.sqrt(loss0) / len(model)

        observation = {
            observation_keys[0]: RMSE0,
            observation_keys[1]: RMSE0,
            }
        chainer.report(observation, observer=model)
        return loss0

    return loss_function, observation_keys


def first_only(model, properties, **kwargs):
    assert model.order >= 1
    if kwargs:
        warnings.warn('In this loss function `first_only`,'
                      ' keyword arguments are ignored.')
    observation_keys = [f'RMSE/{properties[1]}', 'RMSE/total']

    def loss_function(inputs, labels, train=True):
        predictions = model.predict(inputs, train=train)
        _, pred1, *_ = predictions
        _, true1, *_ = labels
        loss1 = F.mean_squared_error(pred1, true1)
        RMSE1 = F.sqrt(loss1)

        observation = {
            observation_keys[0]: RMSE1,
            observation_keys[1]: RMSE1,
            }
        chainer.report(observation, observer=model)
        return loss1

    return loss_function, observation_keys


def mix(model, properties, **kwargs):
    assert model.order >= 1
    mixing_beta = kwargs['mixing_beta']
    observation_keys = [f'RMSE/{properties[0]}', f'RMSE/{properties[1]}',
                        'RMSE/total']

    def loss_function(inputs, labels, train=True):
        predictions = model.predict(inputs, train=train)
        pred0, pred1, *_ = predictions
        true0, true1, *_ = labels
        loss0 = F.mean_squared_error(pred0, true0)
        loss1 = F.mean_squared_error(pred1, true1)
        total_loss = ((1.0 - mixing_beta) * loss0
                      + mixing_beta * loss1)

        RMSE0 = F.sqrt(loss0) / len(model)
        RMSE1 = F.sqrt(loss1)
        total_RMSE = ((1.0 - mixing_beta) * RMSE0
                      + mixing_beta * RMSE1)

        observation = {
            observation_keys[0]: RMSE0,
            observation_keys[1]: RMSE1,
            observation_keys[2]: total_RMSE,
            }
        chainer.report(observation, observer=model)
        return total_loss

    return loss_function, observation_keys
