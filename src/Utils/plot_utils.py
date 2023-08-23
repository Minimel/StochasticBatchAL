"""
Author: Mélanie Gaillochet
"""
import os
import random

import numpy as np
from numpy.testing import assert_almost_equal
import matplotlib.pyplot as plt

import torch
import pytorch_lightning as pl

import networkx as nx

from skimage.measure import find_contours
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE


def plot_edge_weights(img, start_edges, end_edges, edge_weights, savepath=''):
    """We plot the image and the edges between neighbouring nodes (weighted by exponent of intensity difference)
    
        Args:
        img (array of shape [H x W]): 
        start_edges (array of shape [n_edges]): 
        end_edges (array of shape [n_edges]):
        edge_weights (array of shape [n_edges]):
        savepath (str, optional): path where to save image (should end with *.png), if we want to save it
    """
    flat_img = img.flatten()
    N = len(flat_img)

    _indice_array = np.array(range(1, N+1)).reshape(img.shape)

    all_nodes = (_indice_array - 1).flatten().tolist()
    all_edges = [[start.item(), end.item()] for (start, end) in zip(start_edges, end_edges)]

    # We get the position of every node (identified by an index from 0 to N-pixels)
    pos = {}
    H_max = _indice_array.shape[0] - 1
    W_max = _indice_array.shape[1] - 1
    for i in all_nodes:
        y = np.where(_indice_array - 1== i)[0].item()   # -1 because _indice_array starts at 1
        x = np.where(_indice_array - 1== i)[1].item()
        pos[i] = (x, y)

    # We create the graph
    G = nx.Graph()
    G.add_nodes_from(all_nodes)
    G.add_edges_from(all_edges)

    fig = plt.figure(figsize=(20,20), layout="constrained")
    ax  = fig.add_subplot(111)
    ax.imshow(img, alpha=0.2)

    nx.draw_networkx_nodes(G,pos,
                        node_size=0,
                        node_color='black',
                        alpha=0.7)
    nx.draw_networkx_edges(G,pos,
                        width=edge_weights,
                        edge_color='black',
                        alpha=0.6)

    if savepath != '':
        savepath = os.path.join(savepath)
        fig.savefig(savepath)

    return plt


def plot_data_pred_volume(data, target, logits, slice, plot_type='contour', title='', vmin=0, vmax=None):
    """ We plot the data, target and pred of a volume at a given slice

    Args:
        data (tensor of shape [BS, num_slices in volume, H,  W]): 
        target (tensor of shape [BS, num_slices in volume, H,  W]): 
        logits (tensor of shape [BS, num_slices in volume, H,  W]): 
        plot_ype (str, optional): 'contour' or 'image'
        title (str, optional): Indice of volume we plot
    """

    fig = plt.figure(figsize=(10, 10), layout="constrained")
    nrows, ncols = 1, logits.shape[0] - 1

    cur_data = data[0, slice, :, :].detach().cpu().numpy()
    cur_target = target[0, slice, :, :].detach().cpu().numpy()
    cur_pred = torch.argmax(logits[:, slice, :, :], dim=0).detach().cpu().numpy()

    if plot_type == 'contour':
        ax = _plot_contour(fig, 1, 1, cur_data, cur_target, cur_pred)
    elif plot_type == 'image':
        ax = _plot_image(fig, 1, 3, cur_data, cur_target, cur_pred, vmin=vmin, vmax=vmax)
    ax.set_title(title, fontsize=12)
    
    return plt


def plot_data_pred(data, target, logits=None, plot_type='contour', title='', vmin=0, vmax=None):
    """ We plot the data, target and pred, either as 1 subplot (if 'contour') or 3 subplots (if 'image't)
    Args:
        data (tensor of shape [H x W]): 
        target (tensor of shape [H x W]): 
        logits (tensor of shape [C, H x W]): 
        plot_ype (str, optional): 'contour' or 'image'
        title (str, optional): Indice of volume we plot
    """

    fig = plt.figure(figsize=(10, 10), layout="constrained")
    nrows, ncols = 1, len(torch.unique(target)) - 1

    cur_data = data.numpy()
    cur_target = target.numpy()
    cur_pred = torch.argmax(logits[:, :, :], dim=0).numpy() if logits is not None else None

    if plot_type == 'contour':
        ax = _plot_contour(fig, 1, 1, cur_data, cur_target, cur_pred)
    elif plot_type == 'image':
        ax = _plot_image(fig, 1, 3, cur_data, cur_target, cur_pred, vmin=vmin, vmax=vmax)
    ax.set_title(title, fontsize=12)

    return plt


def _plot_contour(fig, nrows, ncols, data, target, pred=None):
    """
    Function to generate 1 subplot: image overlapped with contour of target (blue) and pred (red)
    """
    # Computing the Active Contour for the given image
    contour_target = find_contours(target.T, 0.5)

    ## Image
    ax = fig.add_subplot(nrows, ncols, 1)
    ax.imshow(data, 'gray')
    plt.axis('off')
    
    # Contours of target and pred
    for contour in contour_target:
        ax.plot(contour[:, 0], contour[:, 1], '-b', lw=5)
        plt.axis('off')

    # We also plot prediction contour if it is available
    if pred is not None:
        contour_pred = find_contours(pred.T, 0.5)
        for contour in contour_pred:
            ax.plot(contour[:, 0], contour[:, 1], '-r', lw=5)
            plt.axis('off')
    return ax


def _plot_image(fig, nrows, ncols, data, target, pred=None, vmin=0, vmax=None):
    """
    Function to generate 3 subplots: image, target and pred
    """
    # Image
    i = 1
    ax = fig.add_subplot(nrows, ncols, i)
    ax.imshow(data, 'gray')
    plt.axis('off')

    # Target
    i = 2
    ax = fig.add_subplot(nrows, ncols, i)
    ax.imshow(data, 'gray', interpolation='none')
    plt.axis('off')
    ax.imshow(target, cmap='viridis', alpha=0.7, vmin=vmin, vmax=vmax)
    plt.axis('off')

    # Prediction
    i = 3
    ax = fig.add_subplot(nrows, ncols, i)
    ax.imshow(data, 'gray', interpolation='none')
    plt.axis('off')
    ax.imshow(pred, cmap='viridis', alpha=0.7, vmin=vmin, vmax=vmax)
    plt.axis('off')

    return ax


def plot_uncertainty_image(cur_data, cur_target, cur_pred, cur_uncertainty_map=None, img_indice='', plot_type='contour', vmin=0, vmax=None):
    """
    Function to plot image, target, pred and associated uncertainty 
    (contour if 2 output channels, otherwise separate target and pred)
    Args:
        cur_data (array of shape [H x W]):
        cur_target (array of shape [H x W]):
        cur_pred (array of shape [H x W]):
        cur_uncertainty_map (array of shape [H x W]):
        img_indice (int, optional): indice of image we plot
        plot_type (str, optional): 'contour' or 'image'
    """
    # If 2 output channels, we show the contour
    if plot_type == 'contour':
        fig = plt.figure(figsize=(10, 5), layout="constrained")
        nrows = 1
        ncols = 1 if cur_uncertainty_map is None else 2
        i = 1

        # Computing the Active Contour for the given image
        contour_target = find_contours(cur_target.T, 0.5)

        ## Image
        ax = fig.add_subplot(nrows, ncols, i)
        ax.imshow(cur_data, 'gray')
        plt.axis('off')
        
        # Contours of target and pred
        for contour in contour_target:
            ax.plot(contour[:, 0], contour[:, 1], '-b', lw=5)
            plt.axis('off')

        # We also plot prediction contour if it is available
        contour_pred = find_contours(cur_pred.T, 0.5)
        for contour in contour_pred:
            ax.plot(contour[:, 0], contour[:, 1], '-r', lw=5)
            plt.axis('off')

        ax.set_title('Idx {}'.format(img_indice), fontsize=12)
        i += 1

    # Otherwise, we show individually target and prediction
    elif plot_type == 'image':
        fig = plt.figure(figsize=(15, 5), layout="constrained")
        nrows = 1
        ncols = 2 if cur_uncertainty_map is None else 3
        i = 1

        ax = fig.add_subplot(nrows, ncols, i)
        ax.imshow(cur_data, 'gray', interpolation='none')
        plt.axis('off')
        ax.imshow(cur_target, cmap='viridis', alpha=0.6, vmin=vmin, vmax=vmax)
        plt.axis('off')
        ax.set_title('Idx {}'.format(img_indice), fontsize=12)
        i += 1

        ax = fig.add_subplot(nrows, ncols, i)
        ax.imshow(cur_data, 'gray', interpolation='none')
        plt.axis('off')
        ax.imshow(cur_pred, cmap='viridis', alpha=0.6, vmin=vmin, vmax=vmax)
        plt.axis('off')
        ax.set_title('Pred', fontsize=12)
        i += 1

    if cur_uncertainty_map is not None:
        ax = fig.add_subplot(nrows, ncols, i)
        ax.imshow(cur_uncertainty_map, cmap='viridis', vmin=0)#, vmax=np.log2(out_channels))
        plt.axis('off')
        ax.set_title('Uncertainty: {:.5f}'.format(np.mean(cur_uncertainty_map), fontsize=12))

    return plt


def plot_multiple_data_pred(cur_multiple_data, cur_multiple_pred, query_indice):
    """
    We plot the multiple images and their associated predictions
    """
    # We plot the multiple images and their associated predictions
    fig = plt.figure(figsize=(15, 5), layout="constrained")
    nrows = 4
    ncols = int(np.ceil(len(cur_multiple_data) / 2))
    for i in range(len(cur_multiple_data)):
        ax = fig.add_subplot(nrows, ncols, i + 1)
        ax.imshow(cur_multiple_data[i], 'gray', interpolation='none')
        plt.axis('off')
        if i == 0:
            ax.set_title('Idx {}'.format(query_indice), fontsize=12)
    for i in range(len(cur_multiple_pred)):
        ax = fig.add_subplot(nrows, ncols, 2 * ncols + i + 1)
        ax.imshow(cur_multiple_pred[i], 'viridis', interpolation='none')
        plt.axis('off')

    return plt


def plot_all_uncertain_samples_from_lists(indice_list, data_list, logits_list, target_list, uncertainty_list, budget, 
                                          trainer, title, model_out_channels, multiple_prob_list=None, multiple_data_list=None):
    """
    We prepare the data from lists and plot the top uncertainty samples and their uncertainty map
    Args:
        indice_list (list of int):
        data_list (list of tensor of shape [C, H, W]):
        logits_list (list of tensor of shape [C, H, W]):
        target_list (list of tensor of shape [H, W]):
        budget (int): number of samples to plot
        trainer (pytorch_lightning.trainer):
        title (str): plot title
        model_out_channels (int): number of output channels of model (will determine if plottijg contour or images)
        uncertainty_list (list of tensor of shape [H x W]): list of uncertainty maps or list of uncertainty values
        multiple_prob_list (list of tensor of shape [#inferences, C, H, W]):    
        multiple_data_list (list of tensor of shape [#inferences, C, H, W]):
    """
    # We convert lists to arrays
    data_array = np.stack(data_list, axis=0)
    logits_array = np.stack(logits_list, axis=0)
    target_array = np.stack(target_list, axis=0)
    if type(uncertainty_list[0]) == float:  # If list of uncertainty values
        uncertainty_map_array = None
        mean_uncertainty_list = np.array(uncertainty_list)
    else:  # If list of uncertainty maps
        uncertainty_map_array = np.stack(uncertainty_list, axis=0)
        mean_uncertainty_list = np.mean(uncertainty_map_array, axis=(1, 2))

    if  multiple_prob_list is not None:
        multiple_prob_array = np.stack(multiple_prob_list, axis=0)
    if multiple_data_list is not None:
        multiple_data_array = np.stack(multiple_data_list, axis=0)
    
    # We log the querried images and their uncertainty map
    arg = np.argsort(mean_uncertainty_list)
    query_pool_indices = list(torch.tensor(indice_list)[arg][-budget:].numpy())
    uncertainty_values = list(torch.tensor(mean_uncertainty_list)[arg][-budget:].numpy())

    for idx in range(budget):
        query_indice = query_pool_indices[idx]
        unlabeledloader_position = np.where(indice_list == query_indice)[0][0]

        cur_data = data_array[unlabeledloader_position, 0, :, :]  # shape [H, W]
        cur_target = target_array[unlabeledloader_position, 0, :, :]  # shape [H, W]
        cur_pred = np.argmax(logits_array[unlabeledloader_position, :, :, :], axis=0)  # shape [H, W] 
        cur_uncertainty_map = uncertainty_map_array[unlabeledloader_position, :, :] if uncertainty_map_array is not None else None
        if uncertainty_map_array is not None:
            assert_almost_equal(np.mean(cur_uncertainty_map), uncertainty_values[idx], decimal=6)

        if  multiple_prob_list is not None and multiple_data_list is not None:
            cur_multiple_pred = np.argmax(multiple_prob_array[unlabeledloader_position, :, :, :, :], axis=-3)  # shape [#inferences, H, W]
            cur_multiple_data = multiple_data_array[unlabeledloader_position, :, 0, :, :]  # shape [#inferences, H, W]

            # We plot the multiple images and their associated predictions
            plt = plot_multiple_data_pred(cur_multiple_data, cur_multiple_pred, query_indice)
            if isinstance(trainer.logger, pl.loggers.CometLogger):
            # Saving on comet_ml
                trainer.logger.experiment.log_figure(figure=plt, figure_name='Multiple predictions', step=idx)

        # We plot image, target, pred and uncertainty map
        plot_type = 'contour' if model_out_channels == 2 else 'image'
        plt = plot_uncertainty_image(cur_data, cur_target, cur_pred, cur_uncertainty_map, query_indice, plot_type, vmin=0, vmax=model_out_channels - 1)

        # We save the figure on the logger
        if isinstance(trainer.logger, pl.loggers.CometLogger):
            # Saving on comet_ml
            trainer.logger.experiment.log_figure(figure=plt, figure_name=title, step=idx)
        else:
            # Saving on TensorBoardLogger
            trainer.logger.experiment.add_image(title + '_data', cur_data, idx, dataformats="WD")
            trainer.logger.experiment.add_image(title + '_target', cur_target, idx, dataformats="WD")
            trainer.logger.experiment.add_image(title + '_pred', cur_pred, idx, dataformats="WD")


##### TSNE #####

def apply_PCA(array, n_components=0.95):
    pca = PCA(n_components=n_components)
    pca.fit(array)
    principalComponents = pca.transform(array)

    return principalComponents


def apply_tSNE(array):
    """
    :param array: 2D array
    :return:
    """
    tsne = TSNE(n_components=2, verbose=1, random_state=123, init='random', perplexity=30,
                n_iter=1000)
    z = tsne.fit_transform(array)
    return z
