## Here we take the embeddings one by one and check for dimensionality with other embeddings.
# not sure if the approach before had any bug

import csv
import os
import sys
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from emmaemb.core import Emma
sys.path.append('/home/unix/vkrhk/EmmaEmb/analysis')
from dim_reduction_utils import plot_kde, run_PCA, plot_scatter, prepare_data, get_silhouette_score, run_tSNE, run_UMAP, run_PHATE

IMG_OUTPUT_PATH = '/home/unix/vkrhk/EmmaEmb/img'
DATA_PATH = '/home/unix/vkrhk/EmmaEmb/data'
EMBEDDINGS_PATH = '/media/drive2/vkrhk/embeddings'
SUBSET_SIZE = 50 # take first N proteins from each dataset for testing

emb_spaces = [("Evo2", "/media/drive2/vkrhk/embeddings/genes")]


def plot_scatter1(embeddings, labels, x_idx, y_idx, emb_space, method, path=None):
    if path is not None and not os.path.exists(os.path.dirname(path)):
        os.makedirs(os.path.dirname(path))
    
    pc_x = embeddings[:, x_idx]
    pc_y = embeddings[:, y_idx]
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    x_padding = (pc_x.max() - pc_x.min()) * 0.1
    y_padding = (pc_y.max() - pc_y.min()) * 0.1

    # Plot for Non-binding (labels == 0)
    ax = axes[0]
    ax.scatter(pc_x[labels == 0], pc_y[labels == 0], alpha=0.2, color='blue', s=5, label='Non-binding')
    ax.set_xlim(pc_x.min() - x_padding, pc_x.max() + x_padding)
    ax.set_ylim(pc_y.min() - y_padding, pc_y.max() + y_padding)
    ax.set_xlabel(f'PC{x_idx + 1}')
    ax.set_ylabel(f'PC{y_idx + 1}')
    ax.set_title('Non-binding')
    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.5)

    # Plot for Binding (labels == 1)
    ax = axes[1]
    ax.scatter(pc_x[labels == 1], pc_y[labels == 1], alpha=0.2, color='red', s=5, label='Cryptic Binding')
    ax.set_xlim(pc_x.min() - x_padding, pc_x.max() + x_padding)
    ax.set_ylim(pc_y.min() - y_padding, pc_y.max() + y_padding)
    ax.set_xlabel(f'PC{x_idx + 1}')
    ax.set_ylabel(f'PC{y_idx + 1}')
    ax.set_title('Cryptic Binding')
    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.5)

    fig.suptitle(f'Silhouette Score: {get_silhouette_score(embeddings[:, [x_idx, y_idx]], labels):.4f}\n{emb_space} - {method}', fontsize=12)

    plt.tight_layout()
    if path:
        plt.savefig(path)
    plt.show()


def main(reduction_algorithm, n_components):
    # colect data:
    feature_data = []
    embeddings = {}

    for dataset in ['cryptobench-with-sequence.csv']:
        with open(f'{DATA_PATH}/{dataset}', 'r') as f:
            reader = csv.reader(f, delimiter=';')
            for ii, row in enumerate(reader):
                protein_id = row[0] + row[1]
                annotation = row[3].split(' ')
                annotation = [int(i[1:]) for i in annotation]
                sequence = row[4]

                these_embeddings = {}
                length, have_same_length = 0, True
                for i, (embeddings_name, embeddings_path) in enumerate(emb_spaces):
                    path = f"{embeddings_path}/{protein_id}.npy"
                    if not os.path.exists(path):
                        have_same_length = False
                        break
                    embedding = np.load(path)
                    these_embeddings[embeddings_name] = embedding
                    # check that all embedding spaces have the same length for this protein
                    if i == 0:
                        length = embedding.shape[0]
                    else:
                        if embedding.shape[0] != length:
                            have_same_length = False
                            break

                if not have_same_length:
                    continue
                
                BINDING_FLAG = 'BINDING'
                for i in range(len(sequence)):
                    feature_data.append([sequence[i], BINDING_FLAG if i in annotation else 'NON-BINDING'])

                for embeddings_name in these_embeddings:
                    if embeddings_name not in embeddings:
                        embeddings[embeddings_name] = []
                    embeddings[embeddings_name].append(these_embeddings[embeddings_name])
                if ii > SUBSET_SIZE:
                    break
                
    for embeddings_name in embeddings:
        concatenated_embeddings_path = f"{DATA_PATH}/concatenated-embeddings/{embeddings_name}_binding_site_embeddings.npy"
        embeddings[embeddings_name] = np.concatenate(embeddings[embeddings_name], axis=0)
        np.save(concatenated_embeddings_path, embeddings[embeddings_name])  

    feature_data = pd.DataFrame.from_records(feature_data, columns=["amino acid", "binding_site"])

    # initiate Emma object and load embedding spaces
    emma = Emma(feature_data=feature_data)
    for embeddings_name, _ in emb_spaces:
        emma.add_emb_space(
            embeddings_source=f"{DATA_PATH}/concatenated-embeddings/{embeddings_name}_binding_site_embeddings.npy",
            emb_space_name=embeddings_name)

    for embeddings_name, _ in emb_spaces:
        embeddings, labels = prepare_data(emma, embeddings_name)
        if reduction_algorithm == 'PCA':
            embeddings, _ = run_PCA(n_components=n_components, embeddings=embeddings)
        elif reduction_algorithm == 't-SNE':
            embeddings = run_tSNE(n_components=n_components, embeddings=embeddings)
        elif reduction_algorithm == 'UMAP':
            embeddings = run_UMAP(n_components=n_components, embeddings=embeddings)
        elif reduction_algorithm == 'PHATE':
            embeddings = run_PHATE(n_components=n_components, embeddings=embeddings)
        else:
            raise ValueError(f"Unsupported reduction algorithm: {reduction_algorithm}")

        print(f'Running {reduction_algorithm} for {embeddings_name} ...')
        
        best_idx = (0, 1)
        best_score = -1
        for i in range(n_components):
            for ii in range(i + 1, n_components):
                print(f"\tEvaluating PC pair: ({i}, {ii})")
                score = get_silhouette_score(embeddings[:, [i, ii]], labels)
                if score > best_score:
                    best_score = score
                    best_idx = (i, ii)
        print(f"Best PC pair: {best_idx} with Silhouette Score: {best_score:.4f}")
        print()
        
        plot_scatter1(embeddings, labels, x_idx=best_idx[0], y_idx=best_idx[1], emb_space=embeddings_name, method=reduction_algorithm, \
            path=f'{IMG_OUTPUT_PATH}/tweaked,plot1:{reduction_algorithm},n_components={n_components},best,N={SUBSET_SIZE}/{embeddings_name}_{reduction_algorithm}_best_pair.png')


if __name__ == "__main__":
    argparser = argparse.ArgumentParser(description='Run dimensionality reduction and plot results.')
    argparser.add_argument('--reduction_algorithm', type=str, default='PCA', help='The reduction algorithm to use (e.g., PCA, t-SNE, UMAP, PHATE)')
    argparser.add_argument('--n_components', type=int, default=15, help='Number of principal components to compute')
    args = argparser.parse_args()
    main(args.reduction_algorithm, args.n_components)