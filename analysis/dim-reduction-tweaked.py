## Here we take the embeddings one by one and check for dimensionality with other embeddings.
# not sure if the approach before had any bug

import csv
import os
import sys
import argparse
import pandas as pd
import numpy as np
from emmaemb.core import Emma
sys.path.append('/home/unix/vkrhk/EmmaEmb/analysis')
from dim_reduction_utils import plot_scatter1, run_PCA, plot_scatter, prepare_data, get_silhouette_score, run_tSNE, run_UMAP, run_PHATE

IMG_OUTPUT_PATH = '/home/unix/vkrhk/EmmaEmb/img'
DATA_PATH = '/home/unix/vkrhk/EmmaEmb/data'
EMBEDDINGS_PATH = '/media/drive2/vkrhk/embeddings'
SUBSET_SIZE = 50 # take first N proteins from each dataset for testing

emb_spaces = [
# ("ESM2", f"{EMBEDDINGS_PATH}/esm2_t33_650M_UR50D/layer_33/chopped_1022_overlap_300"),
# ("ANKH", f"{EMBEDDINGS_PATH}/ankh_base/layer_None/chopped_1022_overlap_300"),
# ("ProstT5", f"{EMBEDDINGS_PATH}/Rostlab/ProstT5/layer_None/chopped_1022_overlap_300/"),
# ("ProtT5", f"{EMBEDDINGS_PATH}/Rostlab/prot_t5_xl_uniref50/layer_None/chopped_1022_overlap_300"),
("ESM1", f"{EMBEDDINGS_PATH}/esm1_t34_670M_UR100/layer_34/chopped_1022_overlap_300/"),
# ("ESMC", f"{EMBEDDINGS_PATH}/esmc-300m-2024-12/layer_None/chopped_1022_overlap_300/"),
]

def main(reduction_algorithm, n_components, plot_version):
    # colect data:
    feature_data = []
    embeddings = {}

    for dataset in ['train.txt', 'scPDB_enhanced_binding_sites_translated.csv']:
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
                
                BINDING_FLAG = 'CRYPTIC-BINDING' if dataset == 'train.txt' else 'BINDING'
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

    print(f'Running {reduction_algorithm} for {embeddings_name} ...')

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
        
        if plot_version == 0:
            plot_scatter(embeddings, labels, x_idx=best_idx[0], y_idx=best_idx[1], emb_space=embeddings_name, method=reduction_algorithm, \
                path=f'{IMG_OUTPUT_PATH}/tweaked,plot0:{reduction_algorithm},n_components={n_components},best,N={SUBSET_SIZE}/{embeddings_name}_{reduction_algorithm}_best_pair.png')
        else:
            plot_scatter1(embeddings, labels, x_idx=best_idx[0], y_idx=best_idx[1], emb_space=embeddings_name, method=reduction_algorithm, \
                path=f'{IMG_OUTPUT_PATH}/tweaked,plot1:{reduction_algorithm},n_components={n_components},best,N={SUBSET_SIZE}/{embeddings_name}_{reduction_algorithm}_best_pair.png')


if __name__ == "__main__":
    argparser = argparse.ArgumentParser(description='Run dimensionality reduction and plot results.')
    argparser.add_argument('--reduction_algorithm', type=str, default='PCA', help='The reduction algorithm to use (e.g., PCA, t-SNE, UMAP, PHATE)')
    argparser.add_argument('--n_components', type=int, default=15, help='Number of principal components to compute')
    argparser.add_argument('--plot', type=int, default=0, help='Which plotting function to use (0 or 1)')
    args = argparser.parse_args()
    main(args.reduction_algorithm, args.n_components, args.plot)