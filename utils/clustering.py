import os
import torch
import wandb
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans, DBSCAN, HDBSCAN
from sklearn.metrics import silhouette_score, calinski_harabasz_score
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
from datetime import datetime
from pathlib import Path
from model.cae import CAE
from utils.dataloader import TopomapDataset
from torch.utils.data import DataLoader
from sklearn.decomposition import PCA


def create_dataloader(data_path, batch_size, num_workers=4):
    """Create dataloader for the topomap dataset"""
    dataset = TopomapDataset(folder_path=data_path)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False, 
        num_workers=num_workers,
        pin_memory=True
    )

def load_model(model_path, device, latent_dim=6):
    """Load the pre-trained CAE model"""
    model = CAE(latent_dim=latent_dim)
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    return model

def extract_latent_spaces(model, dataloader, device):
    """Extract latent space representations from the model"""
    latent_spaces = []
    with torch.no_grad():
        for batch in dataloader:
            inputs = batch.to(device)
            _, latent = model(inputs)
            latent_spaces.append(latent.cpu().numpy())
    return np.vstack(latent_spaces)

def plot_clusters_2d_3d(latent_spaces, labels, title):
    """Create 2D and 3D visualizations of clusters"""
    fig = plt.figure(figsize=(15, 5))
    
    # 2D visualization using PCA
    ax1 = fig.add_subplot(121)
    pca = PCA(n_components=2)
    latent_2d = pca.fit_transform(latent_spaces)
    scatter1 = ax1.scatter(latent_2d[:, 0], latent_2d[:, 1], c=labels, cmap='tab20')
    ax1.set_title(f"{title} (PCA 2D)")
    plt.colorbar(scatter1, ax=ax1)
    
    # 3D visualization using PCA
    if latent_spaces.shape[1] >= 3:
        ax2 = fig.add_subplot(122, projection='3d')
        pca = PCA(n_components=3)
        latent_3d = pca.fit_transform(latent_spaces)
        scatter2 = ax2.scatter(latent_3d[:, 0], latent_3d[:, 1], latent_3d[:, 2], 
                             c=labels, cmap='tab20')
        ax2.set_title(f"{title} (PCA 3D)")
        plt.colorbar(scatter2, ax=ax2)
    
    plt.tight_layout()
    return fig

def run_kmeans_analysis(latent_spaces, k_values):
    """Run K-means clustering analysis"""
    metrics_table = wandb.Table(columns=["k", "silhouette_score", 
                                       "calinski_harabasz_score", "inertia",
                                       "cluster_visualization"])
    
    metrics = []
    for k in k_values:
        kmeans = KMeans(n_clusters=k, random_state=42)
        labels = kmeans.fit_predict(latent_spaces)
        
        # Calculate metrics
        sil_score = silhouette_score(latent_spaces, labels)
        ch_score = calinski_harabasz_score(latent_spaces, labels)
        
        # Create visualization
        fig = plot_clusters_2d_3d(latent_spaces, labels, f"K-Means (k={k})")
        
        # Log to wandb
        metrics_table.add_data(
            k, sil_score, ch_score, kmeans.inertia_,
            wandb.Image(fig)
        )
        plt.close(fig)
        
        metrics.append({
            'k': k,
            'silhouette': sil_score,
            'calinski_harabasz': ch_score,
            'inertia': kmeans.inertia_
        })
        
        # Log metrics individually for plotting
        wandb.log({
            f"kmeans/silhouette": sil_score,
            f"kmeans/calinski_harabasz": ch_score,
            f"kmeans/inertia": kmeans.inertia_
        })
    
    wandb.log({"kmeans_metrics": metrics_table})
    return metrics

def run_hdbscan_analysis(latent_spaces, min_cluster_sizes):
    """Run HDBSCAN clustering analysis"""
    metrics_table = wandb.Table(columns=["min_size", "silhouette_score", 
                                       "n_clusters", "noise_ratio",
                                       "cluster_visualization"])
    
    metrics = []
    for min_size in min_cluster_sizes:
        hdb = HDBSCAN(min_cluster_size=min_size)
        labels = hdb.fit_predict(latent_spaces)
        
        # Calculate metrics for non-noise points
        valid_points = labels != -1
        if np.any(valid_points):
            sil_score = silhouette_score(
                latent_spaces[valid_points],
                labels[valid_points]
            )
            
            # Create visualization
            fig = plot_clusters_2d_3d(latent_spaces, labels, 
                                    f"HDBSCAN (min_size={min_size})")
            
            # Log to wandb
            metrics_table.add_data(
                min_size, sil_score,
                len(np.unique(labels)) - 1,
                np.mean(labels == -1),
                wandb.Image(fig)
            )
            plt.close(fig)
            
            metrics.append({
                'min_size': min_size,
                'silhouette': sil_score,
                'n_clusters': len(np.unique(labels)) - 1,
                'noise_ratio': np.mean(labels == -1)
            })
            
            # Log metrics individually for plotting
            wandb.log({
                f"hdbscan/minsize{min_size}"
                f"hdbscan/silhouette": sil_score,
                f"hdbscan/n_clusters": len(np.unique(labels)) - 1,
                f"hdbscan/noise_ratio": np.mean(labels == -1)
            })
    
    wandb.log({"hdbscan_metrics": metrics_table})
    return metrics

def estimate_dbscan_eps(latent_spaces, n_neighbors=5):
    """
    Estimate a good eps value for DBSCAN using k-distance graph
    Returns sorted distances and a recommended eps value
    """
    neigh = NearestNeighbors(n_neighbors=n_neighbors)
    neigh.fit(latent_spaces)
    distances, _ = neigh.kneighbors(latent_spaces)
    distances = np.sort(distances[:, -1])  # Get the distance to the kth nearest neighbor
    
    # Find the elbow point (point of maximum curvature)
    x = np.arange(len(distances))
    coords = np.vstack((x, distances)).T
    dists = np.sqrt(np.sum(np.diff(coords, axis=0)**2, axis=1))
    curvature = np.diff(dists)
    elbow_idx = np.argmax(curvature) + 1
    
    return distances, distances[elbow_idx]

def plot_kdistance_graph(distances, recommended_eps):
    """Create k-distance graph for DBSCAN parameter selection"""
    fig = plt.figure(figsize=(10, 5))
    plt.plot(np.arange(len(distances)), distances)
    plt.axhline(y=recommended_eps, color='r', linestyle='--', 
                label=f'Recommended eps: {recommended_eps:.3f}')
    plt.xlabel('Points')
    plt.ylabel('k-distance')
    plt.title('k-distance Graph')
    plt.legend()
    return fig

def run_dbscan_analysis(latent_spaces, eps_values=None, min_samples_values=None):
    """Run DBSCAN clustering analysis with different parameters"""
    # Always estimate eps and get distances for plotting
    distances, recommended_eps = estimate_dbscan_eps(latent_spaces)
    
    # Set default values if not provided
    if eps_values is None or min_samples_values is None:
        eps_range = np.linspace(recommended_eps * 0.5, recommended_eps * 1.5, 5)
        eps_values = eps_range if eps_values is None else eps_values
        min_samples_values = [3, 5, 10] if min_samples_values is None else min_samples_values
    
    # Create k-distance plot
    kdist_fig = plot_kdistance_graph(distances, recommended_eps)
    wandb.log({"dbscan/kdistance_graph": wandb.Image(kdist_fig)})
    plt.close(kdist_fig)
    
    metrics_table = wandb.Table(columns=["eps", "min_samples", "silhouette_score",
                                       "calinski_harabasz_score", "n_clusters",
                                       "noise_ratio", "cluster_visualization"])
    
    metrics = []
    for eps in eps_values:
        for min_samples in min_samples_values:
            dbscan = DBSCAN(eps=eps, min_samples=min_samples)
            labels = dbscan.fit_predict(latent_spaces)
            
            # Calculate metrics for non-noise points
            valid_points = labels != -1
            if np.any(valid_points) and len(np.unique(labels[valid_points])) > 1:
                sil_score = silhouette_score(
                    latent_spaces[valid_points],
                    labels[valid_points]
                )
                ch_score = calinski_harabasz_score(
                    latent_spaces[valid_points],
                    labels[valid_points]
                )
                
                # Create visualization
                fig = plot_clusters_2d_3d(
                    latent_spaces, 
                    labels, 
                    f"DBSCAN (eps={eps:.3f}, min_samples={min_samples})"
                )
                
                # Log to wandb
                metrics_table.add_data(
                    eps,
                    min_samples,
                    sil_score,
                    ch_score,
                    len(np.unique(labels)) - 1,  # Exclude noise cluster
                    np.mean(labels == -1),
                    wandb.Image(fig)
                )
                plt.close(fig)
                
                metrics.append({
                    'eps': eps,
                    'min_samples': min_samples,
                    'silhouette': sil_score,
                    'calinski_harabasz': ch_score,
                    'n_clusters': len(np.unique(labels)) - 1,
                    'noise_ratio': np.mean(labels == -1)
                })
                
                # Log metrics individually for plotting
                wandb.log({
                    f"dbscan/eps_{eps}_minsamples_{min_samples}/silhouette": sil_score,
                    f"dbscan/eps_{eps}_minsamples_{min_samples}/calinski_harabasz": ch_score,
                    f"dbscan/eps_{eps}_minsamples_{min_samples}/n_clusters": len(np.unique(labels)) - 1,
                    f"dbscan/eps_{eps}_minsamples_{min_samples}/noise_ratio": np.mean(labels == -1)
                })
    
    wandb.log({"dbscan_metrics": metrics_table})
    return metrics

def run_clustering_analysis(config):
    """Main function that runs all clustering analyses with wandb config parameters"""
    # Set up device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Create dataloader
    dataloader = create_dataloader(
        data_path=config.data_path,
        batch_size=config.batch_size,
        num_workers=config.num_workers
    )
    
    # Load model
    model = load_model(
        model_path=config.model_path,
        device=device,
        latent_dim=config.latent_dim
    )
    
    # Extract latent spaces
    latent_spaces = extract_latent_spaces(model, dataloader, device)
    
    # Scale the latent spaces for consistent distance metrics
    scaler = StandardScaler()
    scaled_latent_spaces = scaler.fit_transform(latent_spaces)
    
    # Run clustering analyses
    kmeans_metrics = run_kmeans_analysis(
        scaled_latent_spaces,
        config.kmeans_k_values
    )
    
    hdbscan_metrics = run_hdbscan_analysis(
        scaled_latent_spaces,
        config.hdbscan_min_sizes
    )
    
    dbscan_metrics = run_dbscan_analysis(
        scaled_latent_spaces,
        config.dbscan_eps_values if hasattr(config, 'dbscan_eps_values') else None,
        config.dbscan_min_samples if hasattr(config, 'dbscan_min_samples') else None
    )
    
    # Log final summary metrics
    wandb.run.summary.update({
        "best_kmeans_silhouette": max(m['silhouette'] for m in kmeans_metrics),
        "best_hdbscan_silhouette": max(m['silhouette'] for m in hdbscan_metrics),
        "best_dbscan_silhouette": max(m['silhouette'] for m in dbscan_metrics),
        "completion_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    })
    
    return {
        'kmeans_metrics': kmeans_metrics,
        'hdbscan_metrics': hdbscan_metrics,
        'dbscan_metrics': dbscan_metrics
    }