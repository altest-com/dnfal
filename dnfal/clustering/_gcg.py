import networkx as nx
from sklearn.neighbors import NearestNeighbors


def gcg_cluster(
    features,
    timestamps=None,
    distance_thr: float = 0.5,
    timestamp_thr: float = 0,
    edge_thr: float = 0.5
):
    n_features = len(features)
    dist_neigh = NearestNeighbors(radius=distance_thr)
    dist_neigh.fit(features)
    dist_graph = dist_neigh.radius_neighbors_graph(mode='distance')

    if timestamps is not None and timestamp_thr > 0:
        time_neigh = NearestNeighbors(radius=timestamp_thr)
        time_neigh.fit(timestamps)
        time_graph = time_neigh.radius_neighbors_graph(mode='connectivity')
        dist_graph = dist_graph.multiply(time_graph)
        dist_graph.eliminate_zeros()

    dist_graph = nx.from_scipy_sparse_matrix(dist_graph)

    # Dict of nodes that have not a cluster label assigned
    no_labels = {u: u for u in range(n_features)}

    # Init the clusters list with a random one-element cluster
    clusters = {0: [0]} # TODO: really do this random?

    # Remove node 0 form no labels list
    del no_labels[0]

    # Current growing cluster
    cur_label = 0

    node_boundary = nx.algorithms.boundary.node_boundary

    while True:
        cur_cluster = clusters[cur_label]
        boundary = node_boundary(dist_graph, cur_cluster, no_labels.keys())
        cluster_grow = False

        if len(boundary):
            scores = []
            for node in boundary:
                node_neighs = dist_graph[node]
                inside_edges = [
                    node_neighs[u]['weight'] for u in node_neighs
                    if u in cur_cluster
                ]
                scores.append((
                    len(inside_edges),
                    sum(inside_edges),
                    node
                ))

            best_score = sorted(scores, key=lambda s: (-s[0], s[1]))[0]
            if best_score[0] >= int(edge_thr * len(cur_cluster) + 1):
                best_node = best_score[2]
                clusters[cur_label].append(best_node)
                del no_labels[best_node]
                cluster_grow = True

        if not len(no_labels):
            break

        if not cluster_grow:
            # Increase current cluster label
            cur_label += 1
            # Take the next no labeled node and seed a new cluster with it
            next_node = no_labels.popitem()[0]
            clusters[cur_label] = [next_node]

            if not len(no_labels):
                break

    return list(clusters.values())