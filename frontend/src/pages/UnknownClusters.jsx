import { useState, useEffect } from 'react';
import { query } from '../graphqlClient';
import DetectionCard from '../components/DetectionCard';

const CLUSTERS_QUERY = `
  query {
    unknownClusters {
      clusterId
      detectionCount
      suggestedName
      confirmed
    }
  }
`;

const CLUSTER_DETECTIONS_QUERY = `
  query ($clusterId: String!) {
    clusterDetections(clusterId: $clusterId) {
      detectionId
      timestamp
      cameraId
      cameraLabel
      personName
      confidence
      faceCropUrl
      bodyCropUrl
      clusterId
      trackId
    }
  }
`;

const CONFIRM_MUTATION = `
  mutation ($clusterId: String!, $personName: String!) {
    confirmCluster(clusterId: $clusterId, personName: $personName) {
      clusterId
      confirmed
    }
  }
`;

const REJECT_MUTATION = `
  mutation ($clusterId: String!) {
    rejectClusterSuggestion(clusterId: $clusterId) {
      clusterId
    }
  }
`;

export default function UnknownClusters() {
  const [clusters, setClusters] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [expanded, setExpanded] = useState({});
  const [clusterDetections, setClusterDetections] = useState({});
  const [nameInputs, setNameInputs] = useState({});

  function fetchClusters() {
    setLoading(true);
    query(CLUSTERS_QUERY)
      .then((data) => {
        setClusters(data.unknownClusters || []);
        setError(null);
      })
      .catch((err) => setError(err.message))
      .finally(() => setLoading(false));
  }

  useEffect(() => {
    fetchClusters();
  }, []);

  function handleView(clusterId) {
    if (expanded[clusterId]) {
      setExpanded((prev) => ({ ...prev, [clusterId]: false }));
      return;
    }
    setExpanded((prev) => ({ ...prev, [clusterId]: true }));
    if (!clusterDetections[clusterId]) {
      query(CLUSTER_DETECTIONS_QUERY, { clusterId })
        .then((data) => {
          setClusterDetections((prev) => ({
            ...prev,
            [clusterId]: data.clusterDetections || [],
          }));
        })
        .catch(() => {});
    }
  }

  function handleConfirm(clusterId) {
    const name = nameInputs[clusterId];
    if (!name || !name.trim()) return;
    query(CONFIRM_MUTATION, { clusterId, personName: name.trim() })
      .then(() => fetchClusters())
      .catch((err) => alert('Confirm failed: ' + err.message));
  }

  function handleReject(clusterId) {
    query(REJECT_MUTATION, { clusterId })
      .then(() => fetchClusters())
      .catch((err) => alert('Reject failed: ' + err.message));
  }

  if (loading) return <div className="loading">Loading clusters...</div>;
  if (error) return <div className="error-msg">Error: {error}</div>;

  return (
    <div>
      <div className="page-header">
        <h1>Unknown Clusters</h1>
        <div className="subtitle">{clusters.length} clusters</div>
      </div>
      {clusters.length === 0 ? (
        <div className="empty-msg">No unknown clusters</div>
      ) : (
        <div className="cluster-list">
          {clusters.map((c) => (
            <div key={c.clusterId} className="cluster-row">
              <div className="cluster-info">
                <div className="cluster-id">Cluster {c.clusterId}</div>
                <div className="cluster-meta">
                  {c.detectionCount} detection{c.detectionCount !== 1 ? 's' : ''}
                  {c.suggestedName && ` — suggested: ${c.suggestedName}`}
                  {c.confirmed && ' (confirmed)'}
                </div>
              </div>
              <div className="cluster-actions">
                <button className="btn-view" onClick={() => handleView(c.clusterId)}>
                  {expanded[c.clusterId] ? 'Hide' : 'View'}
                </button>
                <input
                  placeholder={c.suggestedName || 'Person name'}
                  value={nameInputs[c.clusterId] || c.suggestedName || ''}
                  onChange={(e) =>
                    setNameInputs((prev) => ({ ...prev, [c.clusterId]: e.target.value }))
                  }
                />
                <button className="btn-confirm" onClick={() => handleConfirm(c.clusterId)}>
                  Confirm
                </button>
                <button className="btn-reject" onClick={() => handleReject(c.clusterId)}>
                  Reject
                </button>
              </div>
              {expanded[c.clusterId] && (
                <div className="cluster-detections">
                  {clusterDetections[c.clusterId] ? (
                    clusterDetections[c.clusterId].length > 0 ? (
                      <div className="detection-grid">
                        {clusterDetections[c.clusterId].map((d) => (
                          <DetectionCard key={d.detectionId} detection={d} />
                        ))}
                      </div>
                    ) : (
                      <div className="empty-msg">No detections in this cluster</div>
                    )
                  ) : (
                    <div className="loading">Loading detections...</div>
                  )}
                </div>
              )}
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
