import { useState, useEffect, useRef } from 'react';
import { query } from '../graphqlClient';
import DetectionCard from '../components/DetectionCard';

const RECENT_ACTIVITY_QUERY = `
  query ($limit: Int) {
    recentActivity(limit: $limit) {
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

const REFRESH_INTERVAL = 10000;

export default function Dashboard() {
  const [detections, setDetections] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const intervalRef = useRef(null);

  function fetchActivity() {
    query(RECENT_ACTIVITY_QUERY, { limit: 50 })
      .then((data) => {
        setDetections(data.recentActivity || []);
        setError(null);
      })
      .catch((err) => setError(err.message))
      .finally(() => setLoading(false));
  }

  useEffect(() => {
    fetchActivity();
    intervalRef.current = setInterval(fetchActivity, REFRESH_INTERVAL);
    return () => clearInterval(intervalRef.current);
  }, []);

  if (loading) return <div className="loading">Loading recent activity...</div>;
  if (error) return <div className="error-msg">Error: {error}</div>;

  return (
    <div>
      <div className="page-header">
        <h1>
          Recent Activity
          <span className="refresh-indicator">Auto-refreshes every 10s</span>
        </h1>
        <div className="subtitle">{detections.length} detections</div>
      </div>
      {detections.length === 0 ? (
        <div className="empty-msg">No recent activity</div>
      ) : (
        <div className="detection-grid">
          {detections.map((d) => (
            <DetectionCard key={d.detectionId} detection={d} />
          ))}
        </div>
      )}
    </div>
  );
}
