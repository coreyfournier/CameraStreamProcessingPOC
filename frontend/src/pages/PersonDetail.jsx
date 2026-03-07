import { useState, useEffect } from 'react';
import { useParams } from 'react-router-dom';
import { query } from '../graphqlClient';
import DetectionCard from '../components/DetectionCard';
import DateFilter from '../components/DateFilter';

const DETECTIONS_QUERY = `
  query ($personName: String, $startDate: String, $endDate: String, $limit: Int, $offset: Int) {
    detections(personName: $personName, startDate: $startDate, endDate: $endDate, limit: $limit, offset: $offset) {
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

export default function PersonDetail() {
  const { name } = useParams();
  const decodedName = decodeURIComponent(name);
  const [detections, setDetections] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [dateRange, setDateRange] = useState({ startDate: null, endDate: null });

  useEffect(() => {
    setLoading(true);
    query(DETECTIONS_QUERY, {
      personName: decodedName,
      startDate: dateRange.startDate,
      endDate: dateRange.endDate,
      limit: 200,
      offset: 0,
    })
      .then((data) => {
        setDetections(data.detections || []);
        setError(null);
      })
      .catch((err) => setError(err.message))
      .finally(() => setLoading(false));
  }, [decodedName, dateRange]);

  return (
    <div>
      <div className="page-header">
        <h1>{decodedName}</h1>
        <div className="subtitle">{detections.length} detections</div>
      </div>
      <DateFilter onChange={setDateRange} />
      {loading && <div className="loading">Loading detections...</div>}
      {error && <div className="error-msg">Error: {error}</div>}
      {!loading && !error && detections.length === 0 && (
        <div className="empty-msg">No detections found for this person</div>
      )}
      {!loading && !error && detections.length > 0 && (
        <div className="detection-grid">
          {detections.map((d) => (
            <DetectionCard key={d.detectionId} detection={d} />
          ))}
        </div>
      )}
    </div>
  );
}
