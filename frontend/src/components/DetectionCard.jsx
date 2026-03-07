import { Link } from 'react-router-dom';

function timeAgo(timestamp) {
  const now = Date.now();
  const then = new Date(timestamp).getTime();
  const seconds = Math.floor((now - then) / 1000);
  if (seconds < 60) return `${seconds}s ago`;
  const minutes = Math.floor(seconds / 60);
  if (minutes < 60) return `${minutes} min ago`;
  const hours = Math.floor(minutes / 60);
  if (hours < 24) return `${hours}h ago`;
  const days = Math.floor(hours / 24);
  return `${days}d ago`;
}

export default function DetectionCard({ detection }) {
  const {
    personName,
    cameraLabel,
    timestamp,
    confidence,
    faceCropUrl,
    bodyCropUrl,
  } = detection;

  const imageUrl = faceCropUrl || bodyCropUrl;
  const isUnknown = !personName || personName === 'Unknown';
  const displayName = isUnknown ? 'Unknown' : personName;
  const confidencePct = confidence != null ? `${(confidence * 100).toFixed(0)}%` : null;

  const card = (
    <div className="detection-card">
      {imageUrl ? (
        <img
          className="card-image"
          src={imageUrl}
          alt={displayName}
          loading="lazy"
        />
      ) : (
        <div className="card-placeholder">No image</div>
      )}
      <div className="card-body">
        <div className={`card-name ${isUnknown ? 'unknown' : ''}`}>
          {displayName}
        </div>
        <div className="card-meta">
          {cameraLabel && <span>{cameraLabel}</span>}
          <span>{timeAgo(timestamp)}</span>
        </div>
        {confidencePct && (
          <div className="card-confidence">{confidencePct} confidence</div>
        )}
      </div>
    </div>
  );

  if (!isUnknown) {
    return (
      <Link to={`/person/${encodeURIComponent(personName)}`} style={{ textDecoration: 'none' }}>
        {card}
      </Link>
    );
  }

  return card;
}
