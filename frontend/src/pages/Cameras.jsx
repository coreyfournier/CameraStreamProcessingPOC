import { useState, useEffect } from 'react';

export default function Cameras() {
  const [cameras, setCameras] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [activeStream, setActiveStream] = useState(null);

  useEffect(() => {
    fetch('/api/cameras')
      .then((res) => {
        if (!res.ok) throw new Error(`HTTP ${res.status}`);
        return res.json();
      })
      .then((data) => {
        setCameras(data);
        setLoading(false);
      })
      .catch((err) => {
        setError(err.message);
        setLoading(false);
      });
  }, []);

  if (loading) return <div className="loading">Loading cameras...</div>;
  if (error) return <div className="error-msg">Error: {error}</div>;
  if (!cameras.length) return <div className="empty-msg">No cameras configured</div>;

  return (
    <div>
      <div className="page-header">
        <h1>Cameras</h1>
        <p className="subtitle">{cameras.length} camera(s) configured</p>
      </div>

      <div className="camera-grid">
        {cameras.map((cam) => (
          <div key={cam.id} className="camera-card">
            <div className="camera-preview">
              {activeStream === cam.id ? (
                <img
                  src={`${cam.stream_url}?t=${Date.now()}`}
                  alt={`${cam.label} live`}
                  className="camera-stream"
                />
              ) : (
                <div className="camera-placeholder">
                  <span>Stream paused</span>
                </div>
              )}
            </div>
            <div className="camera-info">
              <div className="camera-label">{cam.label}</div>
              <div className="camera-meta">
                <span className="camera-type">{cam.source_type}</span>
                <span className="camera-id">ID: {cam.id}</span>
              </div>
              <button
                className={activeStream === cam.id ? 'btn-stop' : 'btn-view'}
                onClick={() => setActiveStream(activeStream === cam.id ? null : cam.id)}
              >
                {activeStream === cam.id ? 'Stop' : 'View Live'}
              </button>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}
