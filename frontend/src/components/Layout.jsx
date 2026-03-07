import { useState, useEffect } from 'react';
import { NavLink, useNavigate } from 'react-router-dom';
import { query } from '../graphqlClient';

const PERSONS_QUERY = `
  query { persons(limit: 100, offset: 0) { name } }
`;

export default function Layout({ children }) {
  const [persons, setPersons] = useState([]);
  const navigate = useNavigate();

  useEffect(() => {
    query(PERSONS_QUERY)
      .then((data) => setPersons(data.persons || []))
      .catch(() => {});
  }, []);

  function handlePersonSelect(e) {
    const name = e.target.value;
    if (name) {
      navigate(`/person/${encodeURIComponent(name)}`);
      e.target.value = '';
    }
  }

  return (
    <>
      <nav className="nav-bar">
        <span className="nav-title">Cameras</span>
        <NavLink to="/" className={({ isActive }) => isActive ? 'active' : ''}>
          Dashboard
        </NavLink>
        <NavLink to="/clusters" className={({ isActive }) => isActive ? 'active' : ''}>
          Clusters
        </NavLink>
        <div className="nav-persons">
          <select onChange={handlePersonSelect} defaultValue="">
            <option value="" disabled>Go to person...</option>
            {persons.map((p) => (
              <option key={p.name} value={p.name}>{p.name}</option>
            ))}
          </select>
        </div>
      </nav>
      <div className="page-container">
        {children}
      </div>
    </>
  );
}
