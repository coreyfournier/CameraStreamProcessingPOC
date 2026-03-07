import { Routes, Route } from 'react-router-dom';
import Layout from './components/Layout';
import Dashboard from './pages/Dashboard';
import PersonDetail from './pages/PersonDetail';
import UnknownClusters from './pages/UnknownClusters';

export default function App() {
  return (
    <Layout>
      <Routes>
        <Route path="/" element={<Dashboard />} />
        <Route path="/person/:name" element={<PersonDetail />} />
        <Route path="/clusters" element={<UnknownClusters />} />
      </Routes>
    </Layout>
  );
}
