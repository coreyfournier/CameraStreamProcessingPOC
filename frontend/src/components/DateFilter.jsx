import { useState } from 'react';

export default function DateFilter({ onChange }) {
  const [startDate, setStartDate] = useState('');
  const [endDate, setEndDate] = useState('');

  function handleFilter() {
    onChange({
      startDate: startDate || null,
      endDate: endDate || null,
    });
  }

  return (
    <div className="date-filter">
      <label>From</label>
      <input
        type="date"
        value={startDate}
        onChange={(e) => setStartDate(e.target.value)}
      />
      <label>To</label>
      <input
        type="date"
        value={endDate}
        onChange={(e) => setEndDate(e.target.value)}
      />
      <button onClick={handleFilter}>Filter</button>
    </div>
  );
}
