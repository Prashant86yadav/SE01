import React, { useState } from 'react';

const PromptflowClient: React.FC = () => {
  const [inputURL, setInputURL] = useState('');
  const [response, setResponse] = useState<any>(null);
  const [loading, setLoading] = useState(false);

  const handleSubmit = async () => {
    setLoading(true);
    try {
      const res = await fetch('http://localhost:8000/run', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ InputURL: inputURL }),
      });

      const data = await res.json();
      setResponse(data);
    } catch (err) {
      console.error('Error calling API:', err);
      setResponse({ error: 'Failed to fetch data' });
    } finally {
      setLoading(false);
    }
  };

  return (
    <div style={{ padding: '2rem', fontFamily: 'sans-serif' }}>
      <h2>PromptFlow API Client</h2>
      <input
        type="text"
        placeholder="Enter URL"
        value={inputURL}
        onChange={(e) => setInputURL(e.target.value)}
        style={{ width: '60%', padding: '8px' }}
      />
      <button onClick={handleSubmit} style={{ marginLeft: '1rem' }}>
        Run
      </button>

      {loading && <p>Running flow...</p>}

      {response && (
        <div style={{ marginTop: '1rem' }}>
          <h4>Response:</h4>
          <pre>{JSON.stringify(response, null, 2)}</pre>
        </div>
      )}
    </div>
  );
};

export default PromptflowClient;
