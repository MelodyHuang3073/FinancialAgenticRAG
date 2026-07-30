import React, { useState, useEffect } from 'react';
import Header from './components/Header';
import ChatTerminal from './components/ChatTerminal';

export default function App() {
  const [healthStatus,  setHealthStatus]  = useState(false);
  const [isLoading,     setIsLoading]     = useState(false);
  const [messages,      setMessages]      = useState([]);
  const [samplePrompts, setSamplePrompts] = useState([]);
  const [uploadedFiles, setUploadedFiles] = useState([]);

  const fetchUploadedFiles = () => {
    fetch('http://localhost:8000/api/uploaded-files')
      .then(r => r.json())
      .then(setUploadedFiles)
      .catch(console.error);
  };

  useEffect(() => {
    fetch('http://localhost:8000/api/health')
      .then(r => r.json())
      .then(() => setHealthStatus(true))
      .catch(() => setHealthStatus(false));

    fetch('http://localhost:8000/api/sample-prompts')
      .then(r => r.json())
      .then(setSamplePrompts)
      .catch(console.error);

    fetchUploadedFiles();
  }, []);

  const handleQuerySubmit = async (queryText) => {
    setIsLoading(true);
    try {
      const res = await fetch('http://localhost:8000/api/chat', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ query: queryText, max_iterations: 3 }),
      });
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      const data = await res.json();
      setMessages(prev => [...prev, { query: queryText, result: data }]);
    } catch (err) {
      alert(`問答失敗：${err.message}`);
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div style={{ display: 'flex', flexDirection: 'column', height: '100vh', background: '#fff' }}>
      <Header
        onNewChat={() => setMessages([])}
        healthStatus={healthStatus}
      />
      <div style={{ flex: 1, overflow: 'hidden', position: 'relative' }}>
        <ChatTerminal
          onQuerySubmit={handleQuerySubmit}
          isLoading={isLoading}
          messages={messages}
          samplePrompts={samplePrompts}
          uploadedFiles={uploadedFiles}
          onFileUploadSuccess={fetchUploadedFiles}
        />
      </div>
    </div>
  );
}
