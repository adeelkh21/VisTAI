'use client';

import React, { useState } from 'react';

export default function Home() {
  const [activeTab, setActiveTab] = useState<'mobilenet' | 'full'>('mobilenet');
  const [file, setFile] = useState<File | null>(null);
  const [preview, setPreview] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<any>(null);
  const [error, setError] = useState<string | null>(null);

  const handleFileSelect = (e: React.ChangeEvent<HTMLInputElement>) => {
    const f = e.target.files?.[0];
    if (f) {
      setFile(f);
      setError(null);
      const reader = new FileReader();
      reader.onload = (e) => setPreview(e.target?.result as string);
      reader.readAsDataURL(f);
    }
  };

  const runMobileNetPrediction = async () => {
    if (!file) {
      setError('Please select an image');
      return;
    }
    setLoading(true);
    setError(null);
    try {
      const formData = new FormData();
      formData.append('file', file);
      const res = await fetch('/api/mobilenet/predict', {
        method: 'POST',
        body: formData,
      });
      if (!res.ok) throw new Error('Prediction failed');
      setResult(await res.json());
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Error');
    } finally {
      setLoading(false);
    }
  };

  const runFullAnalysis = async () => {
    if (!file) {
      setError('Please select an image');
      return;
    }
    setLoading(true);
    setError(null);
    try {
      // For full analysis, first upload then run inference
      const uploadData = new FormData();
      uploadData.append('file', file);
      const uploadRes = await fetch('/api/upload', {
        method: 'POST',
        body: uploadData,
      });
      if (!uploadRes.ok) throw new Error('Upload failed');
      const uploadResult = await uploadRes.json();
      
      // Then run inference
      const inferRes = await fetch('/api/inference', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          image_id: uploadResult.image_id,
          intent: 'full',
        }),
      });
      if (!inferRes.ok) throw new Error('Inference failed');
      setResult(await inferRes.json());
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Error');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div style={{ minHeight: '100vh', backgroundColor: '#0a0e1a', color: '#f8fafc', fontFamily: 'system-ui, -apple-system, sans-serif' }}>
      {/* Header */}
      <nav style={{ borderBottom: '1px solid #1f2937', backgroundColor: '#0a0e1a', padding: '1rem' }}>
        <div style={{ maxWidth: '1200px', margin: '0 auto', display: 'flex', alignItems: 'center', gap: '1rem' }}>
          <div style={{ fontSize: '1.5rem', fontWeight: 'bold' }}>🏥 VistAI</div>
          <span style={{ fontSize: '0.75rem', color: '#64748b' }}>Bone Tumor Detection</span>
        </div>
      </nav>

      {/* Main Content */}
      <main style={{ maxWidth: '900px', margin: '0 auto', padding: '2rem 1rem' }}>
        {/* Tabs */}
        <div style={{ display: 'flex', gap: '1rem', marginBottom: '2rem' }}>
          <button
            onClick={() => setActiveTab('mobilenet')}
            style={{
              padding: '0.75rem 1.5rem',
              borderRadius: '0.5rem',
              border: 'none',
              cursor: 'pointer',
              backgroundColor: activeTab === 'mobilenet' ? '#3b82f6' : '#1f2937',
              color: '#f8fafc',
              fontWeight: activeTab === 'mobilenet' ? 'bold' : 'normal',
              fontSize: '1rem',
            }}
          >
            🚀 Quick Classification
          </button>
          <button
            onClick={() => setActiveTab('full')}
            style={{
              padding: '0.75rem 1.5rem',
              borderRadius: '0.5rem',
              border: 'none',
              cursor: 'pointer',
              backgroundColor: activeTab === 'full' ? '#3b82f6' : '#1f2937',
              color: '#f8fafc',
              fontWeight: activeTab === 'full' ? 'bold' : 'normal',
              fontSize: '1rem',
            }}
          >
            🔍 Full Analysis
          </button>
        </div>

        {/* Card */}
        <div style={{ backgroundColor: '#111827', border: '1px solid #1f2937', borderRadius: '0.75rem', padding: '2rem' }}>
          <h2 style={{ fontSize: '1.5rem', marginBottom: '1rem', fontWeight: 'bold' }}>
            {activeTab === 'mobilenet' ? '🚀 Fast Classification' : '🔍 Full Tumor Analysis'}
          </h2>

          {/* Upload Area */}
          <div
            onClick={() => document.getElementById('file-input')?.click()}
            style={{
              border: '2px dashed #1f2937',
              borderRadius: '0.5rem',
              padding: '2rem',
              textAlign: 'center',
              cursor: 'pointer',
              marginBottom: '1.5rem',
              backgroundColor: '#0a0e1a',
            }}
          >
            {preview ? (
              <div>
                <img src={preview} alt="Preview" style={{ maxHeight: '300px', marginBottom: '1rem', borderRadius: '0.5rem' }} />
                <p style={{ fontSize: '0.875rem', color: '#64748b' }}>{file?.name}</p>
              </div>
            ) : (
              <div>
                <p style={{ fontSize: '1.5rem', marginBottom: '0.5rem' }}>📸</p>
                <p style={{ color: '#94a3b8', marginBottom: '0.25rem' }}>Click to upload or drag image</p>
                <p style={{ fontSize: '0.75rem', color: '#64748b' }}>PNG, JPG up to 16MB</p>
              </div>
            )}
            <input
              id="file-input"
              type="file"
              accept="image/*"
              onChange={handleFileSelect}
              style={{ display: 'none' }}
            />
          </div>

          {/* Button */}
          <button
            onClick={activeTab === 'mobilenet' ? runMobileNetPrediction : runFullAnalysis}
            disabled={!file || loading}
            style={{
              width: '100%',
              padding: '0.75rem',
              borderRadius: '0.5rem',
              border: 'none',
              backgroundColor: !file || loading ? '#64748b' : '#3b82f6',
              color: '#f8fafc',
              cursor: !file || loading ? 'not-allowed' : 'pointer',
              fontWeight: 'bold',
              fontSize: '1rem',
              marginBottom: '1rem',
            }}
          >
            {loading ? '⏳ Analyzing...' : 'Run Analysis'}
          </button>

          {/* Error */}
          {error && (
            <div style={{ backgroundColor: '#7f1d1d', border: '1px solid #dc2626', borderRadius: '0.5rem', padding: '1rem', marginBottom: '1rem', color: '#fca5a5' }}>
              ⚠️ {error}
            </div>
          )}

          {/* Results */}
          {result && (
            <div style={{ backgroundColor: '#1e3a1f', border: '1px solid #22c55e', borderRadius: '0.5rem', padding: '1.5rem' }}>
              <h3 style={{ marginBottom: '1rem', fontWeight: 'bold' }}>✅ Results</h3>
              
              {result.class_name && (
                <div style={{ marginBottom: '1rem' }}>
                  <p style={{ fontSize: '0.875rem', color: '#94a3b8', marginBottom: '0.5rem' }}>Predicted Class</p>
                  <div style={{ padding: '0.75rem', backgroundColor: '#22c55e', color: '#1e3a1f', borderRadius: '0.5rem', fontWeight: 'bold', fontSize: '1.1rem', marginBottom: '1rem' }}>
                    {result.class_name}
                  </div>
                  <p style={{ fontSize: '0.875rem', color: '#94a3b8', marginBottom: '0.5rem' }}>
                    Confidence: {(result.confidence * 100).toFixed(1)}%
                  </p>
                  <div style={{ width: '100%', backgroundColor: '#064e3b', borderRadius: '0.25rem', height: '0.5rem', overflow: 'hidden' }}>
                    <div
                      style={{
                        width: `${result.confidence * 100}%`,
                        backgroundColor: '#22c55e',
                        height: '100%',
                        transition: 'width 0.3s',
                      }}
                    />
                  </div>
                </div>
              )}

              {result.probabilities && (
                <div>
                  <p style={{ fontSize: '0.875rem', color: '#94a3b8', marginBottom: '0.75rem', fontWeight: 'bold' }}>All Probabilities</p>
                  {Object.entries(result.probabilities)
                    .sort(([, a]: any, [, b]: any) => (b as number) - (a as number))
                    .slice(0, 5)
                    .map(([className, prob]: any) => (
                      <div key={className} style={{ marginBottom: '0.75rem' }}>
                        <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '0.25rem', fontSize: '0.875rem' }}>
                          <span>{className}</span>
                          <span>{(prob * 100).toFixed(1)}%</span>
                        </div>
                        <div style={{ width: '100%', backgroundColor: '#064e3b', borderRadius: '0.25rem', height: '0.25rem' }}>
                          <div
                            style={{
                              width: `${prob * 100}%`,
                              backgroundColor: '#3b82f6',
                              height: '100%',
                            }}
                          />
                        </div>
                      </div>
                    ))}
                </div>
              )}
            </div>
          )}
        </div>
      </main>
    </div>
  );
}
