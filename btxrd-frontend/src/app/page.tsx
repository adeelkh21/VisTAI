'"'"'use client'"'"';

import React, { useState } from '"'"'react'"'"';
import { ImageUploadSection } from '"'"'@/components/vistai/ImageUploadSection'"'"';
import { ResultsPanel } from '"'"'@/components/vistai/ResultsPanel'"'"';
import { ChatPanel } from '"'"'@/components/vistai/ChatPanel'"'"';
import { Header } from '"'"'@/components/vistai/Header'"'"';

type AnalysisMode = '"'"'quick'"'"' | '"'"'full'"'"';

interface PredictionResult {
  class_name: string;
  confidence: number;
  probabilities: Record<string, number>;
}

export default function VistAI() {
  const [mode, setMode] = useState<AnalysisMode>('"'"'quick'"'"');
  const [file, setFile] = useState<File | null>(null);
  const [preview, setPreview] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<PredictionResult | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [showChat, setShowChat] = useState(false);

  const handleFileSelect = (selectedFile: File) => {
    setFile(selectedFile);
    setError(null);
    
    // Create preview
    const reader = new FileReader();
    reader.onload = (e) => setPreview(e.target?.result as string);
    reader.readAsDataURL(selectedFile);
  };

  const handleRunAnalysis = async () => {
    if (!file) {
      setError('"'"'Please select an image first'"'"');
      return;
    }

    setLoading(true);
    setError(null);

    try {
      const formData = new FormData();
      formData.append('"'"'file'"'"', file);

      const endpoint = mode === '"'"'quick'"'"'
        ? '"'"'/api/mobilenet/predict'"'"'
        : '"'"'/api/inference'"'"';

      const response = await fetch(endpoint, {
        method: '"'"'POST'"'"',
        body: formData,
      });

      if (!response.ok) {
        throw new Error(`API Error: ${response.statusText}`);
      }

      const data = await response.json();
      
      // Normalize response format
      const normalizedResult: PredictionResult = {
        class_name: data.class_name || data.class || '"'"'Unknown'"'"',
        confidence: data.confidence || 0,
        probabilities: data.probabilities || {},
      };

      setResult(normalizedResult);
    } catch (err) {
      setError(err instanceof Error ? err.message : '"'"'Analysis failed'"'"');
    } finally {
      setLoading(false);
    }
  };

  const handleFullAnalysis = async () => {
    if (!result) return;
    
    // Implement full analysis logic
    console.log('"'"'Running full analysis on:'"'"', result.class_name);
    // TODO: Connect to full analysis endpoint
  };

  const handleChatOpen = () => {
    setShowChat(true);
  };

  const handleChatClose = () => {
    setShowChat(false);
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-900 via-blue-900 to-slate-900">
      {/* Background blur effects */}
      <div className="fixed inset-0 overflow-hidden pointer-events-none">
        <div className="absolute top-0 left-1/4 w-96 h-96 bg-blue-500 rounded-full mix-blend-multiply filter blur-3xl opacity-20 animate-blob"></div>
        <div className="absolute top-1/3 right-1/4 w-96 h-96 bg-purple-500 rounded-full mix-blend-multiply filter blur-3xl opacity-20 animate-blob animation-delay-2000"></div>
        <div className="absolute bottom-0 left-1/2 w-96 h-96 bg-cyan-500 rounded-full mix-blend-multiply filter blur-3xl opacity-20 animate-blob animation-delay-4000"></div>
      </div>

      {/* Main content */}
      <div className="relative z-10">
        {/* Header */}
        <Header />

        {/* Main container */}
        <main className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
          {/* Mode selector */}
          <div className="flex justify-center mb-8">
            <div className="inline-flex rounded-xl bg-white/10 backdrop-blur-xl border border-white/20 p-1">
              <button
                onClick={() => {
                  setMode('"'"'quick'"'"');
                  setResult(null);
                }}
                className={`px-6 py-2 rounded-lg font-medium transition-all ${
                  mode === '"'"'quick'"'"'
                    ? '"'"'bg-blue-500 text-white shadow-lg shadow-blue-500/50'"'"'
                    : '"'"'text-white/70 hover:text-white'"'"'
                }`}
              >
                ?? Quick Classification
              </button>
              <button
                onClick={() => {
                  setMode('"'"'full'"'"');
                  setResult(null);
                }}
                className={`px-6 py-2 rounded-lg font-medium transition-all ${
                  mode === '"'"'full'"'"'
                    ? '"'"'bg-blue-500 text-white shadow-lg shadow-blue-500/50'"'"'
                    : '"'"'text-white/70 hover:text-white'"'"'
                }`}
              >
                ?? Full Analysis
              </button>
            </div>
          </div>

          {/* Content grid */}
          <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
            {/* Left column - Upload */}
            <div className="lg:col-span-1">
              <ImageUploadSection
                preview={preview}
                loading={loading}
                onFileSelect={handleFileSelect}
                onRunAnalysis={handleRunAnalysis}
              />
            </div>

            {/* Right column - Results */}
            <div className="lg:col-span-2">
              {error && (
                <div className="mb-6 p-4 rounded-xl bg-red-500/20 border border-red-500/40 text-red-200">
                  ?? {error}
                </div>
              )}

              {loading ? (
                <div className="space-y-4">
                  {[1, 2, 3].map((i) => (
                    <div
                      key={i}
                      className="h-12 bg-white/5 rounded-lg animate-pulse"
                    ></div>
                  ))}
                </div>
              ) : result ? (
                <ResultsPanel
                  result={result}
                  onFullAnalysis={handleFullAnalysis}
                  onChatOpen={handleChatOpen}
                />
              ) : (
                <div className="text-center py-16 text-white/50">
                  <p className="text-lg">Upload an X-ray image to begin analysis</p>
                </div>
              )}
            </div>
          </div>
        </main>
      </div>

      {/* Chat panel */}
      {result && showChat && (
        <ChatPanel
          detectedDisease={result.class_name}
          confidence={result.confidence}
          onClose={handleChatClose}
        />
      )}

      {/* Tailwind animations */}
      <style>{`
        @keyframes blob {
          0%, 100% { transform: translate(0, 0) scale(1); }
          33% { transform: translate(30px, -50px) scale(1.1); }
          66% { transform: translate(-20px, 20px) scale(0.9); }
        }
        .animate-blob {
          animation: blob 7s infinite;
        }
        .animation-delay-2000 {
          animation-delay: 2s;
        }
        .animation-delay-4000 {
          animation-delay: 4s;
        }
      `}</style>
    </div>
  );
}
