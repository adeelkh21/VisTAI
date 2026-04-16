'use client';

import React, { useState } from 'react';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { AlertCircle, CheckCircle2, Loader2, Upload } from 'lucide-react';

interface MobileNetPrediction {
  class_name: string;
  confidence: number;
  probabilities: Record<string, number>;
}

interface ClassColor {
  [key: string]: string;
}

const classColors: ClassColor = {
  'giant cell tumor': 'bg-red-100 text-red-900',
  'multiple osteochondromas': 'bg-orange-100 text-orange-900',
  'osteochondroma': 'bg-yellow-100 text-yellow-900',
  'osteofibroma': 'bg-green-100 text-green-900',
  'osteosarcoma': 'bg-red-100 text-red-900',
  'other bt': 'bg-blue-100 text-blue-900',
  'other mt': 'bg-purple-100 text-purple-900',
  'simple bone cyst': 'bg-cyan-100 text-cyan-900',
  'synovial osteochondroma': 'bg-indigo-100 text-indigo-900',
};

export default function MobileNetPanel() {
  const [file, setFile] = useState<File | null>(null);
  const [preview, setPreview] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const [prediction, setPrediction] = useState<MobileNetPrediction | null>(null);
  const [error, setError] = useState<string | null>(null);

  const handleFileSelect = (event: React.ChangeEvent<HTMLInputElement>) => {
    const selectedFile = event.target.files?.[0];
    if (selectedFile) {
      setFile(selectedFile);
      setError(null);

      // Create preview
      const reader = new FileReader();
      reader.onload = (e) => {
        setPreview(e.target?.result as string);
      };
      reader.readAsDataURL(selectedFile);
    }
  };

  const handlePrediction = async () => {
    if (!file) {
      setError('Please select an image first');
      return;
    }

    setLoading(true);
    setError(null);
    setPrediction(null);

    try {
      const formData = new FormData();
      formData.append('file', file);

      const response = await fetch('/api/mobilenet/predict', {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) {
        throw new Error('Prediction failed');
      }

      const data: MobileNetPrediction = await response.json();
      setPrediction(data);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to get prediction');
    } finally {
      setLoading(false);
    }
  };

  // Sort probabilities by descending confidence
  const sortedProbs = prediction
    ? Object.entries(prediction.probabilities)
        .sort(([, a], [, b]) => b - a)
        .slice(0, 5)
    : [];

  return (
    <div className="space-y-4">
      <Card>
        <CardHeader>
          <CardTitle>🚀 MobileNetV2 Lightweight Classification</CardTitle>
          <CardDescription>
            Fast AI-powered tumor classification optimized for edge devices
          </CardDescription>
        </CardHeader>
        <CardContent className="space-y-6">
          {/* Image Upload */}
          <div className="space-y-2">
            <label className="block text-sm font-medium">Upload X-ray Image</label>
            <div className="border-2 border-dashed rounded-lg p-8 text-center cursor-pointer hover:bg-gray-50 transition"
                 onClick={() => document.getElementById('image-input')?.click()}>
              {preview ? (
                <div className="space-y-2">
                  <img src={preview} alt="Preview" className="max-h-64 mx-auto rounded" />
                  <p className="text-sm text-gray-600">{file?.name}</p>
                </div>
              ) : (
                <div className="space-y-2">
                  <Upload className="mx-auto text-gray-400" size={32} />
                  <p className="text-sm text-gray-600">Click to upload or drag and drop</p>
                  <p className="text-xs text-gray-500">PNG, JPG, DICOM up to 16MB</p>
                </div>
              )}
              <input
                id="image-input"
                type="file"
                accept="image/*"
                onChange={handleFileSelect}
                className="hidden"
              />
            </div>
          </div>

          {/* Prediction Button */}
          <Button
            onClick={handlePrediction}
            disabled={!file || loading}
            className="w-full"
            size="lg"
          >
            {loading ? (
              <>
                <Loader2 className="mr-2 animate-spin" size={20} />
                Analyzing...
              </>
            ) : (
              'Run Classification'
            )}
          </Button>

          {/* Error Display */}
          {error && (
            <div className="flex gap-2 p-3 bg-red-50 border border-red-200 rounded-lg text-red-800 text-sm">
              <AlertCircle size={20} className="flex-shrink-0 mt-0.5" />
              <span>{error}</span>
            </div>
          )}

          {/* Prediction Results */}
          {prediction && (
            <div className="space-y-4">
              {/* Main Prediction */}
              <div className="p-4 bg-gradient-to-r from-blue-50 to-indigo-50 rounded-lg border border-blue-200">
                <div className="flex items-center gap-3 mb-3">
                  <CheckCircle2 className="text-green-600" size={24} />
                  <h3 className="font-semibold text-lg">Prediction Result</h3>
                </div>

                <div className="space-y-3">
                  {/* Predicted Class */}
                  <div>
                    <p className="text-xs text-gray-600 mb-1">Predicted Class</p>
                    <div
                      className={`inline-block px-4 py-2 rounded-lg font-semibold text-lg ${
                        classColors[prediction.class_name] || 'bg-gray-200 text-gray-900'
                      }`}
                    >
                      {prediction.class_name}
                    </div>
                  </div>

                  {/* Confidence Score */}
                  <div>
                    <p className="text-xs text-gray-600 mb-2">
                      Confidence: {(prediction.confidence * 100).toFixed(1)}%
                    </p>
                    <div className="w-full bg-gray-200 rounded-full h-3 overflow-hidden">
                      <div
                        className="bg-gradient-to-r from-green-400 to-blue-500 h-full transition-all"
                        style={{ width: `${prediction.confidence * 100}%` }}
                      />
                    </div>
                  </div>
                </div>
              </div>

              {/* Top Predictions */}
              <div>
                <h4 className="font-semibold text-sm mb-3">Top Predictions</h4>
                <div className="space-y-2">
                  {sortedProbs.map(([className, prob], idx) => (
                    <div key={className} className="flex items-center gap-3">
                      <span className="text-xs font-bold text-gray-500 w-6">
                        {idx + 1}.
                      </span>
                      <span className="flex-shrink-0 text-sm">{className}</span>
                      <div className="flex-grow bg-gray-100 rounded-full h-2">
                        <div
                          className="bg-blue-500 h-full rounded-full transition-all"
                          style={{ width: `${prob * 100}%` }}
                        />
                      </div>
                      <span className="text-xs font-semibold text-gray-700 w-12 text-right">
                        {(prob * 100).toFixed(1)}%
                      </span>
                    </div>
                  ))}
                </div>
              </div>

              {/* Feature */}
              <div className="p-3 bg-blue-50 rounded-lg border border-blue-100 text-xs text-blue-900">
                <p>
                  <strong>💡 Note:</strong> MobileNetV2 is a lightweight model optimized for edge
                  devices like NVIDIA Jetson Nano. It provides fast inference while maintaining good
                  accuracy.
                </p>
              </div>
            </div>
          )}
        </CardContent>
      </Card>
    </div>
  );
}
